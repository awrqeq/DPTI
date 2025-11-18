from __future__ import annotations

from typing import Tuple, Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm


def create_optimizer(
    params,
    name: str,
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
) -> optim.Optimizer:
    name = name.lower()
    lr = float(lr)
    weight_decay = float(weight_decay)
    momentum = float(momentum)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"Unsupported optimizer: {name}")


@torch.no_grad()
def evaluate_marked_target_rate(
    model: torch.nn.Module,
    loader: Iterable,
    target_class: int,
    device: torch.device,
) -> Tuple[float, int, int]:
    """
    评估“频域标记”样本被预测为目标类别的比例。
    返回:
      rate: [0,1] 之间的比例
      count_target: 被预测为 target_class 的样本数
      total: 总样本数
    """
    model.eval()
    total = 0
    count_target = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        total += preds.size(0)
        count_target += (preds == target_class).sum().item()
    rate = count_target / total if total > 0 else 0.0
    return rate, count_target, total


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
) -> Tuple[float, float]:
    """
    在给定数据集上评估平均 loss 与 accuracy。
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)
    avg_loss = total_loss / max(total_examples, 1)
    avg_acc = total_correct / max(total_examples, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def representation_shift(
    model: torch.nn.Module,
    clean_loader: Iterable,
    enhanced_loader: Iterable,
    device: torch.device,
) -> float:
    """
    计算 clean vs enhanced 图像在 logits 空间的平均 L2 差异。
    """
    model.eval()
    shifts = []
    for (img_clean, _), (img_enh, _) in zip(clean_loader, enhanced_loader):
        img_clean = img_clean.to(device, non_blocking=True)
        img_enh = img_enh.to(device, non_blocking=True)
        logits_clean = model(img_clean)
        logits_enh = model(img_enh)
        diff = (logits_enh - logits_clean).pow(2).sum(dim=1).sqrt()
        shifts.append(diff.cpu())
    if not shifts:
        return 0.0
    all_shift = torch.cat(shifts, dim=0)
    return float(all_shift.mean().item())


def _mixed_train_epoch(
    model: torch.nn.Module,
    clean_loader: Iterable,
    marked_loader: Iterable,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> Tuple[float, float]:
    """
    使用“混合 batch（clean + marked）”的方式训练一个 epoch。
    这是参考后门攻防论文中常见的训练模式：
      每个 step 同时包含干净样本与带频域标记的样本，
      避免 BN 统计量被某一类数据单侧污染。
    """

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    clean_iter = iter(clean_loader)
    marked_iter = iter(marked_loader)

    # 训练步数取两者中较大的那个
    num_steps = max(len(clean_loader), len(marked_loader))
    pbar = tqdm(range(num_steps), desc=f"Epoch {epoch}/{epochs} [train]", leave=False)

    for _ in pbar:
        try:
            clean_images, clean_labels = next(clean_iter)
        except StopIteration:
            clean_iter = iter(clean_loader)
            clean_images, clean_labels = next(clean_iter)

        try:
            marked_images, marked_labels = next(marked_iter)
        except StopIteration:
            marked_iter = iter(marked_loader)
            marked_images, marked_labels = next(marked_iter)

        # 拼接 clean 与 marked，构成一个混合 batch
        images = torch.cat([clean_images, marked_images], dim=0).to(device, non_blocking=True)
        labels = torch.cat([clean_labels, marked_labels], dim=0).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

        avg_loss = total_loss / max(total_examples, 1)
        avg_acc = total_correct / max(total_examples, 1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    epoch_loss = total_loss / max(total_examples, 1)
    epoch_acc = total_correct / max(total_examples, 1)
    return epoch_loss, epoch_acc


def train_and_evaluate(
    model: torch.nn.Module,
    clean_train_loader,
    marked_train_loader,
    clean_test_loader,
    marked_test_loader,
    target_class: int,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
):
    """
    训练与评估主循环。

    - 训练：每个 epoch 使用混合 batch（clean + marked）
    - 评估：
        * clean_test_loader 上评估标准分类性能
        * marked_test_loader 上统计“频域标记样本被预测为 target_class 的比例”
    """

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        # 训练一个 epoch（混合 batch）
        train_loss, train_acc = _mixed_train_epoch(
            model=model,
            clean_loader=clean_train_loader,
            marked_loader=marked_train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=epochs,
        )

        # 干净测试集表现
        clean_loss, clean_acc = evaluate(model, clean_test_loader, device)

        # 频域标记测试集：统计预测为 target_class 的比例
        marked_rate, marked_count, marked_total = evaluate_marked_target_rate(
            model, marked_test_loader, target_class=target_class, device=device
        )
        marked_rate_pct = marked_rate * 100.0

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"clean_loss={clean_loss:.4f}, clean_acc={clean_acc:.4f} | "
            f"marked_target_rate={marked_rate_pct:.4f}% ({marked_count}/{marked_total})"
        )
