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

    loader 通常来自 build_dataloaders 返回的 marked_test_loader：
      - 其样本为「非 target_class 的测试图像 + 频域增强（已预归一化）」
      - 标签保持为原始类，但这里我们只关心模型是否预测为 target_class
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

    注意：当前实现仍然是“随机 batch 对随机 batch”的统计，
    并非严格的“一张图像增强前后”的成对比较。
    如果后续需要更精细的分析，可以再单独写成对 loader。
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


def _standard_train_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
) -> Tuple[float, float]:
    """
    标准 supervised 训练的一个 epoch。

    loader 通常是 build_dataloaders 返回的 clean_train_loader，
    在当前设定下，它已经是「干净样本 + 频域增强样本」混合后的训练集（poisoned train），
    且所有图像均已预归一化。
    """

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)

    for step, (images, labels) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

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

        # ✅ 不必每 step 都更新 postfix，减少 tqdm 的 Python 开销
        if step % 20 == 0 or step == len(loader):
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    epoch_loss = total_loss / max(total_examples, 1)
    epoch_acc = total_correct / max(total_examples, 1)
    return epoch_loss, epoch_acc


def train_and_evaluate(
    model: torch.nn.Module,
    clean_train_loader,
    marked_train_loader,  # 目前不在训练 loop 中使用，只是为了兼容 main.py 的接口
    clean_test_loader,
    marked_test_loader,
    target_class: int,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
):
    """
    训练与评估主循环。

    - 训练：
        * 只使用 clean_train_loader（实际上是「干净 + 频域增强」混合后的 poisoned train）
        * 不再人为区分 clean / marked batch，也不做 oversampling
        * 训练过程与标准 CIFAR-10 分类完全一致

    - 评估：
        * clean_test_loader 上评估标准分类性能 (clean_loss, clean_acc)
        * marked_test_loader 上统计“频域标记样本被预测为 target_class 的比例”（marked_target_rate）
    """

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        # 标准训练一个 epoch
        train_loss, train_acc = _standard_train_epoch(
            model=model,
            loader=clean_train_loader,
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
