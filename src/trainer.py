from __future__ import annotations

import os
import shutil
from typing import Tuple, Iterable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from tqdm.auto import tqdm


def create_optimizer(
    params,
    name: str,
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
) -> optim.Optimizer:
    """
    根据名称创建优化器，支持:
      - "adam"
      - "sgd"
    其它名称会抛出异常。
    """
    name = name.lower()
    lr = float(lr)
    weight_decay = float(weight_decay)
    momentum = float(momentum)

    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=True,
        )

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
    return float(torch.cat(shifts, dim=0).mean().item())


def _standard_train_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[float, float]:
    """
    标准 supervised 训练的一个 epoch。
    使用 AMP（自动混合精度）加速。
    """

    model.train()
    use_amp = (scaler is not None) and (device.type == "cuda")

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)

    for step, (images, labels) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

        if step % 20 == 0 or step == len(loader):
            avg_loss = total_loss / total_examples
            avg_acc = total_correct / total_examples
            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    epoch_loss = total_loss / total_examples
    epoch_acc = total_correct / total_examples
    return epoch_loss, epoch_acc


def train_and_evaluate(
    model: torch.nn.Module,
    clean_train_loader,
    marked_train_loader,  # 目前不在训练 loop 中使用，只是为了兼容接口
    clean_test_loader,
    marked_test_loader,
    target_class: int,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    scheduler: Optional[LRScheduler] = None,
    use_amp: bool = True,
    exp_dir: str | None = None,
    dataset_name: str | None = None,
    model_name: str | None = None,
):
    """
    训练与评估主循环。

    - 训练：
        * 只使用 clean_train_loader（实际上是「干净 + 频域增强」混合后的 poisoned train）
        * 支持 AMP
        * 支持可选的 LR scheduler（如 Cosine）

    - 评估：
        * clean_test_loader 上评估标准分类性能 (clean_loss, clean_acc)
        * marked_test_loader 上统计“频域标记样本被预测为 target_class 的比例”（marked_target_rate）
    """

    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    log_f = None
    best_clean_asr995 = -1.0
    best_clean_asr100 = -1.0
    best_path_asr995 = None
    best_path_asr100 = None

    if exp_dir is not None:
        os.makedirs(exp_dir, exist_ok=True)
        log_f = open(os.path.join(exp_dir, "train.log"), "a")

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
            scaler=scaler,
        )

        # 干净测试集表现
        clean_loss, clean_acc = evaluate(model, clean_test_loader, device)

        # 频域标记测试集：统计预测为 target_class 的比例
        marked_rate, marked_count, marked_total = evaluate_marked_target_rate(
            model, marked_test_loader, target_class=target_class, device=device
        )
        marked_rate_pct = marked_rate * 100.0

        # 如果有 scheduler，这里 step 一下，并记录当前 lr
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{epochs} | "
            f"lr={current_lr:.5f} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"clean_loss={clean_loss:.4f}, clean_acc={clean_acc:.4f} | "
            f"marked_target_rate={marked_rate_pct:.4f}% ({marked_count}/{marked_total})"
        )

        if log_f is not None:
            log_f.write(
                f"[Epoch {epoch}] clean_acc={clean_acc:.4f}, asr={marked_rate:.4f}, train_acc={train_acc:.4f}\n"
            )
            log_f.flush()

        if exp_dir is not None and dataset_name and model_name:
            if marked_rate >= 0.995 and clean_acc > best_clean_asr995:
                best_clean_asr995 = clean_acc
                filename = f"{dataset_name}_{model_name}_asr{marked_rate:.3f}_ba{clean_acc:.3f}.pth"
                save_path = os.path.join(exp_dir, filename)
                torch.save(model.state_dict(), save_path)
                alias_path = os.path.join(exp_dir, "best_asr995_clean.pth")
                shutil.copyfile(save_path, alias_path)
                best_path_asr995 = save_path

            if marked_rate == 1.0 and clean_acc > best_clean_asr100:
                best_clean_asr100 = clean_acc
                filename = f"{dataset_name}_{model_name}_asr{marked_rate:.3f}_ba{clean_acc:.3f}.pth"
                save_path = os.path.join(exp_dir, filename)
                torch.save(model.state_dict(), save_path)
                alias_path = os.path.join(exp_dir, "best_asr100_clean.pth")
                shutil.copyfile(save_path, alias_path)
                best_path_asr100 = save_path

    if log_f is not None:
        summary_lines = ["\n===== Best Model Summary =====\n"]
        if best_path_asr995 is not None:
            summary_lines.append(
                f"ASR>=99.5 best clean_acc={best_clean_asr995:.4f} saved at {best_path_asr995}"
            )
        else:
            summary_lines.append("No checkpoint reached ASR>=99.5%.")

        if best_path_asr100 is not None:
            summary_lines.append(
                f"ASR=100% best clean_acc={best_clean_asr100:.4f} saved at {best_path_asr100}"
            )
        else:
            summary_lines.append("No checkpoint reached ASR=100%.")

        log_f.write("\n".join(summary_lines) + "\n")
        log_f.close()
