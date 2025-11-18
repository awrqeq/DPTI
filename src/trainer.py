from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def create_optimizer(
    params,
    name: str,
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
) -> optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"Unsupported optimizer: {name}")


@torch.no_grad()
def evaluate_marked_target_rate(model: torch.nn.Module, loader, target_class: int, device: torch.device) -> Tuple[float, int, int]:
    """评估频域标记样本被预测为目标类别的比例。"""

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


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    print_every: int = 50,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for step, (images, labels) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)

        if print_every and step % print_every == 0:
            avg_loss = total_loss / total_examples
            avg_acc = total_correct / total_examples
            print(f"Step {step}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")
    avg_loss = total_loss / max(total_examples, 1)
    avg_acc = total_correct / max(total_examples, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> Tuple[float, float]:
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
def representation_shift(model: torch.nn.Module, clean_loader, enhanced_loader, device: torch.device) -> float:
    """Compute average L2 shift between logits on clean vs enhanced images."""
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
    print_every: int = 50,
):
    """交替训练干净样本与频域标记样本，并在每个 epoch 双重评估。"""

    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        global_step = 0

        for loader_name, loader in (("clean", clean_train_loader), ("marked", marked_train_loader)):
            if len(loader.dataset) == 0:
                continue
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * labels.size(0)
                total_correct += (outputs.argmax(dim=1) == labels).sum().item()
                total_examples += labels.size(0)
                global_step += 1

                if print_every and global_step % print_every == 0:
                    avg_loss = total_loss / max(total_examples, 1)
                    avg_acc = total_correct / max(total_examples, 1)
                    print(
                        f"Step {global_step} ({loader_name}): "
                        f"loss={avg_loss:.4f}, acc={avg_acc:.4f}"
                    )

        train_loss = total_loss / max(total_examples, 1)
        train_acc = total_correct / max(total_examples, 1)

        clean_loss, clean_acc = evaluate(model, clean_test_loader, device)
        marked_rate, marked_count, marked_total = evaluate_marked_target_rate(
            model, marked_test_loader, target_class=target_class, device=device
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"clean_loss={clean_loss:.4f}, clean_acc={clean_acc:.4f}, "
            f"marked_target_rate={marked_rate:.4f} ({marked_count}/{marked_total})"
        )
