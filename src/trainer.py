from __future__ import annotations

import random
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _mixed_batch_iterator(
    clean_loader: Iterable,
    marked_loader: Iterable,
    marked_ratio: float,
) -> Iterable[Tuple[torch.Tensor, torch.Tensor, str]]:
    """按照 marked_ratio 概率从 clean/marked loader 中抽取 batch。"""

    clean_iter = iter(clean_loader)
    marked_iter = iter(marked_loader)
    clean_done = False
    marked_done = False

    while not (clean_done and marked_done):
        use_marked = random.random() < marked_ratio

        if use_marked and not marked_done:
            try:
                images, labels = next(marked_iter)
                yield images, labels, "marked"
                continue
            except StopIteration:
                marked_done = True

        if not clean_done:
            try:
                images, labels = next(clean_iter)
                yield images, labels, "clean"
                continue
            except StopIteration:
                clean_done = True

        # 当某一侧耗尽后，继续消耗另一侧剩余数据
        if not marked_done:
            try:
                images, labels = next(marked_iter)
                yield images, labels, "marked"
                continue
            except StopIteration:
                marked_done = True
        if not clean_done:
            try:
                images, labels = next(clean_iter)
                yield images, labels, "clean"
                continue
            except StopIteration:
                clean_done = True


@torch.no_grad()
def evaluate_clean(model: torch.nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    """干净测试集评估。"""

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
def evaluate_marked(
    model: torch.nn.Module,
    loader,
    target_class: int,
    device: torch.device,
) -> Tuple[float, float, int, int]:
    """标记测试集评估，返回平均目标置信度与命中率。"""

    model.eval()
    total = 0
    count_target = 0
    sum_confidence = 0.0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        total += preds.size(0)
        count_target += (preds == target_class).sum().item()
        sum_confidence += probs[:, target_class].sum().item()
    rate = count_target / total if total > 0 else 0.0
    avg_conf = sum_confidence / total if total > 0 else 0.0
    return rate, avg_conf, count_target, total


@torch.no_grad()
def representation_shift(model: torch.nn.Module, clean_loader, enhanced_loader, device: torch.device) -> float:
    """Compute average L2 shift between logits on clean vs enhanced images."""
    model.eval()
    shifts = []
    for (img_clean, _), (img_enh, _) in zip(clean_loader, enhanced_loader):
        img_clean = img_clean.to(device, non_blocking=True)
        img_enh = img_enh.to(device, non_blocking=True)
        min_bs = min(img_clean.size(0), img_enh.size(0))
        if min_bs == 0:
            continue
        if img_clean.size(0) != min_bs:
            img_clean = img_clean[:min_bs]
        if img_enh.size(0) != min_bs:
            img_enh = img_enh[:min_bs]

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
    marked_ratio: float | None = None,
):
    """混合 batch 训练并在每个 epoch 执行双评估。"""

    criterion = nn.CrossEntropyLoss()
    target_class = int(target_class)
    if marked_ratio is None:
        total_train = len(clean_train_loader.dataset) + len(marked_train_loader.dataset)
        marked_ratio = (
            len(marked_train_loader.dataset) / total_train if total_train > 0 else 0.0
        )
    marked_ratio = float(marked_ratio)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        clean_seen = 0
        marked_seen = 0

        mixed_iter = _mixed_batch_iterator(clean_train_loader, marked_train_loader, marked_ratio)
        total_steps = len(clean_train_loader) + len(marked_train_loader)
        prog = tqdm(
            mixed_iter,
            total=total_steps,
            desc=f"Epoch {epoch}/{epochs} [train]",
            leave=False,
        )

        for step, (images, labels, tag) in enumerate(prog, 1):
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
            if tag == "clean":
                clean_seen += batch_size
            else:
                marked_seen += batch_size

            if print_every and step % print_every == 0:
                avg_loss = total_loss / max(total_examples, 1)
                avg_acc = total_correct / max(total_examples, 1)
                prog.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    acc=f"{avg_acc:.4f}",
                    clean=f"{clean_seen}/{total_examples}",
                    marked=f"{marked_seen}/{total_examples}",
                )

        train_loss = total_loss / max(total_examples, 1)
        train_acc = total_correct / max(total_examples, 1)

        # 评估阶段
        model.eval()
        clean_loss, clean_acc = evaluate_clean(model, clean_test_loader, device)
        marked_rate, marked_conf, marked_count, marked_total = evaluate_marked(
            model, marked_test_loader, target_class=target_class, device=device
        )
        rep_shift = representation_shift(model, clean_test_loader, marked_test_loader, device)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"clean_loss={clean_loss:.4f}, clean_acc={clean_acc:.4f}, "
            f"marked_target_rate={marked_rate:.4f} ({marked_count}/{marked_total}), "
            f"marked_confidence={marked_conf:.4f}, rep_shift={rep_shift:.4f}"
        )
