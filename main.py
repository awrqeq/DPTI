from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import ensure_dir, load_config
from src.data import build_dataloaders, build_datasets
from src.frequency import FrequencyParams, FrequencyStats, build_pca_trigger, collect_mid_vectors
from src.model import build_resnet18
from src.trainer import create_optimizer, train_and_evaluate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def main():
    # 加载配置
    cfg = load_config("configs/config.yaml")

    # 随机种子 & 设备
    set_seed(cfg["experiment"]["seed"])
    device = resolve_device(cfg["experiment"]["device"])
    print(f"Using device: {device}")

    # cuDNN 加速
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    # 频域掩码和 PCA 统计路径
    mask = np.array(cfg["pca"]["mask"], dtype=np.int32)
    pca_path = Path(cfg["pca"]["save_path"])
    ensure_dir(pca_path.parent)

    # 构建或加载频域统计
    if pca_path.exists():
        stats = FrequencyStats.load(pca_path)
        print(f"Loaded frequency stats from {pca_path}")
    else:
        print("Building frequency statistics...")
        base_ds = datasets.CIFAR10(
            root=cfg["data"]["root"],
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        base_loader = DataLoader(
            base_ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=True,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
        )
        vectors = collect_mid_vectors(
            base_loader,
            mask=mask,
            sample_blocks=cfg["data"]["pca_sample_blocks"],
            device=device,
        )
        stats = build_pca_trigger(
            vectors,
            k_tail=cfg["pca"]["k_tail"],
            seed=cfg["experiment"]["seed"],
        )
        stats.save(pca_path)
        print(f"Saved frequency stats to {pca_path}")

    freq_params = FrequencyParams(stats=stats, mask=mask)

    # 构建数据集和 dataloader（离线投毒 + 预归一化）
    datasets_bundle = build_datasets(cfg, freq_params)
    (
        clean_train_loader,
        marked_train_loader,
        clean_test_loader,
        marked_test_loader,
    ) = build_dataloaders(cfg, datasets_bundle)

    # 构建模型
    model = build_resnet18(num_classes=10).to(device)

    # 从 config 中读取训练相关配置
    train_cfg = cfg["train"]
    epochs = int(train_cfg.get("epochs", 100))

    # 构建优化器（支持 adam / sgd）
    optimizer = create_optimizer(
        model.parameters(),
        name=train_cfg.get("optimizer", "sgd"),
        lr=float(train_cfg.get("lr", 0.1)),
        weight_decay=float(train_cfg.get("weight_decay", 5e-4)),
        momentum=float(train_cfg.get("momentum", 0.9)),
    )

    # 构建学习率调度器（目前支持 cosine / none）
    scheduler_name = str(train_cfg.get("scheduler", "none")).lower()
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(train_cfg.get("eta_min", 1e-4)),
        )
        print(f"Using CosineAnnealingLR: T_max={epochs}, eta_min={train_cfg.get('eta_min', 1e-4)}")
    else:
        scheduler = None
        print("No LR scheduler is used (scheduler=none)")

    # 确保 checkpoint 目录存在
    ensure_dir(cfg["log"]["ckpt_dir"])

    # 开始训练与评估（AMP 在 trainer 里控制）
    train_and_evaluate(
        model=model,
        clean_train_loader=clean_train_loader,
        marked_train_loader=marked_train_loader,
        clean_test_loader=clean_test_loader,
        marked_test_loader=marked_test_loader,
        target_class=int(cfg["data"]["target_class"]),
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        scheduler=scheduler,
        use_amp=True,
    )


if __name__ == "__main__":
    main()
