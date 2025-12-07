from __future__ import annotations

"""
主入口：支持可配置 block_size 4/8、数据集感知的中频掩码以及可视化脚本共享的频域流水线。
- 使用 YAML 配置驱动实验，支持 --config CLI 传参。
- 基于 FrequencyTagger 进行频域标记，自动匹配全局能量/PSNR。
- 兼容原有 CIFAR-10 4x4 行为，并扩展至 CIFAR-10(8x8)、GTSRB、ImageNette。
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import ensure_dir, load_config, resolve_pca_stats_path
from src.data import build_dataloaders, build_datasets, build_pca_loader
from src.frequency import FrequencyParams, FrequencyStats, gen_mask_by_sum, get_mid_freq_indices
from src.model import build_densenet121, build_resnet18
from src.trainer import create_optimizer, train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-driven frequency tagging training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径 (YAML)",
    )
    return parser.parse_args()


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
    args = parse_args()

    # ------------------------------
    # 1. 配置 / 随机种子 / 设备
    # ------------------------------
    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = resolve_device(cfg["experiment"]["device"])
    print(f"Using device: {device}")

    freq_cfg = cfg.get("frequency", {})
    use_smallest_eigvec_only = bool(freq_cfg.get("use_smallest_eigvec_only", False))
    if "channel_mode" not in freq_cfg:
        raise KeyError("frequency.channel_mode is required and must be one of: Y / UV / YUV")
    channel_mode = str(freq_cfg["channel_mode"]).upper()

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    dataset_name = cfg["data"]["name"].lower()
    block_size = int(cfg["data"].get("block_size", 8))
    model_name = cfg["model"].get("name", "resnet18").lower()

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    exp_root = "experiments"
    exp_name = f"{dataset_name}_{model_name}_{timestamp}"
    exp_dir = os.path.join(exp_root, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    # ------------------------------
    # 2. 频域掩码 & PCA 统计路径
    # ------------------------------
    pca_cfg = cfg.get("pca", {})
    if "mask_sum_min" in pca_cfg and "mask_sum_max" in pca_cfg:
        mask = gen_mask_by_sum(
            block_size,
            int(pca_cfg["mask_sum_min"]),
            int(pca_cfg["mask_sum_max"]),
            bool(pca_cfg.get("mask_exclude_dc", True)),
        )
    else:
        mask = get_mid_freq_indices(dataset_name, block_size)
    print(f"Using mid-frequency mask size={len(mask)} for dataset={dataset_name}, block_size={block_size}")

    pca_path = resolve_pca_stats_path(
        cfg,
        dataset_name=dataset_name,
        block_size=block_size,
        channel_mode=channel_mode,
        model_name=model_name,
    )
    ensure_dir(pca_path.parent)

    # ------------------------------
    # 3. 构建或加载频域统计
    # ------------------------------
    if pca_path.exists():
        stats = FrequencyStats.load(pca_path)
        print(f"Loaded frequency stats from {pca_path}")
    else:
        raise FileNotFoundError(
            f"Stats {pca_path} not found. Have you run pca_stats.py with channel_mode={channel_mode}?"
        )

    freq_params = FrequencyParams(
        stats=stats,
        mask=mask,
        block_size=block_size,
        dataset_name=dataset_name,
        channel_mode=channel_mode,
    )

    # ------------------------------
    # 4. 构建数据集 + DataLoader
    # ------------------------------
    datasets_bundle = build_datasets(cfg, freq_params)
    (
        clean_train_loader,
        marked_train_loader,
        clean_test_loader,
        marked_test_loader,
    ) = build_dataloaders(cfg, datasets_bundle)

    # ------------------------------
    # 5. 构建模型（支持 resnet18 / densenet121）
    # ------------------------------
    num_classes = int(cfg["model"].get("num_classes", 10))
    img_size = int(cfg["data"].get("img_size", 32))

    if model_name == "resnet18":
        model = build_resnet18(num_classes=num_classes, img_size=img_size).to(device)
    elif model_name == "densenet121":
        model = build_densenet121(num_classes=num_classes, cifar_like=(img_size <= 64)).to(device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # ------------------------------
    # 6. 优化器 & 学习率调度器 (SGD + Cosine)
    # ------------------------------
    train_cfg = cfg["train"]
    epochs = int(train_cfg.get("epochs", 100))

    optimizer = create_optimizer(
        model.parameters(),
        name=train_cfg.get("optimizer", "sgd"),
        lr=float(train_cfg.get("lr", 0.1)),
        weight_decay=float(train_cfg.get("weight_decay", 5e-4)),
        momentum=float(train_cfg.get("momentum", 0.9)),
    )

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

    # ------------------------------
    # 7. 日志 / ckpt 目录
    # ------------------------------
    ensure_dir(cfg["log"]["ckpt_dir"])

    # ------------------------------
    # 8. 训练 & 评估（AMP inside trainer）
    # ------------------------------
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
        use_amp=bool(train_cfg.get("amp", True)),
        exp_dir=exp_dir,
        dataset_name=dataset_name,
        model_name=model_name,
    )


if __name__ == "__main__":
    main()
