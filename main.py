from __future__ import annotations

"""
主入口：支持可配置 block_size 4/8、数据集感知的中频掩码以及可视化脚本共享的频域流水线。
- 使用 YAML 配置驱动实验，支持 --config CLI 传参。
- 基于 FrequencyTagger 进行频域标记，自动匹配全局能量/PSNR。
- 兼容原有 CIFAR-10 4x4 行为，并扩展至 CIFAR-10(8x8)、GTSRB、ImageNette。
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import ensure_dir, load_config
from src.data import build_dataloaders, build_datasets, build_pca_loader
from src.frequency import (
    FrequencyParams,
    FrequencyStats,
    build_pca_trigger,
    collect_mid_vectors,
    get_mid_freq_indices,
)
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
    lambda_align = float(freq_cfg.get("lambda_align", 1.0))
    use_smallest_eigvec_only = bool(freq_cfg.get("use_smallest_eigvec_only", False))
    channel_mode = freq_cfg.get("channel_mode", "Y")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    dataset_name = cfg["data"]["name"].lower()
    block_size = int(cfg["data"].get("block_size", 4))

    # ------------------------------
    # 2. 频域掩码 & PCA 统计路径
    # ------------------------------
    mask_cfg = cfg.get("pca", {}).get("mask", None)
    if mask_cfg is not None:
        mask = [tuple(m) for m in mask_cfg]
    else:
        mask = get_mid_freq_indices(dataset_name, block_size)
    print(f"Using mid-frequency mask size={len(mask)} for dataset={dataset_name}, block_size={block_size}")

    pca_path = Path(cfg["pca"]["save_path"])
    ensure_dir(pca_path.parent)

    # ------------------------------
    # 3. 构建或加载频域统计
    # ------------------------------
    if pca_path.exists():
        stats = FrequencyStats.load(pca_path)
        print(f"Loaded frequency stats from {pca_path}")
    else:
        print("Building frequency statistics...")

        base_loader = build_pca_loader(cfg)
        vectors = collect_mid_vectors(
            base_loader,
            mask=mask,
            block_size=block_size,
            max_blocks=cfg["data"]["pca_sample_blocks"],
            device=device,
        )
        stats = build_pca_trigger(
            vectors,
            k_tail=cfg["pca"]["k_tail"],
            seed=cfg["experiment"]["seed"],
            block_size=block_size,
            dataset_name=dataset_name,
            use_smallest_eigvec_only=use_smallest_eigvec_only,
        )
        stats.save(pca_path)
        print(f"Saved frequency stats to {pca_path}")

    freq_params = FrequencyParams(
        stats=stats,
        mask=mask,
        block_size=block_size,
        dataset_name=dataset_name,
        match_global_energy=True,
        base_block_size_for_energy=4,
        lambda_align=lambda_align,
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
    model_name = cfg["model"].get("name", "resnet18").lower()
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
    )


if __name__ == "__main__":
    main()
