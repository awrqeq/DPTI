from __future__ import annotations

"""
离线生成 PCA 统计文件的脚本，文件名包含通道模式，避免不同 channel_mode 共用同一个 pkl。
"""

import argparse
import random

import numpy as np
import torch

from src.config import ensure_dir, load_config, resolve_pca_stats_path
from src.data import build_pca_loader
from src.frequency import build_pca_trigger, collect_mid_vectors, get_mid_freq_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PCA stats for frequency tagging")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["experiment"]["seed"])
    device = resolve_device(cfg["experiment"]["device"])
    print(f"Using device: {device}")

    dataset_name = cfg["data"]["name"].lower()
    block_size = int(cfg["data"].get("block_size", 4))
    model_name = str(cfg.get("model", {}).get("name", "")).lower() or None

    freq_cfg = cfg.get("frequency", {})
    if "channel_mode" not in freq_cfg:
        raise KeyError("frequency.channel_mode is required and must be one of: Y / UV / YUV")
    channel_mode = str(cfg["frequency"]["channel_mode"]).upper()
    use_smallest_eigvec_only = bool(freq_cfg.get("use_smallest_eigvec_only", False))

    mask_cfg = cfg.get("pca", {}).get("mask", None)
    mask = [tuple(m) for m in mask_cfg] if mask_cfg else get_mid_freq_indices(dataset_name, block_size)
    print(f"Using mid-frequency mask size={len(mask)} for dataset={dataset_name}, block_size={block_size}")

    pca_path = resolve_pca_stats_path(
        cfg,
        dataset_name=dataset_name,
        block_size=block_size,
        channel_mode=channel_mode,
        model_name=model_name,
    )
    ensure_dir(pca_path.parent)

    if pca_path.exists():
        print(f"PCA stats already exist at {pca_path}, skipping rebuild.")
        return

    print(f"Saving PCA stats to: {pca_path.name}")
    base_loader = build_pca_loader(cfg)
    apply_image_format = cfg.get("pca", {}).get("apply_image_format", True)

    vectors = collect_mid_vectors(
        base_loader,
        mask=mask,
        block_size=block_size,
        max_blocks=cfg["data"]["pca_sample_blocks"],
        device=device,
        channel_mode=channel_mode,
        data_cfg=cfg["data"],
        apply_image_format=apply_image_format,
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
    print(f"Saved PCA stats to {pca_path}")


if __name__ == "__main__":
    main()
