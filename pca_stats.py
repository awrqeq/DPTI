from __future__ import annotations

"""
Offline helper to build PCA stats files (Y channel only).
Sampling is balanced per class on a per-image basis (see data.pca_sample_images). Filenames encode dataset/model/block size.
"""

import argparse
import random

import numpy as np
import torch

from src.config import ensure_dir, load_config, resolve_pca_stats_path
from src.datasets import build_pca_loader
from src.frequency import build_pca_trigger, collect_mid_vectors
from src.mask_utils import mask_from_pca_cfg


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
    block_size = int(cfg["data"].get("block_size", 8))
    model_name = str(cfg.get("model", {}).get("name", "")).lower() or None

    pca_cfg = cfg.get("pca", {})
    mask = mask_from_pca_cfg(block_size, pca_cfg, dataset_name=dataset_name)
    print(f"Using mid-frequency mask size={len(mask)} for dataset={dataset_name}, block_size={block_size}")

    pca_path = resolve_pca_stats_path(
        cfg,
        dataset_name=dataset_name,
        block_size=block_size,
        model_name=model_name,
    )
    ensure_dir(pca_path.parent)

    if pca_path.exists():
        print(f"PCA stats already exist at {pca_path}, skipping rebuild.")
        return

    print(f"Saving PCA stats to: {pca_path.name}")
    base_loader = build_pca_loader(cfg)
    vectors = collect_mid_vectors(
        base_loader,
        mask=mask,
        block_size=block_size,
        max_images_per_class=cfg["data"]["pca_sample_images"],
        device=device,
    )

    stats = build_pca_trigger(
        vectors,
        k_tail=int(cfg.get("pca", {}).get("k_tail", 4)),
        seed=int(cfg["experiment"]["seed"]),
        block_size=block_size,
        dataset_name=dataset_name,
        mask=mask,
    )
    stats.save(pca_path)
    print(f"Saved PCA stats to {pca_path}")


if __name__ == "__main__":
    main()
