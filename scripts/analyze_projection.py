from __future__ import annotations

"""
分析 PCA 最小特征方向的投影分布，辅助选择合适的 beta（离群强度）。

用法：
  python scripts/analyze_projection.py --config configs/cifar10_resnet18_bs8.yaml --max-images 2000 --save-hist hist.png
"""

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.config import load_config, resolve_pca_stats_path
from src.datasets import build_pca_loader
from src.frequency import FrequencyStats, block_dct
from src.mask_utils import mask_from_pca_cfg
from src.pca_utils import _mask_to_flat_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PCA projection distribution")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--max-images", type=int, default=2000, help="Max images to sample")
    parser.add_argument("--save-hist", type=str, default=None, help="Optional path to save histogram PNG")
    return parser.parse_args()


def collect_projections(
    loader,
    mask: Sequence[Tuple[int, int]],
    block_size: int,
    w: np.ndarray,
    max_images: int,
    device: torch.device,
) -> np.ndarray:
    flat_indices = _mask_to_flat_indices(mask, block_size).to(device)
    w_t = torch.from_numpy(w.astype(np.float64)).to(device)

    projections = []
    seen = 0
    for images, _ in loader:
        for img in images:
            img = torch.clamp(img, 0.0, 1.0).to(device=device, dtype=torch.float32)
            img = torch.clamp(img * 255.0, 0.0, 255.0)

            # 仅用 Y 通道
            r, g, b = img
            y_ch = 0.299 * r + 0.587 * g + 0.114 * b
            y_ch = torch.clamp(y_ch, 0.0, 255.0)

            coeffs = block_dct(y_ch, block_size=block_size)[0]  # (hb, wb, bs, bs)
            hb, wb = coeffs.shape[:2]
            flat = coeffs.contiguous().view(hb * wb, block_size * block_size)
            masked_blocks = flat[:, flat_indices]
            vector = masked_blocks.reshape(-1)

            if vector.numel() != w_t.numel():
                raise ValueError(
                    f"Vector length mismatch: image vector {vector.numel()} vs w {w_t.numel()}. "
                    "Please regenerate PCA stats with matching config."
                )

            proj = torch.dot(vector.double(), w_t)
            projections.append(proj.item())

            seen += 1
            if seen >= max_images:
                return np.array(projections, dtype=np.float64)

    return np.array(projections, dtype=np.float64)


def summarize_and_plot(proj: np.ndarray, save_hist: Path | None = None) -> None:
    if proj.size == 0:
        print("No projections collected.")
        return
    stats = {
        "count": proj.size,
        "mean": float(np.mean(proj)),
        "std": float(np.std(proj)),
        "min": float(np.min(proj)),
        "max": float(np.max(proj)),
        "p1": float(np.percentile(proj, 1)),
        "p5": float(np.percentile(proj, 5)),
        "p50": float(np.percentile(proj, 50)),
        "p95": float(np.percentile(proj, 95)),
        "p99": float(np.percentile(proj, 99)),
    }
    print("Projection stats:", stats)
    print("建议：可尝试 beta 在 [mean ± 3*std] 范围外作为离群值，例如:",
          f"{stats['mean'] + 3*stats['std']:.3f} 或 {stats['mean'] - 3*stats['std']:.3f}")

    if save_hist is not None:
        plt.figure(figsize=(6, 4))
        plt.hist(proj, bins=100, alpha=0.8, color="steelblue")
        plt.axvline(stats["mean"], color="red", linestyle="--", label="mean")
        plt.title("Projection distribution")
        plt.xlabel("projection value")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        save_hist.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_hist, dpi=200)
        plt.close()
        print(f"Saved histogram to {save_hist}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    dataset_name = cfg["data"]["name"].lower()
    block_size = int(cfg["data"].get("block_size", 8))
    pca_cfg = cfg.get("pca", {})
    mask = mask_from_pca_cfg(block_size, pca_cfg, dataset_name=dataset_name)

    model_name = str(cfg.get("model", {}).get("name", "")).lower() or None
    pca_path = resolve_pca_stats_path(
        cfg,
        dataset_name=dataset_name,
        block_size=block_size,
        model_name=model_name,
    )
    if not pca_path.exists():
        raise FileNotFoundError(f"PCA stats not found: {pca_path}. Please run pca_stats.py first.")

    stats = FrequencyStats.load(pca_path)
    w = stats.w

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_pca_loader(cfg)
    projections = collect_projections(
        loader=loader,
        mask=mask,
        block_size=block_size,
        w=w,
        max_images=args.max_images,
        device=device,
    )

    save_hist = Path(args.save_hist) if args.save_hist else None
    summarize_and_plot(projections, save_hist=save_hist)


if __name__ == "__main__":
    main()
