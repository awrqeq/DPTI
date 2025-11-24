from __future__ import annotations

"""
可视化入口（根目录）：
- 原图 / 标记图 / 残差图 三列展示
- 保证 original 与 tagged 完全独立，不被污染
- PSNR、L2、mean|res|、effective_beta 正确显示
"""

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec

from src.config import ensure_dir, load_config, resolve_pca_stats_path
from src.data import _load_dataset, build_pca_loader
from src.frequency import (
    FrequencyParams,
    FrequencyStats,
    FrequencyTagger,
    build_pca_trigger,
    collect_mid_vectors,
    compute_psnr,
    get_mid_freq_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="频域标记可视化（汇总图）")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="./visualizations")
    parser.add_argument("--scale", type=float, default=20.0)
    return parser.parse_args()


def resolve_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def _load_or_build_stats(
    cfg,
    mask,
    block_size,
    dataset_name,
    device,
    use_smallest_eigvec_only: bool,
    channel_mode: str,
    model_name: str | None,
    pca_path: Path,
) -> FrequencyStats:
    ensure_dir(pca_path.parent)
    if pca_path.exists():
        print(f"Loading PCA stats from {pca_path}")
        return FrequencyStats.load(pca_path)

    print(f"Saving PCA stats to: {pca_path.name}")
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
    print(f"Saved PCA stats to {pca_path}")
    return stats


def _prepare_subplot(ax, image: np.ndarray, title: str, metrics: str, cmap=None):
    if cmap:
        ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
    else:
        ax.imshow(image)
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    ax.text(
        0.5,
        -0.12,
        metrics,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = resolve_device(cfg["experiment"]["device"])
    print(f"Using device: {device}")

    dataset_name = cfg["data"]["name"].lower()
    block_size = int(cfg["data"].get("block_size", 4))
    beta = float(cfg["data"]["beta"])
    freq_cfg = cfg.get("frequency", {})
    use_smallest_eigvec_only = bool(freq_cfg.get("use_smallest_eigvec_only", False))
    if "channel_mode" not in freq_cfg:
        raise KeyError("frequency.channel_mode is required and must be one of: Y / UV / YUV")
    channel_mode = str(freq_cfg["channel_mode"]).upper()
    model_name = str(cfg.get("model", {}).get("name", "")).lower() or None

    mask_cfg = cfg.get("pca", {}).get("mask", None)
    mask = [tuple(m) for m in mask_cfg] if mask_cfg else get_mid_freq_indices(dataset_name, block_size)

    print(f"Mask size={len(mask)}, dataset={dataset_name}, block={block_size}")

    pca_path = resolve_pca_stats_path(
        cfg,
        dataset_name=dataset_name,
        block_size=block_size,
        channel_mode=channel_mode,
        model_name=model_name,
    )

    stats = _load_or_build_stats(
        cfg,
        mask,
        block_size,
        dataset_name,
        device,
        use_smallest_eigvec_only=use_smallest_eigvec_only,
        channel_mode=channel_mode,
        model_name=model_name,
        pca_path=pca_path,
    )
    freq_params = FrequencyParams(
        stats=stats,
        mask=mask,
        block_size=block_size,
        dataset_name=dataset_name,
        channel_mode=channel_mode,
    )
    tagger = FrequencyTagger(freq_params, beta=beta)

    train_images, train_labels, _, _, _ = _load_dataset(cfg)
    num_samples = min(args.num_samples, train_images.shape[0])

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    fig = plt.figure(figsize=(12, 3 * num_samples))
    grid = gridspec.GridSpec(num_samples, 3, figure=fig, wspace=0.05, hspace=0.35)

    fig.suptitle(
        f"Dataset={dataset_name}, Block={block_size}, Beta={beta}, PCA={pca_path.name}",
        fontsize=12,
    )

    for idx in range(num_samples):
        # ----------- 正确的原图 / 注入图逻辑 ------------
        img = torch.clamp(train_images[idx].to(device), 0.0, 1.0)
        orig = img.clone().detach()
        tagged = tagger.apply(img.clone().detach())
        raw_residual = tagged - orig

        # ---------- 指标 ----------
        psnr = compute_psnr(orig, tagged)
        l2 = torch.norm(raw_residual.view(-1)).item()
        eff_beta = tagger._scaled_beta(orig.shape[1], orig.shape[2])
        mean_abs = raw_residual.abs().mean().item()

        torch.save(raw_residual.cpu(), output_dir / f"raw_residual_{idx:03d}.pt")

        # ---------- 可视化数据 ----------
        img_vis = orig.detach().cpu().permute(1, 2, 0).numpy()
        tagged_vis = tagged.detach().cpu().permute(1, 2, 0).numpy()

        residual_vis = torch.clamp(raw_residual.abs() * args.scale, 0.0, 1.0)
        residual_vis = residual_vis.mean(dim=0).detach().cpu().numpy()

        orig_metrics = "Reference image (no perturbation)"
        tagged_metrics = (
            f"PSNR={psnr:.2f} | L2={l2:.4f} | mean|res|={mean_abs:.4f}\n"
            f"eff_beta={eff_beta:.4f} | block={block_size}"
        )
        residual_metrics = (
            f"L2={l2:.4f} | mean|res|={mean_abs:.4f}\n"
            f"scale={args.scale:.2f}"
        )

        # ---------- 三列可视化 ----------
        ax_orig = fig.add_subplot(grid[idx, 0])
        _prepare_subplot(ax_orig, img_vis, "Original", orig_metrics)

        ax_tag = fig.add_subplot(grid[idx, 1])
        _prepare_subplot(ax_tag, tagged_vis, "Tagged", tagged_metrics)

        ax_res = fig.add_subplot(grid[idx, 2])
        _prepare_subplot(ax_res, residual_vis, "Residual (abs*scale)", residual_metrics, cmap="gray")

    summary_path = output_dir / "summary.png"
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(summary_path, dpi=200)
    plt.close(fig)

    print(f"Saved visualization to {summary_path}")
    print(f"Residual tensors saved in {output_dir}")


if __name__ == "__main__":
    main()
