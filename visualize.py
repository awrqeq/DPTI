from __future__ import annotations

"""
可视化入口（根目录）：
- 读取配置并与主流程一致地构建 PCA/掩码/频域标记器。
- 从未归一化的训练数据中抽样，生成单张三列多行的大图：原图 / 标记图 / 残差（abs(raw_residual)*scale）。
- 在图中直接写入 PSNR、L2、effective_beta、block_size、dataset_name 等指标，另存原始残差 .pt。

运行示例：
    python visualize.py --config configs/cifar10_resnet18_bs8.yaml --num-samples 4 --output-dir ./visualizations --scale 20
"""

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import gridspec

from src.config import ensure_dir, load_config
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
    parser.add_argument("--config", type=str, required=True, help="配置文件路径 (YAML)")
    parser.add_argument("--num-samples", type=int, default=4, help="可视化的样本行数")
    parser.add_argument("--output-dir", type=str, default="./visualizations", help="输出目录")
    parser.add_argument("--scale", type=float, default=20.0, help="残差可视化放大倍数")
    parser.add_argument("--save-pt", action="store_true", help="兼容参数：残差 pt 始终保存")
    return parser.parse_args()


def resolve_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def _load_or_build_stats(
    cfg, mask: Sequence[Tuple[int, int]], block_size: int, dataset_name: str, device: torch.device
) -> FrequencyStats:
    pca_path = Path(cfg["pca"]["save_path"])
    ensure_dir(pca_path.parent)
    if pca_path.exists():
        print(f"Loading PCA stats from {pca_path}")
        return FrequencyStats.load(pca_path)

    print("Building PCA stats for visualization...")
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
    )
    stats.save(pca_path)
    return stats


def _prepare_subplot(ax, image: np.ndarray, title: str, metrics: str, cmap: str | None = None) -> None:
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = resolve_device(cfg["experiment"]["device"])
    print(f"Using device: {device}")

    dataset_name = cfg["data"]["name"].lower()
    block_size = int(cfg["data"].get("block_size", 4))
    beta = float(cfg["data"]["beta"])

    mask_cfg = cfg.get("pca", {}).get("mask", None)
    mask = [tuple(m) for m in mask_cfg] if mask_cfg is not None else get_mid_freq_indices(dataset_name, block_size)
    print(f"Using mid-frequency mask size={len(mask)} for dataset={dataset_name}, block_size={block_size}")

    stats = _load_or_build_stats(cfg, mask, block_size, dataset_name, device)
    freq_params = FrequencyParams(
        stats=stats,
        mask=mask,
        block_size=block_size,
        dataset_name=dataset_name,
        match_global_energy=True,
        base_block_size_for_energy=4,
    )
    tagger = FrequencyTagger(freq_params, beta=beta)

    train_images, train_labels, _, _, _ = _load_dataset(cfg)
    num_samples = min(args.num_samples, train_images.shape[0])

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    fig = plt.figure(figsize=(12, 3 * num_samples))
    grid = gridspec.GridSpec(num_samples, 3, figure=fig, wspace=0.05, hspace=0.35)

    suptitle = (
        f"Dataset = {dataset_name}, Block = {block_size}, Beta = {beta}, "
        f"PCA path = {cfg['pca']['save_path']}"
    )
    fig.suptitle(suptitle, fontsize=12)

    for idx in range(num_samples):
        img = torch.clamp(train_images[idx].to(device), 0.0, 1.0)
        orig = img.clone().detach()
        tagged = tagger.apply(img.clone().detach())
        raw_residual = tagged - orig

        psnr = compute_psnr(orig, tagged)
        l2 = torch.norm(raw_residual.view(-1)).item()
        eff_beta = tagger._scaled_beta(orig.shape[1], orig.shape[2])
        mean_abs = raw_residual.abs().mean().item()

        torch.save(raw_residual.cpu(), output_dir / f"raw_residual_{idx:03d}.pt")

        img_vis = torch.clamp(orig, 0.0, 1.0).detach().cpu().permute(1, 2, 0).numpy()
        tagged_vis = torch.clamp(tagged, 0.0, 1.0).detach().cpu().permute(1, 2, 0).numpy()
        residual_vis = torch.clamp(raw_residual.abs() * float(args.scale), 0.0, 1.0)
        residual_vis = residual_vis.mean(dim=0).detach().cpu().numpy()

        metrics_text = (
            f"PSNR={psnr:.2f} | L2={l2:.4f} | mean|res|={mean_abs:.4f}\n"
            f"eff_beta={eff_beta:.4f} | block={block_size} | dataset={dataset_name}"
        )

        ax_orig = fig.add_subplot(grid[idx, 0])
        _prepare_subplot(ax_orig, img_vis, "Original", metrics_text)

        ax_tag = fig.add_subplot(grid[idx, 1])
        _prepare_subplot(ax_tag, tagged_vis, "Tagged", metrics_text)

        ax_res = fig.add_subplot(grid[idx, 2])
        _prepare_subplot(ax_res, residual_vis, "Residual (abs*scale)", metrics_text, cmap="gray")

    summary_path = output_dir / "summary.png"
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(summary_path, dpi=200)
    plt.close(fig)

    print(f"Saved visualization summary to {summary_path.resolve()}")
    print(f"Raw residual tensors stored in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
