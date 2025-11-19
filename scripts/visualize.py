from __future__ import annotations

"""
可视化频域标记效果：读取配置，复用与训练一致的 PCA/掩码/块大小，并保存原图、标记图、残差及指标。
"""

import argparse
from pathlib import Path
from typing import List

import torch
from torchvision.utils import save_image

from src.config import ensure_dir, load_config
from src.data import _load_dataset
from src.frequency import (
    FrequencyParams,
    FrequencyStats,
    FrequencyTagger,
    build_pca_trigger,
    collect_mid_vectors,
    compute_psnr,
    get_mid_freq_indices,
)
from src.data import get_mean_std, build_pca_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化频域标记")
    parser.add_argument("--config", type=str, required=True, help="实验配置文件路径")
    parser.add_argument("--num-samples", type=int, default=8, help="要可视化的样本数量")
    parser.add_argument("--output-dir", type=str, default="./visualizations", help="输出目录")
    parser.add_argument("--scale", type=float, default=20.0, help="残差可视化放大倍数")
    parser.add_argument("--save-pt", action="store_true", help="额外保存未放大的残差 .pt 文件")
    return parser.parse_args()


def _load_or_build_stats(cfg, mask, block_size, dataset_name):
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


def _save_sample(output_dir: Path, idx: int, orig: torch.Tensor, tagged: torch.Tensor, residual: torch.Tensor, meta_lines: List[str]):
    save_image(orig, output_dir / f"sample_{idx:03d}_original.png")
    save_image(tagged, output_dir / f"sample_{idx:03d}_tagged.png")
    save_image(residual, output_dir / f"sample_{idx:03d}_residual.png")
    with (output_dir / f"sample_{idx:03d}_meta.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines))


def main():
    args = parse_args()
    cfg = load_config(args.config)

    dataset_name = cfg["data"]["name"].lower()
    block_size = int(cfg["data"].get("block_size", 4))
    beta = float(cfg["data"]["beta"])

    mask_cfg = cfg.get("pca", {}).get("mask", None)
    mask = [tuple(m) for m in mask_cfg] if mask_cfg is not None else get_mid_freq_indices(dataset_name, block_size)
    print(f"Using mid-frequency mask size={len(mask)} for dataset={dataset_name}, block_size={block_size}")

    stats = _load_or_build_stats(cfg, mask, block_size, dataset_name)
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

    ensure_dir(args.output_dir)
    output_dir = Path(args.output_dir)

    mean, std = get_mean_std(dataset_name)

    for idx in range(num_samples):
        img = train_images[idx]
        tagged = tagger.apply(img)
        raw_residual = tagged - img
        residual = torch.abs(raw_residual) * float(args.scale)
        residual = torch.clamp(residual, 0.0, 1.0)

        psnr = compute_psnr(img, tagged)
        l2 = torch.norm((tagged - img).view(-1)).item()
        eff_beta = tagger._scaled_beta(img.shape[1], img.shape[2])

        meta = [
            f"dataset={dataset_name}",
            f"block_size={block_size}",
            f"beta={beta}",
            f"effective_beta={eff_beta}",
            f"psnr={psnr:.4f}",
            f"l2={l2:.4f}",
            f"mask_size={len(mask)}",
            f"label={int(train_labels[idx])}",
        ]

        orig_denorm = img
        tagged_denorm = tagged
        _save_sample(output_dir, idx, orig_denorm, tagged_denorm, residual, meta)

        if args.save_pt:
            torch.save(raw_residual, output_dir / f"sample_{idx:03d}_residual.pt")

    print(f"Saved visualization to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
