from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .color import rgb_to_yuv, yuv_to_rgb
from .dct import dct2, idct2


def mid_mask() -> np.ndarray:
    """Return the fixed 4x4 mid-frequency mask."""
    return np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.int32,
    )


def mask_indices(mask: np.ndarray) -> np.ndarray:
    """Return linear indices where mask == 1 in row-major order."""
    return np.argwhere(mask > 0)


def extract_mid_vector(block_dct: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return block_dct[mask == 1].astype(np.float64)


def write_mid_vector(block_dct: np.ndarray, mask: np.ndarray, vec: np.ndarray) -> None:
    block_dct[mask == 1] = vec


@dataclass
class FrequencyStats:
    mu: np.ndarray
    cov: np.ndarray
    W_tail: np.ndarray
    w: np.ndarray

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"mu": self.mu, "cov": self.cov, "W_tail": self.W_tail, "w": self.w}
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load(path: str | Path) -> "FrequencyStats":
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        return FrequencyStats(
            mu=data["mu"],
            cov=data["cov"],
            W_tail=data["W_tail"],
            w=data["w"],
        )


@dataclass
class FrequencyParams:
    """预先计算好的频域参数集合，供频域标记直接复用。"""

    stats: FrequencyStats
    mask: np.ndarray


def build_pca_trigger(vectors: np.ndarray, k_tail: int = 4, seed: int = 42) -> FrequencyStats:
    """Compute PCA tail subspace and trigger direction."""
    rng = np.random.default_rng(seed)
    mu = np.mean(vectors, axis=0)
    centered = vectors - mu
    # Covariance with row observations.
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Tail subspace from smallest eigenvalues.
    idx = np.argsort(eigvals)
    tail_idx = idx[:k_tail]
    W_tail = eigvecs[:, tail_idx]
    a = rng.standard_normal(k_tail)
    a = a / np.linalg.norm(a)
    w = W_tail @ a
    w = w / np.linalg.norm(w)
    return FrequencyStats(mu=mu, cov=cov, W_tail=W_tail, w=w)


def build_frequency_params(
    dataloader: DataLoader,
    mask: np.ndarray,
    sample_blocks: int,
    k_tail: int,
    seed: int,
    device: torch.device | str = "cpu",
) -> FrequencyParams:
    """收集频域统计量并构造频域标记所需的参数。"""

    vectors = collect_mid_vectors(
        dataloader,
        mask=mask,
        sample_blocks=sample_blocks,
        device=device,
    )
    stats = build_pca_trigger(vectors, k_tail=k_tail, seed=seed)
    return FrequencyParams(stats=stats, mask=mask.astype(np.int32))


def collect_mid_vectors(
    dataloader: DataLoader,
    mask: np.ndarray,
    sample_blocks: int = 20000,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Collect mid-frequency vectors from dataset until reaching sample_blocks."""

    collected: List[np.ndarray] = []
    bs = mask.shape[0]
    mask_bool = mask.astype(bool)
    device = torch.device(device)

    with tqdm(total=sample_blocks, desc="Collecting frequency vectors") as pbar:
        for images, _ in dataloader:
            images = images.to(device)
            with torch.no_grad():
                # work on CPU numpy for DCT; transfer as needed
                np_imgs = images.cpu().numpy().astype(np.float64)
            for img in np_imgs:
                # img shape C,H,W in [0,1]
                chw = np.clip(img * 255.0, 0.0, 255.0)
                rgb = np.transpose(chw, (1, 2, 0))
                yuv = rgb_to_yuv(rgb)
                y = yuv[..., 0]
                h, w = y.shape
                for i in range(0, h, bs):
                    for j in range(0, w, bs):
                        block = y[i : i + bs, j : j + bs]
                        block_dct = dct2(block)
                        c = extract_mid_vector(block_dct, mask_bool)
                        collected.append(c)
                        pbar.update(1)
                        if len(collected) >= sample_blocks:
                            return np.stack(collected, axis=0)

    return np.stack(collected, axis=0)


def _adjust_y_channel(y_channel: np.ndarray, w_vec: np.ndarray, beta: float, mask: np.ndarray) -> np.ndarray:
    """Adjust Y channel blocks along the direction w."""
    bsz = mask.shape[0]
    mask_bool = mask.astype(bool)
    w_vec = w_vec.astype(np.float64)
    h, w = y_channel.shape
    y_out = np.zeros_like(y_channel, dtype=np.float64)
    for i in range(0, h, bsz):
        for j in range(0, w, bsz):
            block = y_channel[i : i + bsz, j : j + bsz]
            block_dct = dct2(block)
            c = extract_mid_vector(block_dct, mask_bool)
            current = float(np.dot(c, w_vec))
            delta = (beta - current) * w_vec
            c_prime = c + delta
            write_mid_vector(block_dct, mask_bool, c_prime)
            block_new = idct2(block_dct)
            y_out[i : i + bsz, j : j + bsz] = block_new
    return y_out


def enhance_frequency(
    image: torch.Tensor,
    stats: FrequencyStats,
    beta: float,
    mask: np.ndarray | None = None,
) -> torch.Tensor:
    """Enhance an image along the learned frequency direction."""
    if mask is None:
        mask = mid_mask()
    mask = mask.astype(np.int32)
    img_np = image.detach().cpu().numpy().astype(np.float64)
    img_np = np.clip(img_np * 255.0, 0.0, 255.0)
    rgb = np.transpose(img_np, (1, 2, 0))
    yuv = rgb_to_yuv(rgb)
    y = yuv[..., 0]
    y_adj = _adjust_y_channel(y, stats.w, beta=beta, mask=mask)
    yuv_mod = yuv.copy()
    yuv_mod[..., 0] = y_adj
    rgb_mod = yuv_to_rgb(yuv_mod)
    rgb_mod = np.clip(rgb_mod, 0.0, 255.0)
    chw = np.transpose(rgb_mod, (2, 0, 1)) / 255.0
    return torch.from_numpy(chw.astype(np.float32))


def apply_frequency_mark(image: torch.Tensor, params: FrequencyParams, beta: float) -> torch.Tensor:
    """
    对单张图像施加频域标记。

    参数
    ------
    image: torch.Tensor
        形状为 (3, H, W)，数值范围 [0, 1] 的图像张量。
    params: FrequencyParams
        预先计算好的频域统计量和掩码。
    beta: float
        频域扰动强度。
    """

    return enhance_frequency(image, stats=params.stats, beta=beta, mask=params.mask)


def normalize_tensor(t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (t - mean[:, None, None]) / std[:, None, None]


def denormalize_tensor(t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return t * std[:, None, None] + mean[:, None, None]
