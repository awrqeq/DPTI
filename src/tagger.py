from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import torch

from .dct_utils import block_dct, block_idct
from .pca_utils import FrequencyStats, _mask_to_flat_indices


@dataclass
class FrequencyParams:
    stats: FrequencyStats
    mask: Sequence[Tuple[int, int]]
    block_size: int
    dataset_name: str


def normalize_tensor(t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (t - mean[:, None, None]) / std[:, None, None]


def denormalize_tensor(t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return t * std[:, None, None] + mean[:, None, None]


def compute_psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    """在[0,1] 范围内计算 PSNR。"""
    mse = torch.mean((x - y) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


class FrequencyTagger:
    """
    基于 PCA 尾方向的频域标记器（固定 Y 通道）。
    """

    def __init__(
        self,
        params: FrequencyParams,
        beta: float,
    ):
        self.params = params
        self.beta = float(beta)
        self.mask_indices = sorted(list(params.mask), key=lambda p: (p[0], p[1]))
        self.block_size = params.block_size
        self.dataset_name = params.dataset_name
        self.w = torch.from_numpy(params.stats.w.astype(np.float64))
        self.mask_flat = _mask_to_flat_indices(self.mask_indices, self.block_size)

    def _scaled_beta(self, h: int, w: int) -> float:
        return self.beta

    def _apply_alignment_to_channel(self, channel: torch.Tensor, beta_scaled: float) -> torch.Tensor:
        coeffs = block_dct(channel, self.block_size)[0]

        hb, wb = coeffs.shape[:2]
        flat = coeffs.reshape(hb * wb, self.block_size * self.block_size).clone()

        mask_flat = self.mask_flat.to(flat.device)
        vectors = flat[:, mask_flat]
        vector = vectors.reshape(-1)

        w_vec = self.w.to(vector.device)
        if w_vec.numel() != vector.numel():
            raise ValueError(
                f"PCA direction length mismatch: got w with {w_vec.numel()} dims but image provides {vector.numel()} dims. "
                "Please rebuild PCA stats with image-level sampling and matching image size."
            )

        proj = torch.dot(vector, w_vec)
        delta = (beta_scaled - proj) * w_vec
        vector_new = vector + delta

        vectors_new = vector_new.view(hb * wb, mask_flat.numel())
        flat[:, mask_flat] = vectors_new

        coeffs_new = flat.view(hb, wb, self.block_size, self.block_size)
        channel_rec = block_idct(coeffs_new.unsqueeze(0), self.block_size, channel.shape[0], channel.shape[1])[0]
        return torch.clamp(channel_rec, 0.0, 255.0)

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        """
        对单张[0,1] 图像施加频域标记。
        """

        img = img.clone().detach().to(torch.float32)
        img = torch.clamp(img, 0.0, 1.0)

        # YUV 转换
        img_255 = img.to(torch.float64) * 255.0
        r, g, b = img_255
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u_ch = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
        v_ch = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
        y = torch.clamp(y, 0.0, 255.0)
        u_ch = torch.clamp(u_ch, 0.0, 255.0)
        v_ch = torch.clamp(v_ch, 0.0, 255.0)

        h, w = y.shape
        beta_scaled = self._scaled_beta(h, w)

        y_rec = self._apply_alignment_to_channel(y, beta_scaled)
        u_rec, v_rec = u_ch, v_ch

        # 还原 RGB
        v_centered = v_rec - 128.0
        u_centered = u_rec - 128.0
        r_rec = y_rec + 1.402 * v_centered
        g_rec = y_rec - 0.344136 * u_centered - 0.714136 * v_centered
        b_rec = y_rec + 1.772 * u_centered
        rgb = torch.stack([r_rec, g_rec, b_rec], dim=0) / 255.0
        rgb = torch.clamp(rgb, 0.0, 1.0)

        return rgb.to(torch.float32)
