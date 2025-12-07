from __future__ import annotations

import io
from typing import Any, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from .mask_utils import mask_from_pca_cfg


def _get_cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return default


def _split_cfg(cfg: Any) -> tuple[Any, Any | None]:
    if isinstance(cfg, dict) and "data" in cfg:
        return cfg.get("data", {}), cfg.get("pca")
    return cfg, None


def _build_quality_table(quality: int = 75) -> np.ndarray:
    Q50 = np.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ],
        dtype=np.float64,
    )

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality

    Q = np.floor((Q50 * scale + 50) / 100)
    Q = np.clip(Q, 1, 255)
    return Q


def _zigzag_indices(block_size: int = 8) -> list[Tuple[int, int]]:
    order: list[Tuple[int, int]] = []
    for s in range(2 * block_size - 1):
        if s % 2 == 0:
            r = min(s, block_size - 1)
            c = s - r
            while r >= 0 and c < block_size:
                order.append((r, c))
                r -= 1
                c += 1
        else:
            c = min(s, block_size - 1)
            r = s - c
            while c >= 0 and r < block_size:
                order.append((r, c))
                r += 1
                c -= 1
    return order


def _flatten_zigzag(table: np.ndarray) -> list[int]:
    order = _zigzag_indices(table.shape[0])
    return [int(table[u, v]) for (u, v) in order]
def simulate_save_load(img: torch.Tensor, cfg: Any, mask: Sequence[Tuple[int, int]] | None = None) -> torch.Tensor:
    """
    模拟图像的保存与读取。

    - 对非 JPEG 格式保持旧逻辑（uint8/ppm → 量化到 8bit）。
    - JPEG 固定基表质量 75，再将 mask 覆盖位置量化值设为 1，并通过 qtables 写入文件。
    """

    data_cfg, pca_cfg = _split_cfg(cfg)
    image_format = _get_cfg_value(data_cfg, "image_format", None)
    if image_format is None:
        raise ValueError("image_format must be specified in dataset config")

    if image_format in ["uint8", "ppm"]:
        img_uint8 = (img * 255).round().clamp(0, 255).to(torch.uint8)
        return img_uint8.float() / 255.0

    if image_format == "jpeg":
        block_size = int(_get_cfg_value(data_cfg, "block_size", 8))
        if block_size != 8:
            raise ValueError("JPEG 保存仅支持 block_size=8")

        mask_list = list(mask) if mask is not None else mask_from_pca_cfg(block_size, pca_cfg)

        base_q = _build_quality_table(quality=75)
        custom_q = base_q.copy()
        for u, v in mask_list:
            custom_q[u, v] = 1

        q_flat = _flatten_zigzag(custom_q)
        pil = to_pil_image(img)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", qtables=[q_flat], subsampling=0)
        buf.seek(0)
        return to_tensor(Image.open(buf))

    raise ValueError(f"Unknown image_format: {image_format}")
