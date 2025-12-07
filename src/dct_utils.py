from __future__ import annotations

import math
from typing import List

import torch

# ---------------------------------------------------------------------------
# DCT/IDCT 基础
# ---------------------------------------------------------------------------

_DCT_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _cache_key(block_size: int, device: torch.device) -> tuple[int, str]:
    device = torch.device(device)
    return block_size, f"{device.type}:{device.index if device.index is not None else -1}"


def build_dct_matrix(block_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """构建并缓存给定 block_size 的 DCT 矩阵 (block_size, block_size)。

    - 按设备缓存，避免 GPU/CPU 反复拷贝；dtype 固定 float64 以减少数值误差。
    """

    device = torch.device(device)
    key = _cache_key(block_size, device)
    if key not in _DCT_CACHE:
        n = torch.arange(block_size, dtype=torch.float64, device=device)
        k = n.view(-1, 1)
        mat = torch.cos(math.pi / block_size * (n + 0.5) * k)
        mat[0] = mat[0] / math.sqrt(block_size)
        mat[1:] = mat[1:] * math.sqrt(2 / block_size)
        _DCT_CACHE[key] = mat
    return _DCT_CACHE[key]


def _block_dct_2d(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """对单通道 (H, W) 张量做分块 DCT，返回形状(1, hb, wb, bs, bs)。"""
    h, w = x.shape
    bs = block_size
    assert h % bs == 0 and w % bs == 0, "图像尺寸必须被 block_size 整除"
    hb, wb = h // bs, w // bs
    blocks = x.view(hb, bs, wb, bs).permute(0, 2, 1, 3)  # (hb, wb, bs, bs)

    dct = build_dct_matrix(bs, device=x.device)
    blocks = blocks.to(dtype=torch.float64)
    temp = torch.einsum("ij,abjk->abik", dct, blocks)
    coeffs = torch.einsum("abij,jk->abik", temp, dct.t())
    coeffs = coeffs.unsqueeze(0)  # (1, hb, wb, bs, bs)
    return coeffs


def _block_idct_2d(blocks: torch.Tensor, block_size: int, h: int, w: int) -> torch.Tensor:
    """从(1, hb, wb, bs, bs) 的 DCT 系数做 IDCT 还原为 (H, W)。"""
    bs = block_size
    hb, wb = h // bs, w // bs
    blocks = blocks.squeeze(0)  # (hb, wb, bs, bs)
    dct = build_dct_matrix(bs, device=blocks.device)
    temp = torch.einsum("abij,jk->abik", blocks, dct)
    spatial = torch.einsum("ij,abjk->abik", dct.t(), temp)
    spatial = spatial.permute(0, 2, 1, 3).contiguous().view(h, w)
    return spatial


def block_dct(img: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    对输入(C, H, W) 或(H, W) 图像执行分块 DCT，返回与输入通道数匹配的系数。
    返回形状：(C, hb, wb, bs, bs) 或 (1, hb, wb, bs, bs)。
    """
    if img.dim() == 2:
        img = img.unsqueeze(0)
    c, h, w = img.shape
    coeffs: List[torch.Tensor] = []
    for ch in range(c):
        coeffs.append(_block_dct_2d(img[ch], block_size))
    return torch.cat(coeffs, dim=0)


def block_idct(coeffs: torch.Tensor, block_size: int, h: int, w: int) -> torch.Tensor:
    """从分块 DCT 系数复原图像，输出形状 (C, H, W)。"""
    if coeffs.dim() == 4:  # 无通道维，补齐
        coeffs = coeffs.unsqueeze(0)
    channels: List[torch.Tensor] = []
    for ch in range(coeffs.shape[0]):
        channels.append(_block_idct_2d(coeffs[ch : ch + 1], block_size, h, w))
    return torch.stack(channels, dim=0)
