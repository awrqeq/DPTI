from __future__ import annotations

"""
频域工具模块（以 8x8 块为主）。
- 统一的块状 DCT/IDCT 实现，可按 block_size 参数化。
- 数据集感知的中频掩码生成，并支持按配置阈值生成。
- PCA 尾子空间方向构建，并通过 FrequencyTagger 注入频域标记。
- 可同时用于训练（离线/在线标记）与可视化脚本。
"""

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .io_utils import simulate_save_load

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
    """对单通道 (H, W) 张量做分块 DCT，返回形状 (1, hb, wb, bs, bs)。"""
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
    """将 (1, hb, wb, bs, bs) 的 DCT 系数做 IDCT 还原为 (H, W)。"""
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
    对输入 (C, H, W) 或 (H, W) 图像执行分块 DCT，返回与输入通道数匹配的系数。
    返回形状：(C, hb, wb, bs, bs) 或 (1, hb, wb, bs, bs)。
    """
    if img.dim() == 2:
        img = img.unsqueeze(0)
    c, h, w = img.shape
    coeffs = []
    for ch in range(c):
        coeffs.append(_block_dct_2d(img[ch], block_size))
    return torch.cat(coeffs, dim=0)


def block_idct(coeffs: torch.Tensor, block_size: int, h: int, w: int) -> torch.Tensor:
    """从分块 DCT 系数复原图像，输出形状 (C, H, W)。"""
    if coeffs.dim() == 4:  # 无通道维，补齐
        coeffs = coeffs.unsqueeze(0)
    channels = []
    for ch in range(coeffs.shape[0]):
        channels.append(_block_idct_2d(coeffs[ch : ch + 1], block_size, h, w))
    return torch.stack(channels, dim=0)


# ---------------------------------------------------------------------------
# 中频掩码（数据集感知）
# ---------------------------------------------------------------------------

def _mask_to_flat_indices(mask: Sequence[Tuple[int, int]], block_size: int) -> torch.Tensor:
    """将 (u,v) 掩码转换为展平索引（行优先排序，稳定）。"""

    mask_sorted = sorted(mask, key=lambda p: (p[0], p[1]))
    mask_tensor = torch.zeros((block_size, block_size), dtype=torch.bool)
    for u, v in mask_sorted:
        mask_tensor[u, v] = True
    flat = mask_tensor.view(-1)
    # 使用顺序索引可保证 row-major 顺序稳定
    indices = torch.arange(flat.numel(), dtype=torch.long)[flat]
    return indices


def gen_mask_by_sum(block_size: int, s_min: int, s_max: int, exclude_dc: bool = True) -> List[Tuple[int, int]]:
    assert block_size == 8, "当前仅支持 8x8 mask 生成"
    mask = [(u, v) for u in range(block_size) for v in range(block_size) if s_min <= u + v <= s_max]
    if exclude_dc:
        mask = [(u, v) for (u, v) in mask if not (u == 0 and v == 0)]
    return mask


def get_mid_freq_indices(dataset_name: str, block_size: int) -> List[Tuple[int, int]]:
    """根据数据集与 block_size 生成中频坐标列表（用于兼容旧配置）。"""

    name = dataset_name.lower()
    if name == "cifar10" and block_size == 8:
        return gen_mask_by_sum(8, 3, 7)
    if name == "gtsrb" and block_size == 8:
        return gen_mask_by_sum(8, 2, 10)
    if name == "imagenette" and block_size == 8:
        return gen_mask_by_sum(8, 3, 6)
    raise ValueError(f"Unsupported dataset/block_size combination: {dataset_name}, {block_size}")


# ---------------------------------------------------------------------------
# PCA 统计
# ---------------------------------------------------------------------------

@dataclass
class FrequencyStats:
    mu: np.ndarray
    cov: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    w: np.ndarray
    block_size: int
    dataset_name: str
    use_smallest_eigvec_only: bool = False

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mu": self.mu,
            "cov": self.cov,
            "eigvals": self.eigvals,
            "eigvecs": self.eigvecs,
            "w": self.w,
            "block_size": self.block_size,
            "dataset_name": self.dataset_name,
            "use_smallest_eigvec_only": self.use_smallest_eigvec_only,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load(path: str | Path) -> "FrequencyStats":
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        return FrequencyStats(
            mu=data["mu"],
            cov=data["cov"],
            eigvals=data["eigvals"],
            eigvecs=data["eigvecs"],
            w=data["w"],
            block_size=data.get("block_size", 8),
            dataset_name=data.get("dataset_name", "cifar10"),
            use_smallest_eigvec_only=data.get("use_smallest_eigvec_only", False),
        )


def collect_mid_vectors(
    loader: DataLoader,
    mask: Sequence[Tuple[int, int]],
    block_size: int,
    max_blocks: int = 20000,
    device: torch.device | str = "cpu",
    *,
    channel_mode: str = "Y",
) -> np.ndarray:
    """收集分块 DCT 中频向量，数量上限为 max_blocks。"""

    device = torch.device(device)
    flat_indices = _mask_to_flat_indices(mask, block_size).to(device)
    channel_mode = str(channel_mode).upper()
    if channel_mode not in {"Y", "UV", "YUV"}:
        raise ValueError("channel_mode must be one of: Y, UV, YUV")

    collected: List[torch.Tensor] = []
    warned = False
    with tqdm(total=max_blocks, desc="Collecting frequency vectors") as pbar:
        for images, _ in loader:
            images = images.to(dtype=torch.float32)
            for img in images:
                img = torch.clamp(img, 0.0, 1.0).to(device=device, dtype=torch.float32)
                img = torch.clamp(img * 255.0, 0.0, 255.0)
                h, w = img.shape[1:]
                if (h % block_size != 0 or w % block_size != 0) and not warned:
                    print(
                        f"[collect_mid_vectors] Warning: image size {(h, w)} is not divisible by block_size={block_size},",
                        " please check resize settings.",
                    )
                    warned = True
                assert h % block_size == 0 and w % block_size == 0, "img_size must be divisible by block_size"

                r, g, b = img
                y_ch = 0.299 * r + 0.587 * g + 0.114 * b
                u_ch = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
                v_ch = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
                y_ch = torch.clamp(y_ch, 0.0, 255.0)
                u_ch = torch.clamp(u_ch, 0.0, 255.0)
                v_ch = torch.clamp(v_ch, 0.0, 255.0)
                pca_channel = {"Y": y_ch, "UV": u_ch, "YUV": y_ch}[channel_mode]

                pca_channel = torch.clamp(pca_channel, 0.0, 255.0)
                coeffs = block_dct(pca_channel, block_size=block_size)[0]  # (hb, wb, bs, bs)
                hb, wb = coeffs.shape[:2]
                flat = coeffs.contiguous().view(hb * wb, block_size * block_size)
                vectors = flat[:, flat_indices]
                vectors = vectors.to(device=device)

                collected.append(vectors.cpu())
                pbar.update(vectors.size(0))
                if sum(v.shape[0] for v in collected) >= max_blocks:
                    merged = torch.cat(collected, dim=0)[:max_blocks]
                    return merged.double().numpy()
    if not collected:
        return np.empty((0, flat_indices.numel()), dtype=np.float64)
    merged = torch.cat(collected, dim=0)
    return merged.double().numpy()


def build_pca_trigger(
    vectors: np.ndarray,
    k_tail: int = 4,
    seed: int = 42,
    block_size: int = 8,
    dataset_name: str = "cifar10",
    use_smallest_eigvec_only: bool = False,
    mask: Sequence[Tuple[int, int]] | None = None,
) -> FrequencyStats:
    """从中频向量中计算 PCA 尾子空间方向。"""

    mask = mask or get_mid_freq_indices(dataset_name, block_size)
    flat_indices = _mask_to_flat_indices(mask, block_size).cpu().numpy().astype(int)

    rng = np.random.default_rng(seed)
    mu = np.mean(vectors, axis=0)
    centered = vectors - mu
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)
    if use_smallest_eigvec_only:
        smallest_idx = int(idx[0])
        w = eigvecs[:, smallest_idx]
    else:
        tail_idx = idx[:k_tail]
        tail_vecs = eigvecs[:, tail_idx]
        a = rng.standard_normal(len(tail_idx))
        a = a / np.linalg.norm(a)
        w = tail_vecs @ a
    w = w / np.linalg.norm(w)

    return FrequencyStats(
        mu=mu,
        cov=cov,
        eigvals=eigvals,
        eigvecs=eigvecs,
        w=w,
        block_size=block_size,
        dataset_name=dataset_name,
        use_smallest_eigvec_only=use_smallest_eigvec_only,
    )


@dataclass
class FrequencyParams:
    stats: FrequencyStats
    mask: Sequence[Tuple[int, int]]
    block_size: int
    dataset_name: str
    channel_mode: str = "Y"


# ---------------------------------------------------------------------------
# 频域标记器
# ---------------------------------------------------------------------------


def rgb_to_yuv(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RGB→YUV（BT.601），完全对称且在 [0,255] 空间内运算。"""

    assert img.dim() == 3 and img.shape[0] == 3, "输入应为 (3,H,W)"
    img_255 = torch.clamp(img.to(torch.float64), 0.0, 1.0) * 255.0
    r, g, b = img_255
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    y = torch.clamp(y, 0.0, 255.0)
    u = torch.clamp(u, 0.0, 255.0)
    v = torch.clamp(v, 0.0, 255.0)
    return y, u, v


def to_y_channel(img: torch.Tensor) -> torch.Tensor:
    """取 Y 通道并保证范围 [0,255]，dtype=float64。"""
    y, _, _ = rgb_to_yuv(img)
    return torch.clamp(y, 0.0, 255.0)


def yuv_to_rgb(y: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """YUV→RGB（BT.601），输入/输出均在 [0,255] 空间。"""

    y = y.to(torch.float64)
    u = u.to(torch.float64) - 128.0
    v = v.to(torch.float64) - 128.0
    r = y + 1.402 * v
    g = y - 0.344136 * u - 0.714136 * v
    b = y + 1.772 * u
    rgb = torch.stack([r, g, b], dim=0)
    return torch.clamp(rgb, 0.0, 255.0)


def extract_uv(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """复用对称 RGB→YUV 公式，避免色彩漂移。"""
    _, u, v = rgb_to_yuv(img)
    return u, v


def compute_psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    """在 [0,1] 范围内计算 PSNR。"""
    mse = torch.mean((x - y) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


class FrequencyTagger:
    """
    基于 PCA 尾方向的频域标记器。

    - 统一以 8x8 块工作。
    - 针对不同数据集的 mask 及统计量可复用。
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
        channel_mode = getattr(params, "channel_mode", "Y")
        channel_mode = str(channel_mode).upper()
        if channel_mode not in {"Y", "UV", "YUV"}:
            raise ValueError(
                f"Unsupported channel_mode={channel_mode}. Expected one of 'Y', 'UV', 'YUV'."
            )
        self.channel_mode = channel_mode

    def _scaled_beta(self, h: int, w: int) -> float:
        return self.beta

    def _apply_alignment_to_channel(self, channel: torch.Tensor, beta_scaled: float) -> torch.Tensor:
        coeffs = block_dct(channel, self.block_size)[0]

        hb, wb = coeffs.shape[:2]
        flat = coeffs.reshape(hb * wb, self.block_size * self.block_size).clone()

        mask_flat = self.mask_flat.to(flat.device)
        vectors = flat[:, mask_flat]

        w_vec = self.w.to(vectors.device)
        proj = (vectors * w_vec).sum(dim=1, keepdim=True)
        deltas = (beta_scaled - proj) * w_vec.unsqueeze(0)

        vectors_new = vectors + deltas
        flat[:, mask_flat] = vectors_new

        coeffs_new = flat.view(hb, wb, self.block_size, self.block_size)
        channel_rec = block_idct(coeffs_new.unsqueeze(0), self.block_size, channel.shape[0], channel.shape[1])[0]
        return torch.clamp(channel_rec, 0.0, 255.0)

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        """
        对单张 [0,1] 图像施加频域标记。
        必须保证不修改输入图像，否则 PSNR 永远错误。
        """

        # ----------------------------------------------------------
        # 🔥 第一行：深拷贝输入，彻底切断所有共享内存
        # ----------------------------------------------------------
        img = img.clone().detach().to(torch.float32)

        # clamp 不能代替 clone，它不会深拷贝
        img = torch.clamp(img, 0.0, 1.0)

        # ----------------------------------------------------------
        # YUV 转换
        # ----------------------------------------------------------
        y, u_ch, v_ch = rgb_to_yuv(img)
        y = y.to(torch.float64)
        u_ch = u_ch.to(torch.float64)
        v_ch = v_ch.to(torch.float64)

        h, w = y.shape
        beta_scaled = self._scaled_beta(h, w)

        # ----------------------------------------------------------
        # 选择性地对不同通道施加频域增强
        # ----------------------------------------------------------
        if self.channel_mode == "Y":
            y_rec = self._apply_alignment_to_channel(y, beta_scaled)
            u_rec, v_rec = u_ch, v_ch
        elif self.channel_mode == "UV":
            y_rec = y
            u_rec = self._apply_alignment_to_channel(u_ch, beta_scaled)
            v_rec = self._apply_alignment_to_channel(v_ch, beta_scaled)
        else:  # "YUV"
            y_rec = self._apply_alignment_to_channel(y, beta_scaled)
            u_rec = self._apply_alignment_to_channel(u_ch, beta_scaled)
            v_rec = self._apply_alignment_to_channel(v_ch, beta_scaled)

        # ----------------------------------------------------------
        # 还原 RGB
        # ----------------------------------------------------------
        rgb = yuv_to_rgb(y_rec, u_rec, v_rec) / 255.0
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # 返回完全独立的新张量
        return rgb.to(torch.float32)



# ---------------------------------------------------------------------------
# 兼容旧接口
# ---------------------------------------------------------------------------


def apply_frequency_mark(
    image: torch.Tensor,
    params: FrequencyParams,
    beta: float,
    tagger: FrequencyTagger | None = None,
    cfg=None,
) -> torch.Tensor:
    """兼容旧接口的轻量包装：内部复用 FrequencyTagger。

    传入已有 tagger 可避免重复构造（DCT 矩阵已按设备缓存）。
    为符合离线攻击者模型，必须提供包含 image_format 的 cfg，
    使频域标记后经过 simulate_save_load 模拟真实存取效果。
    """

    if cfg is None:
        raise ValueError("cfg is required to simulate offline poisoning in apply_frequency_mark")

    tagger = tagger or FrequencyTagger(params, beta=beta)
    marked = tagger.apply(image)
    return simulate_save_load(marked, cfg)


def normalize_tensor(t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (t - mean[:, None, None]) / std[:, None, None]


def denormalize_tensor(t: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return t * std[:, None, None] + mean[:, None, None]
