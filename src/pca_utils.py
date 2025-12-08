from __future__ import annotations

import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dct_utils import block_dct
from .mask_utils import gen_mask_by_sum


def _mask_to_flat_indices(mask: Sequence[Tuple[int, int]], block_size: int) -> torch.Tensor:
    """将 (u,v) 掩码转换为展平索引（行优先顺序，稳定）。"""

    mask_sorted = sorted(mask, key=lambda p: (p[0], p[1]))
    mask_tensor = torch.zeros((block_size, block_size), dtype=torch.bool)
    for u, v in mask_sorted:
        mask_tensor[u, v] = True
    flat = mask_tensor.view(-1)
    indices = torch.arange(flat.numel(), dtype=torch.long)[flat]
    return indices


def get_mid_freq_indices(dataset_name: str, block_size: int) -> List[Tuple[int, int]]:
    """根据数据集与 block_size 生成中频坐标列表（兼容旧配置）。"""

    name = dataset_name.lower()
    if name == "cifar10" and block_size == 8:
        return gen_mask_by_sum(8, 3, 7)
    if name == "gtsrb" and block_size == 8:
        return gen_mask_by_sum(8, 2, 10)
    if name == "imagenette" and block_size == 8:
        return gen_mask_by_sum(8, 3, 6)
    raise ValueError(f"Unsupported dataset/block_size combination: {dataset_name}, {block_size}")


@dataclass
class FrequencyStats:
    mu: np.ndarray
    cov: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    w: np.ndarray
    block_size: int
    dataset_name: str
    vector_length: int | None = None

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
            "vector_length": self.vector_length if self.vector_length is not None else int(len(self.w)),
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
            vector_length=data.get("vector_length", None) or len(data["w"]),
        )


def collect_mid_vectors(
    loader: DataLoader,
    mask: Sequence[Tuple[int, int]],
    block_size: int,
    max_images_per_class: int = 2000,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """按图像为单位平衡采样各类中频向量。"""

    def _infer_num_classes(ds: Any) -> int | None:
        if hasattr(ds, "classes") and len(getattr(ds, "classes")) > 0:
            return len(getattr(ds, "classes"))
        targets = getattr(ds, "targets", None)
        if targets is not None and len(targets) > 0:
            return len(set(int(t) for t in targets))
        return None

    def _infer_class_counts(ds: Any) -> tuple[Counter[int], int] | None:
        targets = getattr(ds, "targets", None)
        if targets is None and hasattr(ds, "samples"):
            samples = getattr(ds, "samples")
            targets = [s[1] for s in samples] if samples else None
        if targets is None:
            return None
        counter = Counter(int(t) for t in targets)
        num_cls = len(counter) if counter else 0
        return counter, num_cls

    if max_images_per_class <= 0:
        raise ValueError("max_images_per_class must be positive.")
    device = torch.device(device)
    flat_indices = _mask_to_flat_indices(mask, block_size).to(device)

    counts_and_num = _infer_class_counts(loader.dataset)
    if counts_and_num is not None:
        class_total_counts, num_classes = counts_and_num
    else:
        num_classes = _infer_num_classes(loader.dataset)
        class_total_counts = None
    if num_classes is None:
        raise ValueError("Unable to infer number of classes for balanced PCA sampling.")

    desired_per_class: dict[int, int] = {}
    for c in range(num_classes):
        available = class_total_counts.get(c, max_images_per_class) if class_total_counts else max_images_per_class
        desired_per_class[c] = min(max_images_per_class, available)

    required_total = sum(desired_per_class.values())
    if required_total == 0:
        return np.empty((0, flat_indices.numel()), dtype=np.float64)

    collected: List[torch.Tensor] = []
    class_counts: Counter[int] = Counter()
    warned = False
    with tqdm(total=required_total, desc="Collecting frequency vectors (per image)") as pbar:
        for images, labels in loader:
            images = images.to(dtype=torch.float32)
            labels = labels.tolist()
            for img, y in zip(images, labels):
                desired_for_cls = desired_per_class.get(int(y), max_images_per_class)
                if class_counts[int(y)] >= desired_for_cls:
                    continue
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
                pca_channel = torch.clamp(y_ch, 0.0, 255.0)
                coeffs = block_dct(pca_channel, block_size=block_size)[0]  # (hb, wb, bs, bs)
                hb, wb = coeffs.shape[:2]
                flat = coeffs.contiguous().view(hb * wb, block_size * block_size)
                masked_blocks = flat[:, flat_indices]
                image_vector = masked_blocks.reshape(-1).to(device=device)

                collected.append(image_vector.cpu())
                class_counts[int(y)] += 1
                pbar.update(1)

                if len(class_counts) == num_classes and all(
                    class_counts[c] >= desired_per_class.get(c, max_images_per_class) for c in range(num_classes)
                ):
                    merged = torch.stack(collected, dim=0)
                    return merged.double().numpy()

    shortfall = {c: (desired_per_class.get(c, 0) - class_counts.get(c, 0)) for c in range(num_classes)}
    shortfall = {c: v for c, v in shortfall.items() if v > 0}
    if shortfall:
        print(f"[collect_mid_vectors] Warning: dataset不足以满足采样上限，短缺: {shortfall}")
    if not collected:
        return np.empty((0, flat_indices.numel()), dtype=np.float64)
    merged = torch.stack(collected, dim=0)
    return merged.double().numpy()


def build_pca_trigger(
    vectors: np.ndarray,
    k_tail: int = 4,
    seed: int = 42,
    block_size: int = 8,
    dataset_name: str = "cifar10",
    mask: Sequence[Tuple[int, int]] | None = None,
) -> FrequencyStats:
    """从中频向量中计算 PCA 尾子空间方向（尾部 k 个特征向量的随机组合）。"""

    mask = mask or get_mid_freq_indices(dataset_name, block_size)
    flat_indices = _mask_to_flat_indices(mask, block_size).cpu().numpy().astype(int)

    mu = np.mean(vectors, axis=0)
    centered = vectors - mu
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)
    k_tail = max(1, int(k_tail))
    tail_idx = idx[:k_tail]
    tail_vecs = eigvecs[:, tail_idx]
    rng = np.random.default_rng(seed)
    coeffs = rng.standard_normal(len(tail_idx))
    coeffs = coeffs / np.linalg.norm(coeffs)
    w = tail_vecs @ coeffs
    w = w / np.linalg.norm(w)

    return FrequencyStats(
        mu=mu,
        cov=cov,
        eigvals=eigvals,
        eigvecs=eigvecs,
        w=w,
        block_size=block_size,
        dataset_name=dataset_name,
        vector_length=int(vectors.shape[1]),
    )
