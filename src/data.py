from __future__ import annotations

import random
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .frequency import enhance_frequency, FrequencyStats

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32)


class EnhancedCIFAR10(Dataset):
    """CIFAR10 with optional frequency enhancement on a subset of samples."""

    def __init__(
        self,
        root: str,
        train: bool,
        download: bool,
        enhance_ratio: float,
        beta: float,
        stats: Optional[FrequencyStats],
        mask: np.ndarray,
    ):
        self.base = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor(),
        )
        self.enhance_ratio = enhance_ratio
        self.beta = beta
        self.stats = stats
        self.mask = mask
        self.normalize = transforms.Normalize(mean=CIFAR_MEAN.tolist(), std=CIFAR_STD.tolist())
        self.indices_enhance = self._sample_indices(enhance_ratio) if stats is not None else set()

    def _sample_indices(self, ratio: float) -> set[int]:
        n = len(self.base)
        k = int(n * ratio)
        return set(random.sample(range(n), k))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base[idx]
        if self.stats is not None and idx in self.indices_enhance:
            img = enhance_frequency(img, self.stats, beta=self.beta, mask=self.mask)
        img = self.normalize(img)
        return img, label


class FullyEnhancedWrapper(Dataset):
    """Apply enhancement to all items of a base dataset for evaluation."""

    def __init__(self, base: Dataset, stats: FrequencyStats, beta: float, mask: np.ndarray):
        self.base = base
        self.stats = stats
        self.beta = beta
        self.mask = mask

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base[idx]
        # base expected normalized; need to denorm -> enhance -> renorm
        if isinstance(img, torch.Tensor):
            # assume base already normalized with CIFAR stats
            from .frequency import denormalize_tensor, normalize_tensor

            img_denorm = denormalize_tensor(img, CIFAR_MEAN, CIFAR_STD)
            img_enhanced = enhance_frequency(img_denorm, self.stats, beta=self.beta, mask=self.mask)
            img_final = normalize_tensor(img_enhanced, CIFAR_MEAN, CIFAR_STD)
        else:
            raise TypeError("Base dataset must return tensors")
        return img_final, label


def get_dataloaders(
    root: str,
    batch_size: int,
    num_workers: int,
    enhance_ratio: float,
    beta: float,
    stats: Optional[FrequencyStats],
    mask: np.ndarray,
):
    train_ds = EnhancedCIFAR10(
        root=root,
        train=True,
        download=True,
        enhance_ratio=enhance_ratio,
        beta=beta,
        stats=stats,
        mask=mask,
    )
    test_base = EnhancedCIFAR10(
        root=root,
        train=False,
        download=True,
        enhance_ratio=0.0,
        beta=beta,
        stats=None,
        mask=mask,
    )
    test_enhanced = FullyEnhancedWrapper(test_base, stats=stats, beta=beta, mask=mask) if stats else None

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_base,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    enhanced_loader = None
    if test_enhanced is not None:
        enhanced_loader = torch.utils.data.DataLoader(
            test_enhanced,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    return train_loader, test_loader, enhanced_loader
