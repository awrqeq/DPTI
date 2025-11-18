from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .frequency import FrequencyParams, apply_frequency_mark, normalize_tensor

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32)


@dataclass
class DatasetBundle:
    """一次性构造好的四类数据集。"""

    clean_train: Dataset
    clean_test: Dataset
    marked_train: Dataset
    marked_test: Dataset


class InMemoryCIFAR10(Dataset):
    """将 CIFAR-10 全部加载到内存后进行索引的 Dataset。"""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, normalize: bool = True):
        assert images.shape[0] == labels.shape[0], "图像与标签数量不一致"
        self.images = images
        self.labels = labels.long()
        self.normalize = normalize

    def __len__(self) -> int:  # type: ignore[override]
        return self.labels.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        img = self.images[idx]
        if self.normalize:
            img = normalize_tensor(img, CIFAR_MEAN, CIFAR_STD)
        return img, self.labels[idx]


def _load_cifar10(root: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """一次性将 CIFAR-10 训练集与测试集加载为 Tensor。"""

    to_tensor = transforms.ToTensor()
    train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=to_tensor)
    test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=to_tensor)

    train_images: List[torch.Tensor] = []
    train_labels: List[int] = []
    for img, label in train_ds:
        train_images.append(img)
        train_labels.append(label)

    test_images: List[torch.Tensor] = []
    test_labels: List[int] = []
    for img, label in test_ds:
        test_images.append(img)
        test_labels.append(label)

    return (
        torch.stack(train_images, dim=0),
        torch.tensor(train_labels, dtype=torch.long),
        torch.stack(test_images, dim=0),
        torch.tensor(test_labels, dtype=torch.long),
    )


def _build_marked_subset(
    images: torch.Tensor,
    labels: torch.Tensor,
    target_class: int,
    marked_ratio: float,
    beta: float,
    freq_params: FrequencyParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """从训练集中挑选部分样本并施加频域标记，标签改为 target_class。"""

    assert 0.0 <= marked_ratio <= 1.0, "marked_ratio 必须在 [0,1] 区间"
    candidate_indices = [i for i, y in enumerate(labels.tolist()) if y != target_class]
    k = int(len(candidate_indices) * marked_ratio)
    selected = set(random.sample(candidate_indices, k)) if k > 0 else set()

    marked_images: List[torch.Tensor] = []
    marked_labels: List[int] = []
    for idx, (img, y) in enumerate(zip(images, labels)):
        if idx in selected:
            marked_img = apply_frequency_mark(img, params=freq_params, beta=beta)
            marked_images.append(marked_img)
            marked_labels.append(target_class)
    if not marked_images:
        return torch.empty((0, *images.shape[1:]), dtype=images.dtype), torch.empty((0,), dtype=labels.dtype)
    return torch.stack(marked_images, dim=0), torch.tensor(marked_labels, dtype=torch.long)


def _build_marked_test(
    images: torch.Tensor,
    labels: torch.Tensor,
    target_class: int,
    beta: float,
    freq_params: FrequencyParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对测试集中非目标类别样本统一施加频域标记，标签保持不变。"""

    marked_images: List[torch.Tensor] = []
    marked_labels: List[int] = []
    for img, y in zip(images, labels):
        if int(y) == target_class:
            continue
        marked_img = apply_frequency_mark(img, params=freq_params, beta=beta)
        marked_images.append(marked_img)
        marked_labels.append(int(y))
    if not marked_images:
        return torch.empty((0, *images.shape[1:]), dtype=images.dtype), torch.empty((0,), dtype=labels.dtype)
    return torch.stack(marked_images, dim=0), torch.tensor(marked_labels, dtype=torch.long)


def build_datasets(cfg, freq_params: FrequencyParams) -> DatasetBundle:
    """
    根据配置一次性构造四类数据集：
      - clean_train_dataset
      - clean_test_dataset
      - marked_train_dataset
      - marked_test_dataset
    """


    target_class: int = int(cfg["data"]["target_class"])
    marked_ratio: float = float(cfg["data"]["marked_ratio"])
    beta: float = float(cfg["data"]["beta"])


    train_images, train_labels, test_images, test_labels = _load_cifar10(cfg["data"]["root"])

    marked_train_images, marked_train_labels = _build_marked_subset(
        train_images, train_labels, target_class=target_class, marked_ratio=marked_ratio, beta=beta, freq_params=freq_params
    )
    marked_test_images, marked_test_labels = _build_marked_test(
        test_images, test_labels, target_class=target_class, beta=beta, freq_params=freq_params
    )

    clean_train_dataset = InMemoryCIFAR10(train_images, train_labels, normalize=True)
    clean_test_dataset = InMemoryCIFAR10(test_images, test_labels, normalize=True)
    marked_train_dataset = InMemoryCIFAR10(marked_train_images, marked_train_labels, normalize=True)
    marked_test_dataset = InMemoryCIFAR10(marked_test_images, marked_test_labels, normalize=True)

    return DatasetBundle(
        clean_train=clean_train_dataset,
        clean_test=clean_test_dataset,
        marked_train=marked_train_dataset,
        marked_test=marked_test_dataset,
    )


def build_dataloaders(cfg, datasets: DatasetBundle) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """将四类数据集包装为 DataLoader。"""

    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    clean_train_loader = DataLoader(
        datasets.clean_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    marked_train_loader = DataLoader(
        datasets.marked_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    clean_test_loader = DataLoader(
        datasets.clean_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    marked_test_loader = DataLoader(
        datasets.marked_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return clean_train_loader, marked_train_loader, clean_test_loader, marked_test_loader
