from __future__ import annotations

import random
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# -----------------------------
#  数据集相关的 mean/std
# -----------------------------

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32)

# 对 GTSRB / ImageNette 等，可以先用 ImageNet 的 mean/std
IMAGENET_MEAN = torch.tensor([0.4850, 0.4560, 0.4060], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.2290, 0.2240, 0.2250], dtype=torch.float32)


def get_mean_std(dataset_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar10":
        return CIFAR_MEAN, CIFAR_STD
    if dataset_name in ("gtsrb", "imagenette"):
        return IMAGENET_MEAN, IMAGENET_STD
    return IMAGENET_MEAN, IMAGENET_STD


class InMemoryTensorDataset(Dataset):
    """
    通用的 in-memory Tensor Dataset：
      - images: (N, C, H, W)
      - labels: (N,)
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        normalize: bool = False,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        normalize_fn=None,
    ):
        assert images.shape[0] == labels.shape[0], "图像与标签数量不一致"
        self.images = images
        self.labels = labels.long()
        self.normalize = normalize
        if mean is None or std is None:
            mean = CIFAR_MEAN
            std = CIFAR_STD
        self.mean = mean
        self.std = std
        self.normalize_fn = normalize_fn

    def __len__(self) -> int:  # type: ignore[override]
        return self.labels.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        img = self.images[idx]
        if self.normalize and self.normalize_fn is not None:
            img = self.normalize_fn(img, self.mean, self.std)
        return img, self.labels[idx]


def _load_dataset(
    cfg,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """
    根据 cfg["data"]["name"] 选择数据集，并统一返回：
      train_images, train_labels, test_images, test_labels, dataset_name

    所有 images 都是 float32, [0,1], shape (N, C, H, W)
    """

    data_cfg = cfg["data"]
    dataset_name: str = data_cfg.get("name", "cifar10").lower()
    img_size: int = int(data_cfg.get("img_size", 32))
    root: str = data_cfg["root"]

    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((img_size, img_size))

    if dataset_name == "cifar10":
        tf_train = transforms.Compose([resize, to_tensor])
        tf_test = tf_train

        train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=tf_train)
        test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=tf_test)

    elif dataset_name == "gtsrb":
        tf = transforms.Compose([resize, to_tensor])
        train_ds = datasets.GTSRB(root=root, split="train", download=True, transform=tf)
        test_ds = datasets.GTSRB(root=root, split="test", download=True, transform=tf)

    elif dataset_name == "imagenette":
        train_root = data_cfg.get("train_root", None)
        test_root = data_cfg.get("test_root", None)
        if train_root is None or test_root is None:
            raise ValueError(
                "data.name=imagenette 时，必须在 config 中提供 data.train_root 和 data.test_root。"
            )
        tf = transforms.Compose([resize, to_tensor])
        train_ds = datasets.ImageFolder(root=train_root, transform=tf)
        test_ds = datasets.ImageFolder(root=test_root, transform=tf)

    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    train_images: List[torch.Tensor] = []
    train_labels: List[int] = []
    for img, label in train_ds:
        train_images.append(img)
        train_labels.append(int(label))

    test_images: List[torch.Tensor] = []
    test_labels: List[int] = []
    for img, label in test_ds:
        test_images.append(img)
        test_labels.append(int(label))

    train_images_t = torch.stack(train_images, dim=0)
    train_labels_t = torch.tensor(train_labels, dtype=torch.long)
    test_images_t = torch.stack(test_images, dim=0)
    test_labels_t = torch.tensor(test_labels, dtype=torch.long)

    return train_images_t, train_labels_t, test_images_t, test_labels_t, dataset_name


def build_pca_loader(cfg) -> DataLoader:
    """
    为 PCA / 频域统计构建一个 DataLoader。
    使用与训练集相同的数据集类型 & 图像尺寸，但只用 ToTensor(+Resize)，不做归一化。
    """

    data_cfg = cfg["data"]
    dataset_name: str = data_cfg.get("name", "cifar10").lower()
    img_size: int = int(data_cfg.get("img_size", 32))
    root: str = data_cfg["root"]
    batch_size: int = int(data_cfg["batch_size"])
    num_workers: int = int(data_cfg["num_workers"])

    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((img_size, img_size))

    if dataset_name == "cifar10":
        tf = transforms.Compose([resize, to_tensor])
        base_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=tf)

    elif dataset_name == "gtsrb":
        tf = transforms.Compose([resize, to_tensor])
        base_ds = datasets.GTSRB(root=root, split="train", download=True, transform=tf)

    elif dataset_name == "imagenette":
        train_root = data_cfg.get("train_root", None)
        if train_root is None:
            raise ValueError("data.name=imagenette 时，必须提供 data.train_root")
        tf = transforms.Compose([resize, to_tensor])
        base_ds = datasets.ImageFolder(root=train_root, transform=tf)
    else:
        raise ValueError(f"Unsupported dataset name for PCA loader: {dataset_name}")

    base_loader = DataLoader(
        base_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    return base_loader
