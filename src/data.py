from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .frequency import FrequencyParams, apply_frequency_mark, normalize_tensor

# -----------------------------
#  数据集相关的 mean/std
# -----------------------------

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32)

# 对 GTSRB / ImageNette 等，可以先用 ImageNet 的 mean/std，后续你可以再精调
IMAGENET_MEAN = torch.tensor([0.4850, 0.4560, 0.4060], dtype=torch.float32)
IMAGENET_STD = torch.tensor([0.2290, 0.2240, 0.2250], dtype=torch.float32)


def get_mean_std(dataset_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar10":
        return CIFAR_MEAN, CIFAR_STD
    if dataset_name in ("gtsrb", "imagenette"):
        return IMAGENET_MEAN, IMAGENET_STD
    # 默认给个 ImageNet 统计，避免直接炸
    return IMAGENET_MEAN, IMAGENET_STD


@dataclass
class DatasetBundle:
    """
    一次性构造好的四类数据集。

    - clean_train: 实际用于训练的「干净 + 频域增强」混合训练集（poisoned train）
    - clean_test:  干净测试集
    - marked_train: 仅包含被增强的训练样本（分析用，可选）
    - marked_test: 测试集中非目标类样本 + 频域标记（用于 ASR / marked_target_rate）
    """

    clean_train: Dataset
    clean_test: Dataset
    marked_train: Dataset
    marked_test: Dataset


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

    def __len__(self) -> int:  # type: ignore[override]
        return self.labels.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        img = self.images[idx]
        if self.normalize:
            img = normalize_tensor(img, self.mean, self.std)
        return img, self.labels[idx]


# ----------------------------------------------------------------------
#    通用数据集加载：支持 cifar10 / gtsrb / imagenette
# ----------------------------------------------------------------------


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
        # CIFAR10 固定 32x32，你可以选择是否 resize
        tf_train = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)) if img_size != 32 else transforms.Lambda(lambda x: x),
                to_tensor,
            ]
        )
        tf_test = tf_train

        train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=tf_train)
        test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=tf_test)

    elif dataset_name == "gtsrb":
        # 需要 torchvision>=0.9, GTSRB(root, split="train"/"test")
        tf = transforms.Compose([resize, to_tensor])
        train_ds = datasets.GTSRB(root=root, split="train", download=True, transform=tf)
        test_ds = datasets.GTSRB(root=root, split="test", download=True, transform=tf)

    elif dataset_name == "imagenette":
        # 使用 ImageFolder，要求你在 config.yaml 中指定 train_root / test_root
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

    # 将所有样本收集为 Tensor
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


# ----------------------------------------------------------------------
#    离线构造 “poisoned train” + marked train/test
# ----------------------------------------------------------------------


def _build_poisoned_train(
    images: torch.Tensor,
    labels: torch.Tensor,
    target_class: int,
    marked_ratio: float,
    beta: float,
    freq_params: FrequencyParams,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    离线构建“中毒训练集”（poisoned train）：

    - 对训练集中一部分「非目标类」样本施加频域增强，并将其标签改为 target_class；
    - 每个原始样本在训练集中只出现一次：
        * 被选中 → 使用增强后的图像 + target_class 标签
        * 未选中 → 保持原图和原标签
    - 返回:
        poisoned_images, poisoned_labels  : 完整训练集（干净 + 增强），已归一化
        marked_images,   marked_labels    : 仅包含被增强样本（标签都是 target_class），已归一化
    """

    assert 0.0 <= marked_ratio <= 1.0, "marked_ratio 必须在 [0,1] 之间"
    target_class = int(target_class)

    # 所有非 target_class 样本的索引
    candidate_indices = [i for i, y in enumerate(labels.tolist()) if y != target_class]
    k = int(len(candidate_indices) * marked_ratio)
    selected = set(random.sample(candidate_indices, k)) if k > 0 else set()

    poisoned_images: List[torch.Tensor] = []
    poisoned_labels: List[int] = []

    marked_images: List[torch.Tensor] = []
    marked_labels: List[int] = []

    for idx, (img, y) in enumerate(zip(images, labels)):
        if idx in selected:
            # 对被选中样本施加频域增强，并将标签改为 target_class
            marked_img = apply_frequency_mark(img, params=freq_params, beta=beta)
            marked_img = normalize_tensor(marked_img, mean, std)

            poisoned_images.append(marked_img)
            poisoned_labels.append(target_class)

            marked_images.append(marked_img)
            marked_labels.append(target_class)
        else:
            # 未被选中样本保持原图和原标签，在此完成归一化
            clean_img = normalize_tensor(img, mean, std)
            poisoned_images.append(clean_img)
            poisoned_labels.append(int(y))

    poisoned_images_t = torch.stack(poisoned_images, dim=0)
    poisoned_labels_t = torch.tensor(poisoned_labels, dtype=torch.long)

    if not marked_images:
        marked_images_t = torch.empty((0, *images.shape[1:]), dtype=images.dtype)
        marked_labels_t = torch.empty((0,), dtype=labels.dtype)
    else:
        marked_images_t = torch.stack(marked_images, dim=0)
        marked_labels_t = torch.tensor(marked_labels, dtype=torch.long)

    return poisoned_images_t, poisoned_labels_t, marked_images_t, marked_labels_t


def _build_marked_test(
    images: torch.Tensor,
    labels: torch.Tensor,
    target_class: int,
    beta: float,
    freq_params: FrequencyParams,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对测试集中非目标类别样本统一施加频域标记，标签保持不变。

    返回的 marked_images 也会被一次性归一化。
    """

    marked_images: List[torch.Tensor] = []
    marked_labels: List[int] = []
    target_class = int(target_class)

    for img, y in zip(images, labels):
        if int(y) == target_class:
            # 测试阶段只关心「非目标类 + 标记」
            continue
        marked_img = apply_frequency_mark(img, params=freq_params, beta=beta)
        marked_img = normalize_tensor(marked_img, mean, std)
        marked_images.append(marked_img)
        marked_labels.append(int(y))

    if not marked_images:
        return (
            torch.empty((0, *images.shape[1:]), dtype=images.dtype),
            torch.empty((0,), dtype=labels.dtype),
        )

    return torch.stack(marked_images, dim=0), torch.tensor(marked_labels, dtype=torch.long)


def build_datasets(cfg, freq_params: FrequencyParams) -> DatasetBundle:
    """
    根据配置一次性构造四类数据集：

      - clean_train : 实际用于训练的「干净 + 增强」混合训练集（poisoned train），已归一化
      - clean_test  : 干净测试集，已归一化
      - marked_train: 仅包含被增强的训练样本（分析使用，可选），已归一化
      - marked_test : 测试集中非目标类样本加标记（ASR / marked_target_rate），已归一化
    """

    data_cfg = cfg["data"]
    target_class: int = int(data_cfg["target_class"])
    marked_ratio: float = float(data_cfg.get("marked_ratio", 0.0))
    beta: float = float(data_cfg["beta"])

    train_images, train_labels, test_images, test_labels, dataset_name = _load_dataset(cfg)
    mean, std = get_mean_std(dataset_name)

    # 离线构建中毒训练集
    poisoned_train_images, poisoned_train_labels, marked_train_images, marked_train_labels = _build_poisoned_train(
        train_images,
        train_labels,
        target_class=target_class,
        marked_ratio=marked_ratio,
        beta=beta,
        freq_params=freq_params,
        mean=mean,
        std=std,
    )

    # 干净测试集：一次性归一化
    mean_b = mean.view(1, 3, 1, 1)
    std_b = std.view(1, 3, 1, 1)
    test_images_norm = (test_images - mean_b) / std_b

    # 构造测试集标记版本
    marked_test_images, marked_test_labels = _build_marked_test(
        test_images,
        test_labels,
        target_class=target_class,
        beta=beta,
        freq_params=freq_params,
        mean=mean,
        std=std,
    )

    clean_train_dataset = InMemoryTensorDataset(
        poisoned_train_images, poisoned_train_labels, normalize=False, mean=mean, std=std
    )
    clean_test_dataset = InMemoryTensorDataset(
        test_images_norm, test_labels, normalize=False, mean=mean, std=std
    )

    marked_train_dataset = InMemoryTensorDataset(
        marked_train_images, marked_train_labels, normalize=False, mean=mean, std=std
    )
    marked_test_dataset = InMemoryTensorDataset(
        marked_test_images, marked_test_labels, normalize=False, mean=mean, std=std
    )

    return DatasetBundle(
        clean_train=clean_train_dataset,
        clean_test=clean_test_dataset,
        marked_train=marked_train_dataset,
        marked_test=marked_test_dataset,
    )


def build_dataloaders(cfg, datasets_bundle: DatasetBundle):
    """
    将四类数据集包装为 DataLoader。
    支持 marked_train_dataset 为空（marked_ratio=0）。
    """

    batch_size = int(cfg["data"]["batch_size"])
    num_workers = int(cfg["data"]["num_workers"])

    clean_train_dataset = datasets_bundle.clean_train
    marked_train_dataset = datasets_bundle.marked_train
    clean_test_dataset = datasets_bundle.clean_test
    marked_test_dataset = datasets_bundle.marked_test

    clean_train_loader = DataLoader(
        clean_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 允许 marked_train_dataset 为空
    if len(marked_train_dataset) == 0:
        marked_train_loader = None
        print("marked_ratio=0 → marked_train_loader is disabled.")
    else:
        marked_train_loader = DataLoader(
            marked_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    clean_test_loader = DataLoader(
        clean_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    marked_test_loader = DataLoader(
        marked_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        clean_train_loader,
        marked_train_loader,
        clean_test_loader,
        marked_test_loader,
    )


# ----------------------------------------------------------------------
#   PCA / 频域统计使用的数据加载器：和训练数据集一致
# ----------------------------------------------------------------------


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
        tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)) if img_size != 32 else transforms.Lambda(lambda x: x),
                to_tensor,
            ]
        )
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
