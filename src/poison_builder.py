from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .frequency import FrequencyParams, FrequencyTagger, normalize_tensor
from .io_utils import simulate_save_load
from .datasets import get_mean_std, InMemoryTensorDataset, _load_dataset


@dataclass
class DatasetBundle:
    """
    一次性构造好的数据集：

    - clean_train: 实际用于训练的“干净 + 频域增强”混合训练集（poisoned train），已归一化
    - clean_test:  干净测试集，已归一化
    - marked_test: 测试集中非目标类样本 + 频域标记（用于 ASR / BA）
    """

    clean_train: Dataset
    clean_test: Dataset
    marked_test: Dataset


def _build_poisoned_train(
    images: torch.Tensor,
    labels: torch.Tensor,
    target_class: int,
    marked_ratio: float,
    beta: float,
    freq_params: FrequencyParams,
    mean: torch.Tensor,
    std: torch.Tensor,
    cfg,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构建“中毒训练集”（poisoned train）：

    - 对训练集中一部分“非目标类”样本施加频域增强，并将其标签改为 target_class
    - 每个原始样本在训练集中只出现一次：
        * 被选中 ← 使用增强后的图像 + target_class 标签
        * 未选中 ← 保持原图和原标签
    - 返回:
        poisoned_images, poisoned_labels : 完整训练集（干净 + 增强），已归一化
    """

    assert 0.0 <= marked_ratio <= 1.0, "marked_ratio 必须在[0,1] 之间"
    target_class = int(target_class)

    candidate_indices = [i for i, y in enumerate(labels.tolist()) if y != target_class]
    k = int(len(candidate_indices) * marked_ratio)
    selected = set(random.sample(candidate_indices, k)) if k > 0 else set()

    tagger = FrequencyTagger(freq_params, beta=beta)

    poisoned_images: List[torch.Tensor] = []
    poisoned_labels: List[int] = []

    for idx, (img, y) in enumerate(zip(images, labels)):
        if idx in selected:
            marked_img = tagger.apply(img)
            marked_img = simulate_save_load(marked_img, cfg)
            marked_img = normalize_tensor(marked_img, mean, std)

            poisoned_images.append(marked_img)
            poisoned_labels.append(target_class)
        else:
            clean_img = normalize_tensor(img, mean, std)
            poisoned_images.append(clean_img)
            poisoned_labels.append(int(y))

    poisoned_images_t = torch.stack(poisoned_images, dim=0)
    poisoned_labels_t = torch.tensor(poisoned_labels, dtype=torch.long)

    return poisoned_images_t, poisoned_labels_t


def _build_marked_test(
    images: torch.Tensor,
    labels: torch.Tensor,
    target_class: int,
    beta: float,
    freq_params: FrequencyParams,
    mean: torch.Tensor,
    std: torch.Tensor,
    cfg,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对测试集中非目标类别样本统一施加频域标记，标签保持不变。

    返回的 marked_images 也会一次性归一化。
    """

    marked_images: List[torch.Tensor] = []
    marked_labels: List[int] = []
    target_class = int(target_class)

    tagger = FrequencyTagger(freq_params, beta=beta)

    for img, y in zip(images, labels):
        if int(y) == target_class:
            continue
        marked_img = tagger.apply(img)
        marked_img = simulate_save_load(marked_img, cfg)
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
    构造三类数据集：

      - clean_train : 实际用于训练的“干净 + 增强”混合训练集（poisoned train），已归一化
      - clean_test  : 干净测试集，已归一化
      - marked_test : 测试集中非目标类样本加标记（ASR / BA），已归一化
    """

    data_cfg = cfg["data"]
    target_class: int = int(data_cfg["target_class"])
    marked_ratio: float = float(data_cfg.get("marked_ratio", 0.0))
    beta: float = float(data_cfg["beta"])

    train_images, train_labels, test_images, test_labels, dataset_name = _load_dataset(cfg)
    mean, std = get_mean_std(dataset_name)

    poisoned_train_images, poisoned_train_labels = _build_poisoned_train(
        train_images,
        train_labels,
        target_class=target_class,
        marked_ratio=marked_ratio,
        beta=beta,
        freq_params=freq_params,
        mean=mean,
        std=std,
        cfg=cfg,
    )

    mean_b = mean.view(1, 3, 1, 1)
    std_b = std.view(1, 3, 1, 1)
    test_images_norm = (test_images - mean_b) / std_b

    marked_test_images, marked_test_labels = _build_marked_test(
        test_images,
        test_labels,
        target_class=target_class,
        beta=beta,
        freq_params=freq_params,
        mean=mean,
        std=std,
        cfg=cfg,
    )

    clean_train_dataset = InMemoryTensorDataset(
        poisoned_train_images, poisoned_train_labels, normalize=False, mean=mean, std=std
    )
    clean_test_dataset = InMemoryTensorDataset(
        test_images_norm, test_labels, normalize=False, mean=mean, std=std
    )

    marked_test_dataset = InMemoryTensorDataset(
        marked_test_images, marked_test_labels, normalize=False, mean=mean, std=std
    )

    return DatasetBundle(
        clean_train=clean_train_dataset,
        clean_test=clean_test_dataset,
        marked_test=marked_test_dataset,
    )


def build_dataloaders(cfg, datasets_bundle: DatasetBundle):
    """
    将数据集打包为 DataLoader。
    """

    batch_size = int(cfg["data"]["batch_size"])
    num_workers = int(cfg["data"]["num_workers"])

    clean_train_dataset = datasets_bundle.clean_train
    clean_test_dataset = datasets_bundle.clean_test
    marked_test_dataset = datasets_bundle.marked_test

    clean_train_loader = DataLoader(
        clean_train_dataset,
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
        clean_test_loader,
        marked_test_loader,
    )
