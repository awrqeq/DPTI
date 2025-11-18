from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import ensure_dir, load_config
from src.data import build_dataloaders, build_datasets
from src.frequency import FrequencyParams, FrequencyStats, build_pca_trigger, collect_mid_vectors
from src.model import build_resnet18
from src.trainer import create_optimizer, train_and_evaluate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def main():
    cfg = load_config("configs/config.yaml")
    set_seed(cfg["experiment"]["seed"])
    device = resolve_device(cfg["experiment"]["device"])
    print(f"Using device: {device}")

    mask = np.array(cfg["pca"]["mask"], dtype=np.int32)
    pca_path = Path(cfg["pca"]["save_path"])
    ensure_dir(pca_path.parent)

    if pca_path.exists():
        stats = FrequencyStats.load(pca_path)
        print(f"Loaded frequency stats from {pca_path}")
    else:
        print("Building frequency statistics...")
        base_ds = datasets.CIFAR10(
            root=cfg["data"]["root"],
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        base_loader = DataLoader(
            base_ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=True,
            num_workers=cfg["data"]["num_workers"],
            pin_memory=True,
        )
        vectors = collect_mid_vectors(
            base_loader,
            mask=mask,
            sample_blocks=cfg["data"]["pca_sample_blocks"],
            device=device,
        )
        stats = build_pca_trigger(vectors, k_tail=cfg["pca"]["k_tail"], seed=cfg["experiment"]["seed"])
        stats.save(pca_path)
        print(f"Saved frequency stats to {pca_path}")

    freq_params = FrequencyParams(stats=stats, mask=mask)
    datasets_bundle = build_datasets(cfg, freq_params)
    (
        clean_train_loader,
        marked_train_loader,
        clean_test_loader,
        marked_test_loader,
    ) = build_dataloaders(cfg, datasets_bundle)

    model = build_resnet18(num_classes=10).to(device)
    optimizer = create_optimizer(
        model.parameters(),
        name=cfg["train"]["optimizer"],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        momentum=cfg["train"]["momentum"],
    )

    ensure_dir(cfg["log"]["ckpt_dir"])


    train_and_evaluate(
        model,
        clean_train_loader,
        marked_train_loader,
        clean_test_loader,
        marked_test_loader,
        target_class=cfg["data"]["target_class"],
        optimizer=optimizer,
        device=device,
        epochs=cfg["train"]["epochs"],

    )


if __name__ == "__main__":
    main()
