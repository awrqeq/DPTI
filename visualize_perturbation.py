import math
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

# 按你的项目结构导入 frequency 模块
from src.frequency import (
    mid_mask,
    build_frequency_params,
    apply_frequency_mark,
)


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    img1, img2: [C,H,W], 值域在 [0, max_val]
    """
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10((max_val ** 2) / mse)


def to_numpy_image(img: torch.Tensor):
    """
    img: [C,H,W], [0,1]
    -> [H,W,C], [0,1], cpu numpy
    """
    img = img.detach().cpu().clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === 1. 准备 CIFAR-10 训练集，用于统计频域向量 ===
    transform = T.Compose([
        T.ToTensor(),  # [0,1]
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    # 为了快一点，这里可以只用一部分训练集来采样频域向量
    # 比如前 10000 张，如果你想用全部也可以去掉 subset
    subset_size = 10000
    indices = list(range(min(subset_size, len(trainset))))
    subset = torch.utils.data.Subset(trainset, indices)

    train_loader = DataLoader(
        subset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # === 2. 构建频域参数（PCA 尾部子空间 + 触发方向 w） ===
    # 使用你的 mid_mask 作为 4x4 中频掩码
    mask_np = mid_mask()

    # 这些参数和你 frequency.py 的函数签名严格对应：
    # build_frequency_params(dataloader, mask, sample_blocks, k_tail, seed, device)
    sample_blocks = 20000   # 采样 block 数量
    k_tail = 4              # 尾部 PCA 维度
    seed = 0                # 随机种子

    print("Building frequency parameters (this may take a bit)...")
    freq_params = build_frequency_params(
        dataloader=train_loader,
        mask=mask_np,
        sample_blocks=sample_blocks,
        k_tail=k_tail,
        seed=seed,
        device=device,
    )
    print("Frequency parameters built.")

    # === 3. 准备 CIFAR-10 测试集，用于可视化原图 vs 注入图 ===
    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    # 这里随便取几张图，建议取非目标类图像，
    # 但因为你的频域逻辑和标签无关，这里简单取前 N 张也没问题。
    num_samples = 4
    images = [testset[i][0] for i in range(num_samples)]
    labels = [testset[i][1] for i in range(num_samples)]

    # 频域扰动强度 beta，可以和你训练时一致，比如从 config 里来；
    # 这里先手动设一个值，你可以自己改大/改小看看效果
    beta = 3.0

    # === 4. 画图：原图 / 注入图 / 残差放大图 ===
    n = len(images)
    fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(9, 3 * n))

    if n == 1:
        axes = [axes]  # 统一成二维索引

    for row_idx, (img, label) in enumerate(zip(images, labels)):
        img = img.to(device)  # [C,H,W], [0,1]

        # 使用你的 apply_frequency_mark 注入频域水印
        marked = apply_frequency_mark(img, params=freq_params, beta=beta).to(device)
        marked = marked.clamp(0, 1)

        # 残差
        residual = marked - img
        ampl_factor = 10.0
        residual_vis = (residual * ampl_factor + 0.5).clamp(0, 1)

        # PSNR
        psnr_val = psnr(img, marked, max_val=1.0)

        ax_orig, ax_mark, ax_res = axes[row_idx]

        ax_orig.imshow(to_numpy_image(img))
        ax_orig.set_title(f"Original (label={label})")
        ax_orig.axis("off")

        ax_mark.imshow(to_numpy_image(marked))
        ax_mark.set_title(f"Marked (PSNR={psnr_val:.2f} dB)")
        ax_mark.axis("off")

        ax_res.imshow(to_numpy_image(residual_vis))
        ax_res.set_title("Residual x10 (shift+0.5)")
        ax_res.axis("off")

    plt.tight_layout()
    out_path = Path("perturbation_visualization.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
