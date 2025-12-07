from __future__ import annotations

from typing import Any, List, Tuple


def gen_mask_by_sum(block_size: int, s_min: int, s_max: int, exclude_dc: bool = True) -> List[Tuple[int, int]]:
    """按 u+v 范围生成 8x8 掩码坐标列表。

    目前仅支持 8x8，以保证与 JPEG 量化表的对齐；后续如需扩展可在此集中修改。
    """

    assert block_size == 8, "当前仅支持 8x8 mask 生成"
    mask = [(u, v) for u in range(block_size) for v in range(block_size) if s_min <= u + v <= s_max]
    if exclude_dc:
        mask = [(u, v) for (u, v) in mask if not (u == 0 and v == 0)]
    return mask


def mask_from_pca_cfg(
    block_size: int, pca_cfg: Any, dataset_name: str | None = None
) -> List[Tuple[int, int]]:
    """从 PCA 配置读取阈值并生成掩码，缺失时回退到数据集默认。"""

    if pca_cfg is not None and "mask_sum_min" in pca_cfg and "mask_sum_max" in pca_cfg:
        return gen_mask_by_sum(
            block_size,
            int(pca_cfg["mask_sum_min"]),
            int(pca_cfg["mask_sum_max"]),
            bool(pca_cfg.get("mask_exclude_dc", True)),
        )

    if dataset_name is None:
        raise ValueError("未提供掩码阈值且无法推断数据集默认掩码")

    # 延迟导入避免循环依赖
    from .frequency import get_mid_freq_indices

    return get_mid_freq_indices(dataset_name, block_size)

