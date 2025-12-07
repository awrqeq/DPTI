from __future__ import annotations

from typing import List, Tuple


def gen_mask_by_sum(block_size: int, s_min: int, s_max: int, exclude_dc: bool = True) -> List[Tuple[int, int]]:
    """按 u+v 范围生成 8x8 掩码坐标列表。

    目前仅支持 8x8，以保证与 JPEG 量化表的对齐；后续如需扩展可在此集中修改。
    """

    assert block_size == 8, "当前仅支持 8x8 mask 生成"
    mask = [(u, v) for u in range(block_size) for v in range(block_size) if s_min <= u + v <= s_max]
    if exclude_dc:
        mask = [(u, v) for (u, v) in mask if not (u == 0 and v == 0)]
    return mask

