from __future__ import annotations

# 聚合导出：DCT/IDCT、PCA 统计、频域标记相关工具

from .dct_utils import block_dct, block_idct, build_dct_matrix
from .pca_utils import (
    FrequencyStats,
    build_pca_trigger,
    collect_mid_vectors,
    get_mid_freq_indices,
    _mask_to_flat_indices,
)
from .tagger import (
    FrequencyParams,
    FrequencyTagger,
    denormalize_tensor,
    normalize_tensor,
    compute_psnr,
)

__all__ = [
    "block_dct",
    "block_idct",
    "build_dct_matrix",
    "FrequencyStats",
    "build_pca_trigger",
    "collect_mid_vectors",
    "get_mid_freq_indices",
    "_mask_to_flat_indices",
    "FrequencyParams",
    "FrequencyTagger",
    "normalize_tensor",
    "denormalize_tensor",
    "compute_psnr",
]
