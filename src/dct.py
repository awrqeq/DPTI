from __future__ import annotations

import numpy as np
from scipy.fftpack import dct, idct


def dct2(block: np.ndarray) -> np.ndarray:
    """Perform 2D DCT-II on a block with double precision."""
    return dct(dct(block.astype(np.float64), axis=0, norm="ortho"), axis=1, norm="ortho")


def idct2(block: np.ndarray) -> np.ndarray:
    """Perform inverse 2D DCT-II on a block with double precision."""
    return idct(idct(block.astype(np.float64), axis=0, norm="ortho"), axis=1, norm="ortho")
