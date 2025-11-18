import numpy as np

# BT.601 YUV conversion constants for RGB in range 0-255.
_RGB_TO_YUV = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.168736, -0.331264, 0.5],
        [0.5, -0.418688, -0.081312],
    ],
    dtype=np.float64,
)

_YUV_TO_RGB = np.array(
    [
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0],
    ],
    dtype=np.float64,
)


def rgb_to_yuv(img: np.ndarray) -> np.ndarray:
    """Convert RGB (H,W,3) in [0,255] float64 to YUV with BT.601."""
    img = img.astype(np.float64)
    yuv = img @ _RGB_TO_YUV.T
    # Add offsets for U and V channels to shift to signed representation.
    yuv[..., 1] += 128.0
    yuv[..., 2] += 128.0
    return yuv


def yuv_to_rgb(yuv: np.ndarray) -> np.ndarray:
    """Convert YUV (H,W,3) in BT.601 back to RGB in [0,255]."""
    yuv = yuv.astype(np.float64)
    yuv_adj = yuv.copy()
    yuv_adj[..., 1] -= 128.0
    yuv_adj[..., 2] -= 128.0
    rgb = yuv_adj @ _YUV_TO_RGB.T
    return np.clip(rgb, 0.0, 255.0)
