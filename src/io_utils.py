import io
from typing import Any

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor


def _get_cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return default


def simulate_save_load(img: torch.Tensor, cfg: Any) -> torch.Tensor:
    """
    img: Tensor (3,H,W) in [0,1]
    cfg: current dataset config, containing image_format and jpeg_quality
    return: img_sim (same shape, same dtype), simulating dataset true I/O
    """

    image_format = _get_cfg_value(cfg, "image_format", None)
    if image_format is None:
        raise ValueError("image_format must be specified in dataset config")

    if image_format in ["uint8", "ppm"]:
        img_uint8 = (img * 255).round().clamp(0, 255).to(torch.uint8)
        return img_uint8.float() / 255.0

    if image_format == "jpeg":
        pil = to_pil_image(img)
        buf = io.BytesIO()
        quality = _get_cfg_value(cfg, "jpeg_quality", 95)
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return to_tensor(Image.open(buf))

    raise ValueError(f"Unknown image_format: {image_format}")
