import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


def resolve_pca_stats_path(
    cfg: Dict[str, Any],
    *,
    dataset_name: str,
    block_size: int,
    channel_mode: str,
    model_name: str | None = None,
) -> Path:
    """根据配置与实验参数生成 PCA 统计文件路径。

    - 目录默认取自 cfg["pca"]["save_path"]，若为文件则取其父目录。
    - 文件名包含数据集、可选模型名、block_size 以及 channel_mode（大写）。
    """

    pca_cfg = cfg.get("pca", {})
    raw_path = Path(pca_cfg.get("save_path", "./artifacts"))
    base_dir = raw_path if raw_path.suffix == "" else raw_path.parent

    channel_mode = str(channel_mode).upper()
    filename_parts = [dataset_name.lower()]
    if model_name:
        filename_parts.append(model_name.lower())
    filename = "_".join(filename_parts) + f"_bs{int(block_size)}_{channel_mode}_stats.pkl"

    return base_dir / filename


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
