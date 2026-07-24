"""Configuration loading utilities."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"


def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load configuration from YAML file, falling back to defaults."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
