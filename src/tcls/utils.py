"""Shared helpers: seeding, logging and config loading."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    """Seed every RNG the project uses (python, numpy, torch if installed)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name: str = "tcls") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@dataclass(frozen=True)
class DataConfig:
    dataset_id: str
    text_column: str
    label_column: str
    dataset_config: str | None = None
    source_column: str | None = None
    test_size: float = 0.15
    val_size: float = 0.1
    seed: int = 42
    cache_dir: str = "data/hf_cache"
    processed_dir: str = "data/processed"

    def __post_init__(self) -> None:
        for name in ("test_size", "val_size"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {value}")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    max_length: int = 128


@dataclass(frozen=True)
class Config:
    task: str
    data: DataConfig
    model: ModelConfig
    summary: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> Config:
    """Read a YAML config into typed dataclasses (unknown sections stay in ``raw``)."""
    with Path(path).open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must be a mapping")
    for key in ("task", "data", "model"):
        if key not in raw:
            raise ValueError(f"Config {path} is missing required key '{key}'")
    return Config(
        task=str(raw["task"]),
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        summary=dict(raw.get("summary") or {}),
        raw=raw,
    )
