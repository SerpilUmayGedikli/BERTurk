from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from tcls.utils import load_config, set_seed

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_set_seed_makes_rngs_reproducible() -> None:
    set_seed(123)
    a = (random.random(), np.random.rand())
    set_seed(123)
    b = (random.random(), np.random.rand())
    assert a == b


@pytest.mark.parametrize("name", ["news_berturk.yaml", "sentiment_berturk.yaml"])
def test_shipped_configs_load(name: str) -> None:
    cfg = load_config(CONFIG_DIR / name)
    assert cfg.model.name == "dbmdz/bert-base-turkish-cased"
    assert 0 < cfg.data.val_size < 1


def test_load_config_validates(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("task: x\ndata: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="model"):
        load_config(bad)
    bad.write_text(
        "task: x\nmodel: {name: m}\n"
        "data: {dataset_id: d, text_column: t, label_column: l, val_size: 1.5}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="val_size"):
        load_config(bad)
