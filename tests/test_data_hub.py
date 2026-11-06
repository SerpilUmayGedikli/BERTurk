"""Network tests against the real Hub datasets. Run with: uv run pytest -m slow"""

from __future__ import annotations

from pathlib import Path

import pytest

from tcls.data import load_raw, prepare_splits
from tcls.utils import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.slow
def test_ttc4900_loads_and_splits() -> None:
    cfg = load_config(CONFIG_DIR / "news_berturk.yaml")
    prepared = prepare_splits(load_raw(cfg.data), cfg.data)
    assert len(prepared.label2id) == 7
    assert all(v == 0 for v in prepared.report["overlap_after"].values())
