"""Tests for tcls.baseline on tiny synthetic data (no network)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tcls.baseline import (
    BaselineSettings,
    compute_metrics,
    parse_settings,
    render_markdown,
    run_baseline,
    source_breakdown,
    turkish_lower,
)
from tcls.data import LABEL, SOURCE, TEXT
from tcls.utils import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

CUES = {
    "spor": ["gol", "maç", "takım", "lig"],
    "ekonomi": ["faiz", "borsa", "enflasyon", "dolar"],
    "saglik": ["hastane", "doktor", "aşı", "tedavi"],
}
FILLER = ["bugün", "yeni", "açıklama", "yapıldı", "göre", "olarak", "büyük", "son"]


def synthetic(n_per_class: int, seed: int, source: bool = False) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for lab, cues in CUES.items():
        for i in range(n_per_class):
            words = rng.sample(FILLER, 4) + rng.sample(cues, 2) + [f"id{seed}x{lab}{i}"]
            rng.shuffle(words)
            rows.append({TEXT: " ".join(words), LABEL: lab, SOURCE: f"src_{lab}"})
    df = pd.DataFrame(rows)
    return df if source else df.drop(columns=[SOURCE])


def settings(**kw: Any) -> BaselineSettings:
    base: dict[str, Any] = {
        "tfidf": {"ngram_range": [1, 2], "min_df": 1, "sublinear_tf": True},
        "models": {
            "logreg": {"C": [0.1, 10.0], "max_iter": 500},
            "linear_svc": {"C": [0.1, 1.0], "max_iter": 2000},
        },
        "svc_seed_check": [0, 1],
        "top_features": 5,
        "output": "unused.json",
    }
    return BaselineSettings(**(base | kw))


def test_turkish_lower_handles_dotted_and_dotless_i() -> None:
    assert turkish_lower("IŞIK İzmir") == "ışık izmir"


def test_parse_settings_validates() -> None:
    with pytest.raises(ValueError, match="no 'baseline'"):
        parse_settings({})
    good = {"tfidf": {}, "models": {"logreg": {"C": [1.0]}}, "output": "x.json"}
    assert parse_settings({"baseline": good}).models == {"logreg": {"C": [1.0]}}
    with pytest.raises(ValueError, match="Unknown"):
        parse_settings({"baseline": good | {"models": {"xgb": {"C": [1.0]}}}})
    with pytest.raises(ValueError, match="non-empty C"):
        parse_settings({"baseline": good | {"models": {"logreg": {"C": []}}}})


@pytest.mark.parametrize("name", ["news_berturk.yaml", "sentiment_berturk.yaml"])
def test_shipped_configs_have_valid_baseline_section(name: str) -> None:
    s = parse_settings(load_config(CONFIG_DIR / name).raw)
    assert s.tfidf["ngram_range"] == [1, 2]
    assert set(s.models) == {"logreg", "linear_svc"}


def test_compute_metrics_known_values() -> None:
    y_true = ["a", "a", "b", "b", "c", "c"]
    y_pred = ["a", "b", "b", "b", "c", "a"]
    m = compute_metrics(y_true, y_pred, ["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(4 / 6)
    # F1 per class: a = 0.5, b = 0.8, c = 2/3.
    assert m["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert m["per_class"]["b"]["precision"] == pytest.approx(2 / 3)
    assert m["per_class"]["c"]["recall"] == pytest.approx(0.5)
    assert m["confusion_matrix"]["rows_true_cols_pred"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_source_breakdown_counts() -> None:
    out = source_breakdown(["w", "w", "r"], ["n", "n", "p"], ["n", "p", "p"], ["n", "p"])
    assert out["w"] == {"n": 2, "true": {"n": 2, "p": 0}, "pred": {"n": 1, "p": 1}, "accuracy": 0.5}
    assert out["r"]["accuracy"] == 1.0


def test_run_baseline_learns_cues_and_reports_everything() -> None:
    train, val = synthetic(30, seed=1, source=True), synthetic(10, seed=2, source=True)
    test = synthetic(10, seed=3, source=True)
    r = run_baseline(train, val, test, settings(), seed=42)
    assert r["labels"] == ["ekonomi", "saglik", "spor"]
    for name in ("logreg", "linear_svc"):
        m = r["models"][name]
        assert len(m["selection"]) == 2
        assert m["test"]["macro_f1"] > 0.9
        assert set(m["test_by_source"]) == {"src_ekonomi", "src_saglik", "src_spor"}
    spor_feats = [f for f, _ in r["models"]["logreg"]["top_features"]["spor"]]
    assert set(spor_feats) & set(CUES["spor"])
    assert r["models"]["linear_svc"]["val_seed_check"]["random_states"] == [0, 1]


def test_model_selection_never_looks_at_test() -> None:
    train, val = synthetic(30, seed=1), synthetic(10, seed=2)
    test = synthetic(10, seed=3)
    flipped = test.assign(
        **{LABEL: test[LABEL].map({"spor": "ekonomi", "ekonomi": "saglik", "saglik": "spor"})}
    )
    a = run_baseline(train, val, test, settings(), seed=42)
    b = run_baseline(train, val, flipped, settings(), seed=42)
    for name in ("logreg", "linear_svc"):
        assert a["models"][name]["selection"][0]["val_macro_f1"] == pytest.approx(
            b["models"][name]["selection"][0]["val_macro_f1"]
        )
        assert a["models"][name]["best_C"] == b["models"][name]["best_C"]
        assert a["models"][name]["validation"] == b["models"][name]["validation"]
        assert a["models"][name]["test"]["macro_f1"] > b["models"][name]["test"]["macro_f1"]


def test_render_markdown_contains_tables_and_notes() -> None:
    train, val = synthetic(20, seed=1, source=True), synthetic(8, seed=2, source=True)
    r = run_baseline(train, val, synthetic(8, seed=3, source=True), settings(), seed=0)
    md = render_markdown({"toy": r}, notes="## Note\nhand-written")
    assert "| toy | logreg |" in md and "| toy | linear_svc |" in md
    assert "Confusion matrix" in md and "by data source" in md
    assert md.rstrip().endswith("hand-written")
