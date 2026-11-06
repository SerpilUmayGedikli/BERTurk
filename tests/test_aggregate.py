from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tcls.aggregate import (
    NOT_RUN,
    Planned,
    aggregate,
    load_runs,
    mean_std,
    planned_runs,
    render_markdown,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def run(task: str, tag: str, seed: int, test_f1: float, smoke: bool = False) -> dict[str, Any]:
    pc = {lab: {"precision": 0.9, "recall": 0.8, "f1": test_f1} for lab in ("a", "b")}
    metrics = {"macro_f1": test_f1, "accuracy": test_f1 + 0.01, "per_class": pc}
    return {
        "task": task,
        "tag": tag,
        "seed": seed,
        "smoke": smoke,
        "labels": ["a", "b"],
        "validation": {"macro_f1": 0.5, "accuracy": 0.6, "per_class": pc},
        "test": metrics,
        "test_focus_label": {"label": "a", "precision": 0.9, "recall": test_f1, "f1": 0.8},
        "hyperparameters": {"max_train_samples": 100},
        "data": {"used_sizes": {"train": 100}},
        "device": {"gpu_name": "Tesla T4"},
    }


def test_mean_std_matches_sample_std() -> None:
    m, s = mean_std([0.90, 0.92, 0.94])
    assert m == pytest.approx(0.92)
    assert s == pytest.approx(0.02)  # ddof=1
    assert mean_std([0.5]) == (0.5, None)
    with pytest.raises(ValueError):
        mean_std([])


def test_aggregate_groups_seeds_and_marks_missing() -> None:
    runs = [run("news", "len256", s, f) for s, f in [(42, 0.90), (43, 0.92), (44, 0.94)]]
    runs += [run("news", "len512", 42, 0.95)]
    planned = [Planned("news", t, "x.yaml") for t in ("len256", "len512", "len512_headtail")]
    agg = aggregate(runs, planned)
    full = agg["news"]["len256"]
    assert full["status"] == "complete" and full["seeds"] == [42, 43, 44]
    assert full["test_macro_f1"]["mean"] == pytest.approx(0.92)
    assert full["test_macro_f1"]["std"] == pytest.approx(0.02)
    assert full["focus_label"]["recall"]["mean"] == pytest.approx(0.92)
    assert agg["news"]["len512"]["status"] == "partial: missing seeds [43, 44]"
    assert agg["news"]["len512"]["test_macro_f1"]["std"] is None
    assert agg["news"]["len512_headtail"] == {"status": NOT_RUN}


def test_aggregate_rejects_duplicate_seeds() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate([run("news", "t", 42, 0.9), run("news", "t", 42, 0.8)], [])


def test_load_runs_skips_smoke(tmp_path: Path) -> None:
    for r in (run("news", "t", 42, 0.9), run("news", "t", 43, 0.1, smoke=True)):
        name = f"berturk_news_t_seed{r['seed']}{'_smoke' if r['smoke'] else ''}.json"
        (tmp_path / name).write_text(json.dumps(r), encoding="utf-8")
    assert [r["seed"] for r in load_runs(tmp_path)] == [42]


def test_planned_runs_from_shipped_configs() -> None:
    planned = planned_runs(sorted(CONFIG_DIR.glob("*.yaml")))
    assert {(p.task, p.tag) for p in planned} == {
        ("news", "len128"),
        ("news", "len256"),
        ("news", "len512"),
        ("news", "len512_headtail"),
        ("sentiment", "len128_n50k"),
        ("sentiment", "len128_n50k_weighted"),
    }


def test_render_markdown_shows_baseline_and_not_run() -> None:
    agg = aggregate([run("news", "len256", 42, 0.9)], [Planned("news", "len512", "x")])
    baseline = {
        "sizes": {"train": 3278},
        "models": {
            "logreg": {
                "validation": {"macro_f1": 0.91},
                "test": {"macro_f1": 0.905, "accuracy": 0.9, "per_class": {}},
            }
        },
    }
    md = render_markdown(agg, {"news": baseline})
    assert "| TF-IDF + logreg | 1 (deterministic) | 3278 | 91.00 | 90.50 | 90.00 |" in md
    assert f"| BERTurk `len512` | {NOT_RUN} |" in md
    assert "partial: missing seeds [43, 44]" in md
