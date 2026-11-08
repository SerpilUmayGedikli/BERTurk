"""The ``model://card`` resource.

Built from the checkpoint's config.json (labels) and, if MODEL_REPORT is set, from the run
JSON written by ``python -m tcls.train`` (task, data sizes, test metrics). Nothing is
invented: without a report, metrics are reported as unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from tcls_mcp import __version__

# Static facts about the two datasets this project trains on (see the repo README).
DATASETS: dict[str, dict[str, str]] = {
    "winvoker/turkish-sentiment-analysis-dataset": {
        "description": "Turkish sentiment (Positive / Negative / Notr) from product reviews, "
        "store reviews, tweets and Wikipedia sentences.",
        "license": "CC BY-SA 4.0",
        "caveat": "All Notr training examples come from Wikipedia/random sources, none from "
        "reviews, so 'neutral' partly means 'encyclopedic style'. Neutral customer reviews "
        "may be misclassified.",
    },
    "savasy/ttc4900": {
        "description": "TTC4900 Turkish news, 7 categories (dunya, ekonomi, kultur, saglik, "
        "siyaset, spor, teknoloji). Lowercased and preprocessed.",
        "license": "unknown (Hub card); source: Kemik group",
        "caveat": "Texts are lowercased; cased real-world news is out of distribution.",
    },
}


class TestMetrics(BaseModel):
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    test_size: int | None = None


class ModelCard(BaseModel):
    name: str
    version: str = Field(description="Server version + model identifier.")
    status: Literal["trained", "smoke", "unknown"] = Field(
        description="'smoke' = pipeline check on a tiny subset; its metrics are NOT results."
    )
    task: str | None
    base_model: str | None
    labels: list[str]
    max_tokens: int
    device: str
    training_data: dict[str, Any]
    test_metrics: TestMetrics | None
    warnings: list[str]


def _load_report(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def build_card(
    model_path: Path,
    labels: list[str],
    max_tokens: int,
    device: str,
    report_path: Path | None = None,
) -> ModelCard:
    report = _load_report(report_path)
    warnings: list[str] = []
    name = model_path.parent.name if model_path.name.startswith("checkpoint-") else model_path.name
    version = f"tcls-mcp {__version__} / {name}/{model_path.name}"

    if report is None:
        warnings.append("MODEL_REPORT is not set: task, training data and test metrics unknown.")
        return ModelCard(
            name=name,
            version=version,
            status="unknown",
            task=None,
            base_model=None,
            labels=labels,
            max_tokens=max_tokens,
            device=device,
            training_data={},
            test_metrics=None,
            warnings=warnings,
        )

    smoke = bool(report.get("smoke", False))
    if smoke:
        warnings.append(
            "SMOKE checkpoint: trained for 1 epoch on 200 examples to test the pipeline. "
            "Predictions and metrics are NOT meaningful."
        )
    if report.get("labels") and list(report["labels"]) != labels:
        warnings.append(f"Report labels {report['labels']} differ from model labels {labels}.")
    trained_len = report.get("hyperparameters", {}).get("max_length")
    if trained_len and trained_len < max_tokens:
        warnings.append(
            f"Model was fine-tuned with max_length={trained_len}; inputs longer than that "
            f"(up to {max_tokens}) are longer than anything seen in training."
        )

    if report.get("hyperparameters", {}).get("truncation") == "head_tail":
        warnings.append(
            "Model was trained with head+tail truncation; this server keeps only the head of "
            "long texts, so long-text predictions may differ from the reported metrics."
        )

    dataset_id = report.get("dataset_id")
    data = report.get("data", {})
    training_data: dict[str, Any] = {
        "dataset_id": dataset_id,
        **DATASETS.get(str(dataset_id), {}),
        "split_sizes_full": data.get("full_sizes"),
        "split_sizes_used": data.get("used_sizes"),
        "train_subsample": data.get("subsample") or data.get("max_train_samples"),
    }

    test = report.get("test")
    metrics = None
    if test:
        per_class = test.get("per_class", {})
        metrics = TestMetrics(
            accuracy=round(float(test["accuracy"]), 4),
            macro_f1=round(float(test["macro_f1"]), 4),
            per_class_f1={k: round(float(v["f1"]), 4) for k, v in per_class.items()},
            test_size=(data.get("used_sizes") or {}).get("test"),
        )

    return ModelCard(
        name=str(report.get("run", name)),
        version=version,
        status="smoke" if smoke else "trained",
        task=report.get("task"),
        base_model=report.get("model"),
        labels=labels,
        max_tokens=max_tokens,
        device=device,
        training_data=training_data,
        test_metrics=metrics,
        warnings=warnings,
    )
