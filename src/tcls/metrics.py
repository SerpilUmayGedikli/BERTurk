"""Classification metrics shared by the baseline and BERTurk runs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

__all__ = ["compute_metrics", "focus_label_metrics", "source_breakdown"]


def compute_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    report = classification_report(
        y_true, y_pred, labels=list(labels), output_dict=True, zero_division=0
    )
    per_class = {
        lab: {
            "precision": float(report[lab]["precision"]),
            "recall": float(report[lab]["recall"]),
            "f1": float(report[lab]["f1-score"]),
            "support": int(report[lab]["support"]),
        }
        for lab in labels
    }
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="macro")),
        "per_class": per_class,
        "confusion_matrix": {"labels": list(labels), "rows_true_cols_pred": cm.tolist()},
    }


def source_breakdown(
    sources: Sequence[str], y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Per data source: gold label counts, predicted label counts and accuracy."""
    df = pd.DataFrame({"source": sources, "true": y_true, "pred": y_pred})
    out: dict[str, dict[str, Any]] = {}
    for src, g in df.groupby("source", sort=True):
        out[str(src)] = {
            "n": len(g),
            "true": {lab: int((g["true"] == lab).sum()) for lab in labels},
            "pred": {lab: int((g["pred"] == lab).sum()) for lab in labels},
            "accuracy": float((g["true"] == g["pred"]).mean()),
        }
    return out


def focus_label_metrics(metrics: dict[str, Any], label: str) -> dict[str, float]:
    """Precision/recall/F1 of one label (e.g. sentiment ``Notr``), pulled out for reporting."""
    pc = metrics["per_class"][label]
    return {"precision": pc["precision"], "recall": pc["recall"], "f1": pc["f1"]}
