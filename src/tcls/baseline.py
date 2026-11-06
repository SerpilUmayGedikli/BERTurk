"""TF-IDF (word 1-2 grams) + LogisticRegression / LinearSVC baselines.

Protocol:
1. Fit TF-IDF on the train split only.
2. For each model, fit one classifier per C in the config grid on train and score it on
   validation (macro-F1). The best C is chosen on validation only.
3. Evaluate the selected classifier on the test split once, at the end.

Both solvers are deterministic for fixed data. lbfgs LogisticRegression has no randomness.
LinearSVC (dual coordinate descent) shuffles with ``random_state``, so we fix it to the
data seed and, separately, measure its validation-only sensitivity to that seed.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.svm import LinearSVC

from tcls.data import LABEL, SOURCE, TEXT
from tcls.utils import Config, get_logger, load_config, set_seed

logger = get_logger(__name__)

# LogisticRegression | LinearSVC; sklearn ships no type stubs, so this is Any to mypy.
Classifier: TypeAlias = Any
MODEL_NAMES = ("logreg", "linear_svc")


def turkish_lower(text: str) -> str:
    """Lowercase with Turkish dotted/dotless I rules (str.lower maps 'I' to 'i')."""
    return text.replace("I", "ı").replace("İ", "i").lower()


@dataclass(frozen=True)
class BaselineSettings:
    tfidf: dict[str, Any]
    models: dict[str, dict[str, Any]]
    svc_seed_check: list[int]
    top_features: int
    output: str


def parse_settings(raw: Mapping[str, Any]) -> BaselineSettings:
    if "baseline" not in raw:
        raise ValueError("Config has no 'baseline' section")
    b = raw["baseline"]
    models = dict(b["models"])
    unknown = set(models) - set(MODEL_NAMES)
    if unknown:
        raise ValueError(f"Unknown baseline models: {sorted(unknown)}")
    for name, params in models.items():
        if not params.get("C"):
            raise ValueError(f"Model '{name}' needs a non-empty C grid")
    return BaselineSettings(
        tfidf=dict(b["tfidf"]),
        models=models,
        svc_seed_check=list(b.get("svc_seed_check") or []),
        top_features=int(b.get("top_features", 20)),
        output=str(b["output"]),
    )


def make_vectorizer(tfidf: Mapping[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=turkish_lower,
        analyzer="word",
        ngram_range=tuple(tfidf.get("ngram_range", (1, 2))),
        min_df=tfidf.get("min_df", 1),
        max_df=tfidf.get("max_df", 1.0),
        sublinear_tf=bool(tfidf.get("sublinear_tf", True)),
        max_features=tfidf.get("max_features"),
        dtype=np.float32,
    )


def make_classifier(name: str, c: float, params: Mapping[str, Any], seed: int) -> Classifier:
    max_iter = int(params.get("max_iter", 1000))
    if name == "logreg":
        return LogisticRegression(C=c, solver="lbfgs", max_iter=max_iter)
    if name == "linear_svc":
        return LinearSVC(C=c, max_iter=max_iter, random_state=seed)
    raise ValueError(f"Unknown model '{name}'")


def fit_classifier(clf: Classifier, x: Any, y: Sequence[str]) -> bool:
    """Fit and return whether the solver converged (no ConvergenceWarning)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x, y)
    return not any(issubclass(w.category, ConvergenceWarning) for w in caught)


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


def top_features(
    vectorizer: TfidfVectorizer, clf: Classifier, k: int
) -> dict[str, list[tuple[str, float]]]:
    """Highest-weight n-grams per class (one-vs-rest weights for multiclass)."""
    vocab = vectorizer.get_feature_names_out()
    coef = np.asarray(clf.coef_)
    classes = [str(c) for c in clf.classes_]
    if coef.shape[0] == 1:  # binary: one row, positive class = classes[1]
        coef = np.vstack([-coef[0], coef[0]])
    out: dict[str, list[tuple[str, float]]] = {}
    for i, cls in enumerate(classes):
        idx = np.argsort(coef[i])[::-1][:k]
        out[cls] = [(str(vocab[j]), round(float(coef[i, j]), 4)) for j in idx]
    return out


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


def select_on_validation(
    name: str,
    params: Mapping[str, Any],
    x_train: Any,
    y_train: Sequence[str],
    x_val: Any,
    y_val: Sequence[str],
    seed: int,
) -> tuple[list[dict[str, Any]], Classifier]:
    """Fit one classifier per C; keep the best by validation macro-F1 (ties: smaller C)."""
    trials: list[dict[str, Any]] = []
    best: tuple[float, Classifier] | None = None
    for c in sorted(float(v) for v in params["C"]):
        clf = make_classifier(name, c, params, seed)
        t0 = time.perf_counter()
        converged = fit_classifier(clf, x_train, y_train)
        pred = clf.predict(x_val)
        score = float(f1_score(y_val, pred, average="macro"))
        trials.append(
            {
                "C": c,
                "val_macro_f1": score,
                "val_accuracy": float(accuracy_score(y_val, pred)),
                "converged": converged,
                "fit_seconds": round(time.perf_counter() - t0, 2),
            }
        )
        logger.info("%s C=%g val macro-F1=%.4f converged=%s", name, c, score, converged)
        if best is None or score > best[0]:
            best = (score, clf)
    assert best is not None
    return trials, best[1]


def run_baseline(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    settings: BaselineSettings,
    seed: int,
) -> dict[str, Any]:
    """Validation-based model selection, then a single test evaluation per model."""
    labels = sorted(set(train[LABEL]))
    t0 = time.perf_counter()
    vectorizer = make_vectorizer(settings.tfidf)
    x_train = vectorizer.fit_transform(train[TEXT].tolist())
    x_val = vectorizer.transform(validation[TEXT].tolist())
    y_train, y_val = train[LABEL].tolist(), validation[LABEL].tolist()
    result: dict[str, Any] = {
        "labels": labels,
        "sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "tfidf": {**settings.tfidf, "vocab_size": len(vectorizer.vocabulary_)},
        "vectorize_seconds": round(time.perf_counter() - t0, 2),
        "models": {},
    }

    selected: dict[str, Classifier] = {}
    for name, params in settings.models.items():
        trials, clf = select_on_validation(name, params, x_train, y_train, x_val, y_val, seed)
        selected[name] = clf
        best_c = float(clf.C)
        entry: dict[str, Any] = {
            "selection": trials,
            "best_C": best_c,
            "validation": compute_metrics(y_val, clf.predict(x_val).tolist(), labels),
        }
        if name == "linear_svc" and settings.svc_seed_check:
            scores = []
            for s in settings.svc_seed_check:
                alt = make_classifier(name, best_c, params, s)
                fit_classifier(alt, x_train, y_train)
                scores.append(float(f1_score(y_val, alt.predict(x_val), average="macro")))
            entry["val_seed_check"] = {
                "random_states": settings.svc_seed_check,
                "val_macro_f1": scores,
                "max_minus_min": max(scores) - min(scores),
            }
        result["models"][name] = entry

    # ---- test: touched once, after all selection is finished ----
    x_test = vectorizer.transform(test[TEXT].tolist())
    y_test = test[LABEL].tolist()
    for name, clf in selected.items():
        pred = clf.predict(x_test).tolist()
        entry = result["models"][name]
        entry["test"] = compute_metrics(y_test, pred, labels)
        entry["top_features"] = top_features(vectorizer, clf, settings.top_features)
        if SOURCE in test.columns:
            entry["test_by_source"] = source_breakdown(
                test[SOURCE].astype(str).tolist(), y_test, pred, labels
            )
    return result


# --------------------------------------------------------------------------- reporting


def _pct(x: float) -> str:
    return f"{100 * x:.2f}"


def render_markdown(results: Mapping[str, Mapping[str, Any]], notes: str | None = None) -> str:
    lines = [
        "# Baseline results (TF-IDF word 1-2 grams)",
        "",
        "Generated by `python -m tcls.baseline`. C was chosen on validation macro-F1; each",
        "selected model was evaluated on test once. Both models are deterministic for a fixed",
        "split (see `val_seed_check` in the JSON for LinearSVC), so each number is one run,",
        "with no ± std. Variance from different data splits was not measured.",
        "",
        "| task | model | best C | val macro-F1 | test macro-F1 | test accuracy |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task, r in results.items():
        for name, m in r["models"].items():
            lines.append(
                f"| {task} | {name} | {m['best_C']:g} | {_pct(m['validation']['macro_f1'])} "
                f"| {_pct(m['test']['macro_f1'])} | {_pct(m['test']['accuracy'])} |"
            )
    for task, r in results.items():
        lines += ["", f"## {task}", ""]
        labels = r["labels"]
        for name, m in r["models"].items():
            t = m["test"]
            lines += [
                f"### {name} (test, per class)",
                "",
                "| label | precision | recall | F1 | support |",
                "|---|---:|---:|---:|---:|",
            ]
            for lab in labels:
                p = t["per_class"][lab]
                lines.append(
                    f"| {lab} | {_pct(p['precision'])} | {_pct(p['recall'])} "
                    f"| {_pct(p['f1'])} | {p['support']} |"
                )
            lines += ["", "Confusion matrix (rows = true, columns = predicted):", ""]
            lines.append("| | " + " | ".join(labels) + " |")
            lines.append("|---|" + "---:|" * len(labels))
            for lab, row in zip(labels, t["confusion_matrix"]["rows_true_cols_pred"], strict=True):
                lines.append(f"| **{lab}** | " + " | ".join(str(v) for v in row) + " |")
            lines.append("")
            if "test_by_source" in m:
                lines += [
                    "Test predictions by data source (analysis only, the source is not a feature):",
                    "",
                    "| source | n | gold "
                    + " / ".join(labels)
                    + " | predicted "
                    + " / ".join(labels)
                    + " | accuracy |",
                    "|---|---:|---|---|---:|",
                ]
                for src, s in m["test_by_source"].items():
                    gold = " / ".join(str(s["true"][lab]) for lab in labels)
                    pred = " / ".join(str(s["pred"][lab]) for lab in labels)
                    lines.append(f"| {src} | {s['n']} | {gold} | {pred} | {_pct(s['accuracy'])} |")
                lines.append("")
        if "logreg" in r["models"]:
            lines += ["### logreg: top-weighted n-grams per class (first 12)", ""]
            for lab, feats in r["models"]["logreg"]["top_features"].items():
                shown = ", ".join(f"`{f}`" for f, _ in feats[:12])
                lines.append(f"- **{lab}**: {shown}")
            lines.append("")
    if notes:
        lines += ["", notes.strip(), ""]
    lines.append("")
    return "\n".join(lines)


def run_task(cfg: Config) -> dict[str, Any]:
    settings = parse_settings(cfg.raw)
    set_seed(cfg.data.seed)
    d = Path(cfg.data.processed_dir)
    missing = [s for s in ("train", "validation", "test") if not (d / f"{s}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {d}; run `python -m tcls.data --config ...` first"
        )
    train, validation, test = (
        pd.read_parquet(d / f"{s}.parquet") for s in ("train", "validation", "test")
    )
    result = run_baseline(train, validation, test, settings, cfg.data.seed)
    result = {
        "task": cfg.task,
        "dataset_id": cfg.data.dataset_id,
        "seed": cfg.data.seed,
        "sklearn_version": sklearn.__version__,
        **result,
    }
    out = Path(settings.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TF-IDF baselines (validation-selected).")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--report", default="reports/baseline_results.md")
    parser.add_argument(
        "--render-only", action="store_true", help="Rebuild the markdown from existing JSON"
    )
    parser.add_argument(
        "--notes", default=None, help="Markdown file appended to the report (hand-written notes)"
    )
    args = parser.parse_args(argv)

    notes = Path(args.notes).read_text(encoding="utf-8") if args.notes else None
    results: dict[str, dict[str, Any]] = {}
    for path in args.config:
        cfg = load_config(path)
        if args.render_only:
            out = Path(parse_settings(cfg.raw).output)
            results[cfg.task] = json.loads(out.read_text(encoding="utf-8"))
        else:
            results[cfg.task] = run_task(cfg)
    report = Path(args.report)
    report.write_text(render_markdown(results, notes), encoding="utf-8")
    logger.info("Wrote %s", report)


if __name__ == "__main__":
    main()
