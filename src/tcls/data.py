"""Data loading, cleaning, leakage checks, stratified splitting and tokenization.

Pipeline (see ``prepare_splits``):
1. Load raw splits from the Hugging Face Hub and normalise to ``text``/``label`` columns.
2. Clean every split: normalise whitespace, drop empty texts, drop texts that appear with
   more than one label (label conflicts), drop exact duplicates (case-insensitive key).
3. If the dataset has no test split, carve one out (stratified, fixed seed).
4. Remove from the train pool every text that also occurs in test (test stays untouched).
5. Split validation from the train pool (stratified, fixed seed).
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tcls.utils import Config, DataConfig, get_logger, load_config, set_seed

logger = get_logger(__name__)

TEXT = "text"
LABEL = "label"
SOURCE = "source"
_WS = re.compile(r"\s+")


# --------------------------------------------------------------------------- loading


def load_raw(cfg: DataConfig) -> dict[str, pd.DataFrame]:
    """Download (or read from cache) all splits and return ``text``/``label`` frames.

    ``ClassLabel`` integer labels are converted to their string names so every
    dataset is handled the same way downstream.
    """
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset_id, cfg.dataset_config, cache_dir=cfg.cache_dir)
    out: dict[str, pd.DataFrame] = {}
    for split_name, split in ds.items():
        feature = split.features[cfg.label_column]
        names: list[str] | None = getattr(feature, "names", None)
        df = split.to_pandas()
        assert isinstance(df, pd.DataFrame)
        columns = {cfg.text_column: TEXT, cfg.label_column: LABEL}
        if cfg.source_column is not None:
            # Kept for analysis only (e.g. shortcut checks); never used as a feature.
            columns[cfg.source_column] = SOURCE
        df = df[list(columns)].rename(columns=columns)
        if names is not None:
            df[LABEL] = [names[i] for i in df[LABEL]]
        df[LABEL] = df[LABEL].astype(str)
        out[str(split_name)] = df.reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- cleaning


def normalize_text(text: object) -> str:
    """Collapse whitespace and strip. Non-strings (None/NaN) become ``""``."""
    if not isinstance(text, str):
        return ""
    return _WS.sub(" ", text).strip()


def dedup_key(text: str) -> str:
    """Key used for duplicate / leakage detection (whitespace- and case-insensitive)."""
    return normalize_text(text).casefold()


@dataclass
class CleanStats:
    n_in: int = 0
    n_empty: int = 0
    n_conflicting: int = 0
    n_duplicates: int = 0
    n_out: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleanStats]:
    """Normalise text, drop empty rows, label-conflicting texts and duplicates."""
    stats = CleanStats(n_in=len(df))
    df = df.copy()
    df[TEXT] = df[TEXT].map(normalize_text)

    empty = df[TEXT] == ""
    stats.n_empty = int(empty.sum())
    df = df[~empty]

    keys = df[TEXT].map(dedup_key)
    labels_per_key = df.groupby(keys)[LABEL].nunique()
    conflicting_keys = set(labels_per_key[labels_per_key > 1].index)
    conflict_mask = keys.isin(conflicting_keys)
    stats.n_conflicting = int(conflict_mask.sum())
    df, keys = df[~conflict_mask], keys[~conflict_mask]

    dup_mask = keys.duplicated(keep="first")
    stats.n_duplicates = int(dup_mask.sum())
    df = df[~dup_mask].reset_index(drop=True)

    stats.n_out = len(df)
    return df, stats


def remove_overlap(df: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows of ``df`` whose text (by ``dedup_key``) also appears in ``reference``."""
    ref_keys = set(reference[TEXT].map(dedup_key))
    mask = df[TEXT].map(dedup_key).isin(ref_keys)
    return df[~mask].reset_index(drop=True), int(mask.sum())


def count_overlap(a: pd.DataFrame, b: pd.DataFrame) -> int:
    return len(set(a[TEXT].map(dedup_key)) & set(b[TEXT].map(dedup_key)))


# --------------------------------------------------------------------------- labels / split


def build_label_mapping(labels: Sequence[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Deterministic mapping: sorted unique label names -> 0..K-1."""
    names = sorted(set(labels))
    label2id = {name: i for i, name in enumerate(names)}
    id2label = {i: name for name, i in label2id.items()}
    return label2id, id2label


def stratified_split(
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified (by label), seeded split. Returns ``(rest, held_out)``."""
    rest, held = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[LABEL])
    assert isinstance(rest, pd.DataFrame) and isinstance(held, pd.DataFrame)
    return rest.reset_index(drop=True), held.reset_index(drop=True)


def stratified_subsample(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Seeded, label-stratified subset of ``n`` rows (the full frame if ``n`` is None/too big)."""
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    subset, _ = train_test_split(df, train_size=n, random_state=seed, stratify=df[LABEL])
    assert isinstance(subset, pd.DataFrame)
    return subset.reset_index(drop=True)


@dataclass
class PreparedData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    label2id: dict[str, int]
    id2label: dict[int, str]
    report: dict[str, Any] = field(default_factory=dict)

    def splits(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


def prepare_splits(raw: Mapping[str, pd.DataFrame], cfg: DataConfig) -> PreparedData:
    """Clean raw splits and build leakage-free, stratified train/validation/test."""
    report: dict[str, Any] = {"raw_splits": {k: len(v) for k, v in raw.items()}, "clean": {}}

    cleaned: dict[str, pd.DataFrame] = {}
    for name, df in raw.items():
        cleaned[name], stats = clean(df)
        report["clean"][name] = stats.as_dict()

    if "train" not in cleaned:
        raise ValueError(f"Dataset has no 'train' split: {list(cleaned)}")

    if "test" in cleaned:
        pool, test = cleaned["train"], cleaned["test"]
        report["test_source"] = "official test split"
    else:
        pool, test = stratified_split(cleaned["train"], cfg.test_size, cfg.seed)
        report["test_source"] = f"stratified {cfg.test_size:.0%} of cleaned train (seed={cfg.seed})"

    report["train_test_overlap_before"] = count_overlap(pool, test)
    pool, n_removed = remove_overlap(pool, test)
    report["removed_from_train_for_test_overlap"] = n_removed

    train, validation = stratified_split(pool, cfg.val_size, cfg.seed)
    report["validation_source"] = f"stratified {cfg.val_size:.0%} of train pool (seed={cfg.seed})"

    report["overlap_after"] = {
        "train-validation": count_overlap(train, validation),
        "train-test": count_overlap(train, test),
        "validation-test": count_overlap(validation, test),
    }

    label2id, id2label = build_label_mapping(train[LABEL].tolist())
    unseen = (set(validation[LABEL]) | set(test[LABEL])) - set(label2id)
    if unseen:
        raise ValueError(f"Labels missing from train: {sorted(unseen)}")

    return PreparedData(train, validation, test, label2id, id2label, report)


# --------------------------------------------------------------------------- tokenization

Tokenizer = Callable[..., Any]


def tokenize_texts(
    texts: Sequence[str], tokenizer: Tokenizer, max_length: int
) -> dict[str, list[list[int]]]:
    """Tokenize with truncation (padding is left to the data collator)."""
    enc = tokenizer(list(texts), truncation=True, max_length=max_length)
    return {k: list(v) for k, v in dict(enc).items()}


def token_lengths(texts: Sequence[str], tokenizer: Tokenizer, batch_size: int = 1000) -> np.ndarray:
    """Untruncated token counts (incl. special tokens) for each text."""
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        enc = tokenizer(batch, truncation=False, add_special_tokens=True)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return np.asarray(lengths, dtype=np.int64)


def length_stats(lengths: np.ndarray, thresholds: Sequence[int] = (128, 256)) -> dict[str, float]:
    if lengths.size == 0:
        raise ValueError("No lengths given")
    stats: dict[str, float] = {
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p95": float(np.percentile(lengths, 95)),
        "max": float(lengths.max()),
    }
    for t in thresholds:
        stats[f"frac_gt_{t}"] = float((lengths > t).mean())
    return stats


# --------------------------------------------------------------------------- summary


def class_distribution(df: pd.DataFrame) -> dict[str, int]:
    counts = df[LABEL].value_counts().sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def summarize_task(
    cfg: Config, prepared: PreparedData, tokenizer: Tokenizer | None
) -> dict[str, Any]:
    """Collect sizes, class distributions and token-length stats per split."""
    sample = cfg.summary.get("token_stats_sample")
    out: dict[str, Any] = {"splits": {}}
    for name, df in prepared.splits().items():
        entry: dict[str, Any] = {"n": len(df), "classes": class_distribution(df)}
        entry["chars_mean"] = float(df[TEXT].str.len().mean())
        if tokenizer is not None:
            texts = df[TEXT]
            if sample is not None and len(df) > int(sample):
                texts = texts.sample(n=int(sample), random_state=cfg.data.seed)
                entry["token_stats_on"] = f"random sample of {int(sample)} (seed={cfg.data.seed})"
            else:
                entry["token_stats_on"] = "all rows"
            entry["tokens"] = length_stats(token_lengths(texts.tolist(), tokenizer))
        out["splits"][name] = entry
    return out


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_task_markdown(cfg: Config, prepared: PreparedData, summary: Mapping[str, Any]) -> str:
    r = prepared.report
    lines = [
        f"## {cfg.task}: `{cfg.data.dataset_id}`",
        "",
        f"- Raw splits: {r['raw_splits']}",
        f"- Test split: {r['test_source']}",
        f"- Validation split: {r['validation_source']}",
        f"- Labels ({len(prepared.label2id)}): {prepared.label2id}",
        "",
        "### Cleaning",
        "",
        "| split | in | empty | label-conflicting | duplicates | out |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in r["clean"].items():
        lines.append(
            f"| {name} | {s['n_in']} | {s['n_empty']} | {s['n_conflicting']} "
            f"| {s['n_duplicates']} | {s['n_out']} |"
        )
    lines += [
        "",
        f"- Train/test overlapping texts before removal: {r['train_test_overlap_before']} "
        f"-> removed {r['removed_from_train_for_test_overlap']} rows from train",
        f"- Overlap after cleaning: {r['overlap_after']}",
        "",
        "### Class distribution",
        "",
    ]
    labels = list(prepared.label2id)
    split_names = list(summary["splits"])
    lines.append("| label | " + " | ".join(split_names) + " |")
    lines.append("|---|" + "---:|" * len(split_names))
    for lab in labels:
        cells = []
        for sn in split_names:
            s = summary["splits"][sn]
            c = s["classes"].get(lab, 0)
            cells.append(f"{c} ({_fmt_pct(c / s['n'])})")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    totals = [str(summary["splits"][sn]["n"]) for sn in split_names]
    lines.append("| **total** | " + " | ".join(totals) + " |")
    train_counts = summary["splits"]["train"]["classes"]
    ratio = max(train_counts.values()) / min(train_counts.values())
    lines += ["", f"- Train imbalance (largest / smallest class): {ratio:.2f}"]

    if any("tokens" in s for s in summary["splits"].values()):
        lines += [
            "",
            f"### Token lengths (`{cfg.model.name}` tokenizer, incl. [CLS]/[SEP])",
            "",
            "| split | computed on | mean | median | p95 | max | >128 | >256 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for sn, s in summary["splits"].items():
            t = s["tokens"]
            lines.append(
                f"| {sn} | {s['token_stats_on']} | {t['mean']:.1f} | {t['median']:.0f} "
                f"| {t['p95']:.0f} | {t['max']:.0f} | {_fmt_pct(t['frac_gt_128'])} "
                f"| {_fmt_pct(t['frac_gt_256'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def save_processed(prepared: PreparedData, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in prepared.splits().items():
        df.to_parquet(out / f"{name}.parquet", index=False)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare splits and write a data summary.")
    parser.add_argument("--config", nargs="+", required=True, help="One or more YAML configs")
    parser.add_argument("--summary", default="reports/data_summary.md")
    parser.add_argument("--no-tokens", action="store_true", help="Skip token-length stats")
    args = parser.parse_args(argv)

    sections = ["# Data summary", "", "Generated by `python -m tcls.data`.", ""]
    for path in args.config:
        cfg = load_config(path)
        set_seed(cfg.data.seed)
        logger.info("Preparing %s (%s)", cfg.task, cfg.data.dataset_id)
        prepared = prepare_splits(load_raw(cfg.data), cfg.data)
        save_processed(prepared, cfg.data.processed_dir)
        tokenizer = None
        if not args.no_tokens:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, cache_dir=cfg.data.cache_dir)
        summary = summarize_task(cfg, prepared, tokenizer)
        sections.append(render_task_markdown(cfg, prepared, summary))

    out = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sections), encoding="utf-8")
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
