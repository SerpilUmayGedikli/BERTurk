"""Tests for tcls.data. Use tiny synthetic frames and a fake tokenizer (no downloads)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from tcls.data import (
    LABEL,
    TEXT,
    build_label_mapping,
    clean,
    count_overlap,
    length_stats,
    normalize_text,
    prepare_splits,
    remove_overlap,
    stratified_split,
    token_lengths,
    tokenize_texts,
)
from tcls.utils import DataConfig


def frame(rows: list[tuple[Any, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[TEXT, LABEL])


def balanced_frame(n_per_class: int, labels: tuple[str, ...] = ("a", "b", "c")) -> pd.DataFrame:
    rows = [(f"metin {lab} {i}", lab) for lab in labels for i in range(n_per_class)]
    return frame(rows)


class FakeTokenizer:
    """Whitespace tokenizer that mimics the HF call signature."""

    def __call__(
        self,
        texts: list[str],
        truncation: bool = False,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, list[list[int]]]:
        ids = []
        for t in texts:
            toks = [len(w) for w in t.split()]
            if add_special_tokens:
                toks = [101, *toks, 102]
            if truncation and max_length is not None:
                toks = toks[:max_length]
            ids.append(toks)
        return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}


def config(**kw: Any) -> DataConfig:
    base: dict[str, Any] = {"dataset_id": "fake", "text_column": "t", "label_column": "l"}
    return DataConfig(**(base | kw))


def test_normalize_text_collapses_whitespace_and_handles_non_strings() -> None:
    assert normalize_text("  Merhaba \n\t dünya  ") == "Merhaba dünya"
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""


def test_clean_drops_empty_and_duplicate_texts() -> None:
    df = frame([("Haber bir", "a"), ("  ", "a"), ("", "b"), ("haber  BIR", "a"), ("İki", "b")])
    out, stats = clean(df)
    assert out[TEXT].tolist() == ["Haber bir", "İki"]
    assert (stats.n_empty, stats.n_duplicates, stats.n_out) == (2, 1, 2)


def test_clean_drops_all_rows_of_label_conflicting_texts() -> None:
    df = frame([("aynı metin", "a"), ("Aynı metin", "b"), ("başka", "a")])
    out, stats = clean(df)
    assert out[TEXT].tolist() == ["başka"]
    assert stats.n_conflicting == 2


def test_label_mapping_is_sorted_and_invertible() -> None:
    label2id, id2label = build_label_mapping(["spor", "ekonomi", "spor", "dunya"])
    assert label2id == {"dunya": 0, "ekonomi": 1, "spor": 2}
    assert all(id2label[i] == name for name, i in label2id.items())


def test_stratified_split_is_deterministic_and_preserves_class_ratio() -> None:
    df = frame([(f"a{i}", "a") for i in range(80)] + [(f"b{i}", "b") for i in range(20)])
    rest1, held1 = stratified_split(df, 0.2, seed=7)
    _, held2 = stratified_split(df, 0.2, seed=7)
    pd.testing.assert_frame_equal(held1, held2)
    assert len(held1) == 20 and len(rest1) == 80
    assert held1[LABEL].value_counts().to_dict() == {"a": 16, "b": 4}
    assert count_overlap(rest1, held1) == 0
    _, held3 = stratified_split(df, 0.2, seed=8)
    assert set(held3[TEXT]) != set(held1[TEXT])


def test_remove_overlap_is_case_and_whitespace_insensitive() -> None:
    train = frame([("Ortak  Metin", "a"), ("sadece train", "b")])
    test = frame([("ortak metin", "a")])
    out, n = remove_overlap(train, test)
    assert n == 1
    assert out[TEXT].tolist() == ["sadece train"]


def test_prepare_splits_without_test_split_is_leak_free() -> None:
    raw = balanced_frame(40)
    # Inject duplicates and an empty row that must be cleaned.
    raw = pd.concat([raw, raw.head(5), frame([("", "a")])], ignore_index=True)
    prepared = prepare_splits({"train": raw}, config(test_size=0.2, val_size=0.25, seed=1))
    sizes = {k: len(v) for k, v in prepared.splits().items()}
    assert sizes == {"train": 72, "validation": 24, "test": 24}
    assert all(v == 0 for v in prepared.report["overlap_after"].values())
    assert prepared.report["clean"]["train"]["n_duplicates"] == 5
    assert prepared.report["clean"]["train"]["n_empty"] == 1
    assert prepared.label2id == {"a": 0, "b": 1, "c": 2}


def test_prepare_splits_keeps_official_test_and_removes_overlap_from_train() -> None:
    train = balanced_frame(30)
    test = pd.concat([frame([("metin a 0", "a"), ("Metin  B 1", "b")]), frame([("yeni", "c")])])
    prepared = prepare_splits({"train": train, "test": test}, config(val_size=0.1, seed=3))
    assert len(prepared.test) == 3  # test is never shrunk by overlap removal
    assert prepared.report["removed_from_train_for_test_overlap"] == 2
    assert count_overlap(prepared.train, prepared.test) == 0
    assert count_overlap(prepared.validation, prepared.test) == 0


def test_prepare_splits_rejects_missing_train() -> None:
    with pytest.raises(ValueError, match="no 'train' split"):
        prepare_splits({"test": balanced_frame(5)}, config())


def test_tokenize_texts_truncates_to_max_length() -> None:
    enc = tokenize_texts(["bir iki üç dört beş", "kısa"], FakeTokenizer(), max_length=4)
    assert [len(x) for x in enc["input_ids"]] == [4, 3]
    assert set(enc) == {"input_ids", "attention_mask"}


def test_token_lengths_and_stats() -> None:
    texts = ["bir"] * 9 + [" ".join(["kelime"] * 300)]
    lengths = token_lengths(texts, FakeTokenizer(), batch_size=3)
    assert lengths.tolist() == [3] * 9 + [302]
    stats = length_stats(lengths)
    assert stats["max"] == 302
    assert stats["frac_gt_128"] == pytest.approx(0.1)
    assert stats["frac_gt_256"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        length_stats(np.array([], dtype=np.int64))
