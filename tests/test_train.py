"""Tests for tcls.train with fake tokenizer/model/Trainer (no torch or network needed)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from tcls.data import LABEL, SOURCE, TEXT, stratified_subsample
from tcls.train import (
    Deps,
    TrainConfig,
    best_validation_epoch,
    class_weights,
    encode_texts,
    parse_train_config,
    prepare_run_data,
    resolve_output_dir,
    special_token_layout,
    train_and_evaluate,
)
from tcls.utils import Config, DataConfig, ModelConfig, load_config, load_raw_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


# --------------------------------------------------------------------------- fakes


class FakeTokenizer:
    def __call__(
        self,
        texts: list[str],
        truncation: bool = False,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, list[list[int]]]:
        ids = []
        for t in texts:
            toks = [int(w) if w.isdigit() else len(w) for w in t.split()]
            if add_special_tokens:
                toks = [101, *toks, 102]
            if truncation and max_length is not None:
                toks = [*toks[: max_length - 1], 102] if len(toks) > max_length else toks
            ids.append(toks)
        return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}

    # Deliberately no build_inputs_with_special_tokens: transformers 5 tokenizers lack it.


class FakeTrainer:
    """Records what it is given; 'predicts' the gold labels of whatever dataset it sees."""

    instances: ClassVar[list[FakeTrainer]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[str] = []
        self.state = SimpleNamespace(log_history=[], best_model_checkpoint="ckpt-2")
        FakeTrainer.instances.append(self)

    def _name(self, ds: Any) -> str:
        return str(ds["name"])

    def train(self) -> None:
        self.calls.append("train")
        self.state.log_history = [
            {"loss": 1.0, "epoch": 1.0, "step": 10},
            {"eval_macro_f1": 0.5, "eval_loss": 0.9, "epoch": 1.0, "step": 10},
            {"eval_macro_f1": 0.8, "eval_loss": 0.5, "epoch": 2.0, "step": 20},
        ]

    def predict(self, ds: Any) -> Any:
        self.calls.append(f"predict:{self._name(ds)}")
        k = 3
        logits = np.eye(k)[np.asarray(ds["labels"])]
        return SimpleNamespace(predictions=logits, label_ids=np.asarray(ds["labels"]))


def fake_deps(names: list[str]) -> Deps:
    it = iter(names)

    def dataset(d: dict[str, list[Any]]) -> dict[str, Any]:
        return {**d, "name": next(it)}

    return Deps(
        tokenizer=lambda name: FakeTokenizer(),
        model=lambda name, k, id2label, label2id: SimpleNamespace(k=k, id2label=id2label),
        training_args=lambda **kw: kw,
        trainer=FakeTrainer,
        early_stopping=lambda p: ("early_stopping", p),
        collator=lambda tok: "collator",
        dataset=dataset,
        weighted_loss=lambda w: ("weighted_loss", w),
        device=lambda: {"device": "cpu", "gpu_name": None, "n_gpu": 0},
    )


def toy_splits(source: bool = True) -> dict[str, pd.DataFrame]:
    def make(n: int, offset: int) -> pd.DataFrame:
        labs = ["Negative", "Notr", "Positive"]
        rows = [
            {TEXT: f"metin {offset + i} kelime", LABEL: labs[i % 3], SOURCE: f"s{i % 2}"}
            for i in range(n)
        ]
        df = pd.DataFrame(rows)
        return df if source else df.drop(columns=[SOURCE])

    return {"train": make(60, 0), "validation": make(21, 1000), "test": make(30, 2000)}


def toy_config() -> Config:
    raw: dict[str, Any] = {"task": "sentiment", "model": {"name": "fake", "max_length": 16}}
    return Config(
        task="sentiment",
        data=DataConfig(dataset_id="fake", text_column="t", label_column="l"),
        model=ModelConfig(name="fake", max_length=16),
        raw=raw,
    )


def toy_tcfg(**kw: Any) -> TrainConfig:
    return TrainConfig(**({"tag": "toy", "focus_label": "Notr"} | kw))


# --------------------------------------------------------------------------- config


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_all_shipped_configs_parse(path: Path) -> None:
    cfg = load_config(path)
    tcfg = parse_train_config(cfg.raw)
    assert tcfg.warmup_ratio == 0.1 and tcfg.learning_rate == 2e-5
    assert cfg.model.name == "dbmdz/bert-base-turkish-cased"


def test_config_inheritance_overrides_and_keeps_base() -> None:
    cfg = load_config(CONFIG_DIR / "news_berturk_len512_headtail.yaml")
    tcfg = parse_train_config(cfg.raw)
    assert cfg.model.max_length == 512  # from len512 parent
    assert (tcfg.truncation, tcfg.head_tokens, tcfg.tag) == ("head_tail", 128, "len512_headtail")
    assert tcfg.gradient_accumulation_steps == 4  # parent
    assert tcfg.weight_decay == 0.01  # grandparent
    weighted = parse_train_config(load_config(CONFIG_DIR / "sentiment_berturk_weighted.yaml").raw)
    assert weighted.class_weighted_loss and weighted.max_train_samples == 50000


def test_config_inheritance_detects_cycles(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("base: b.yaml\ntask: x\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("base: a.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Circular"):
        load_raw_config(tmp_path / "a.yaml")


def test_parse_train_config_rejects_bad_values() -> None:
    base = {"model": {"max_length": 128}, "training": {"tag": "t"}}
    assert parse_train_config(base).fp16 == "auto"
    bad: list[tuple[dict[str, Any], str]] = [
        ({"tag": "t", "truncation": "middle"}, "truncation"),
        ({"tag": "t", "warmup_ratio": 1.5}, "warmup_ratio"),
        ({"tag": "t", "fp16": "yes"}, "fp16"),
        ({"tag": "t", "bogus": 1}, "Unknown"),
        ({"tag": "t", "truncation": "head_tail", "head_tokens": 127}, "head_tokens"),
        ({"tag": "a b"}, "tag"),
    ]
    for training, match in bad:
        with pytest.raises(ValueError, match=match):
            parse_train_config({**base, "training": training})
    with pytest.raises(ValueError, match="no 'training'"):
        parse_train_config({"model": {"max_length": 128}})


def test_resolve_output_dir_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    tcfg = toy_tcfg(output_dir="from_config")
    monkeypatch.delenv("TCLS_OUTPUT_DIR", raising=False)
    assert resolve_output_dir(tcfg, None) == "from_config"
    monkeypatch.setenv("TCLS_OUTPUT_DIR", "from_env")
    assert resolve_output_dir(tcfg, None) == "from_env"
    assert resolve_output_dir(tcfg, "from_cli") == "from_cli"


# --------------------------------------------------------------------------- data


def test_stratified_subsample_is_seeded_and_stratified() -> None:
    df = pd.DataFrame({TEXT: [f"t{i}" for i in range(1000)], LABEL: ["a"] * 700 + ["b"] * 300})
    s1, s2 = stratified_subsample(df, 100, seed=5), stratified_subsample(df, 100, seed=5)
    pd.testing.assert_frame_equal(s1, s2)
    assert s1[LABEL].value_counts().to_dict() == {"a": 70, "b": 30}
    assert set(stratified_subsample(df, 100, seed=6)[TEXT]) != set(s1[TEXT])
    assert len(stratified_subsample(df, None, seed=5)) == 1000
    assert len(stratified_subsample(df, 5000, seed=5)) == 1000
    with pytest.raises(ValueError):
        stratified_subsample(df, 0, seed=5)


def test_prepare_run_data_smoke_and_max_train_samples() -> None:
    splits = toy_splits()
    tcfg = toy_tcfg(
        max_train_samples=30, smoke=replace(toy_tcfg().smoke, train_samples=15, eval_samples=9)
    )
    full = prepare_run_data(splits, tcfg, seed=1, smoke=False)
    assert full.info["used_sizes"] == {"train": 30, "validation": 21, "test": 30}
    assert full.train[LABEL].value_counts().to_dict() == {
        "Negative": 10,
        "Notr": 10,
        "Positive": 10,
    }
    smoke = prepare_run_data(splits, tcfg, seed=1, smoke=True)
    assert smoke.info["used_sizes"] == {"train": 15, "validation": 9, "test": 9}
    assert "NOT results" in smoke.info["subsample"]
    assert smoke.label2id == {"Negative": 0, "Notr": 1, "Positive": 2}


def test_class_weights_inverse_frequency() -> None:
    w = class_weights(["a"] * 6 + ["b"] * 2, {"a": 0, "b": 1})
    assert w == pytest.approx([8 / (2 * 6), 8 / (2 * 2)])
    with pytest.raises(ValueError, match="no training examples"):
        class_weights(["a"], {"a": 0, "b": 1})


def test_encode_texts_head_and_head_tail() -> None:
    tok = FakeTokenizer()
    text = " ".join(str(i) for i in range(1, 21))  # tokens 1..20
    head = encode_texts([text], tok, max_length=8, truncation="head", head_tokens=0)
    assert head["input_ids"][0] == [101, 1, 2, 3, 4, 5, 6, 102]
    ht = encode_texts([text, "7 8"], tok, max_length=8, truncation="head_tail", head_tokens=2)
    assert ht["input_ids"][0] == [101, 1, 2, 17, 18, 19, 20, 102]
    assert ht["input_ids"][1] == [101, 7, 8, 102]  # short texts untouched
    assert ht["attention_mask"][0] == [1] * 8
    with pytest.raises(ValueError, match="head_tokens"):
        encode_texts([text], tok, max_length=8, truncation="head_tail", head_tokens=6)


def test_special_token_layout_detects_prefix_and_suffix() -> None:
    assert special_token_layout(FakeTokenizer()) == ([101], [102])


def test_smoke_scales_head_tokens_to_smoke_max_length() -> None:
    FakeTrainer.instances.clear()
    result = train_and_evaluate(
        toy_config(),
        toy_tcfg(truncation="head_tail", head_tokens=128),
        seed=0,
        smoke=True,
        output_dir="out",
        splits=toy_splits(),
        deps=fake_deps(["train", "validation", "test"]),
    )
    assert result["hyperparameters"]["max_length"] == 64
    assert result["hyperparameters"]["head_tokens"] == 16
    assert result["smoke"] is True and result["run"].endswith("_smoke")


# --------------------------------------------------------------------------- selection


def test_best_validation_epoch_reads_only_eval_metrics() -> None:
    history = [
        {"eval_macro_f1": 0.70, "epoch": 1.0, "step": 5},
        {"test_macro_f1": 0.99, "epoch": 2.0, "step": 10},  # must be ignored
        {"eval_macro_f1": 0.75, "epoch": 2.0, "step": 10},
        {"eval_macro_f1": 0.72, "epoch": 3.0, "step": 15},
    ]
    assert best_validation_epoch(history) == {"epoch": 2.0, "step": 10, "val_macro_f1": 0.75}
    assert best_validation_epoch([{"test_macro_f1": 0.99}]) is None


def test_training_selects_on_validation_and_scores_test_once() -> None:
    FakeTrainer.instances.clear()
    result = train_and_evaluate(
        toy_config(),
        toy_tcfg(),
        seed=42,
        output_dir="out",
        splits=toy_splits(),
        deps=fake_deps(["train", "validation", "test"]),
    )
    (trainer,) = FakeTrainer.instances
    assert trainer.kwargs["eval_dataset"]["name"] == "validation"
    assert trainer.kwargs["train_dataset"]["name"] == "train"
    assert trainer.calls == ["train", "predict:validation", "predict:test"]
    args = trainer.kwargs["args"]
    assert args["metric_for_best_model"] == "macro_f1" and args["greater_is_better"] is True
    assert args["load_best_model_at_end"] is True and args["eval_strategy"] == "epoch"
    assert args["bf16"] is False and args["fp16"] is False  # cpu + fp16: auto
    assert args["warmup_steps"] == 0.1
    assert args["output_dir"] == str(Path("out") / "berturk_sentiment_toy_seed42")
    assert trainer.kwargs["compute_loss_func"] is None
    assert result["selection"]["best"] == {"epoch": 2.0, "step": 20, "val_macro_f1": 0.8}
    assert (
        result["test"]["macro_f1"] == 1.0 and result["test"]["per_class"]["Notr"]["support"] == 10
    )
    assert result["test_focus_label"]["label"] == "Notr"
    assert set(result["test_by_source"]) == {"s0", "s1"}
    assert result["smoke"] is False and result["data"]["used_sizes"]["train"] == 60


def test_class_weighted_loss_is_wired_in() -> None:
    FakeTrainer.instances.clear()
    result = train_and_evaluate(
        toy_config(),
        toy_tcfg(class_weighted_loss=True),
        seed=1,
        output_dir="out",
        splits=toy_splits(source=False),
        deps=fake_deps(["train", "validation", "test"]),
    )
    loss = FakeTrainer.instances[0].kwargs["compute_loss_func"]
    assert loss[0] == "weighted_loss" and loss[1] == pytest.approx([1.0, 1.0, 1.0])
    assert "test_by_source" not in result


def test_weighted_loss_matches_cross_entropy_for_uniform_weights() -> None:
    torch = pytest.importorskip("torch")
    from tcls.train import _weighted_loss

    logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 0.2, 0.3]])
    labels = torch.tensor([0, 2])
    expected = torch.nn.functional.cross_entropy(logits, labels)
    out = _weighted_loss([1.0, 1.0, 1.0])({"logits": logits}, labels, num_items_in_batch=2)
    assert float(out) == pytest.approx(float(expected), rel=1e-6)
