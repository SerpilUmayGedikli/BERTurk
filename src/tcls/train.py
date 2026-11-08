"""BERTurk fine-tuning with the Hugging Face Trainer API.

    python -m tcls.train --config configs/news_berturk_len256.yaml --seed 42 [--smoke]

Protocol (per run):
1. Read the processed splits written by ``python -m tcls.data`` (train / validation / test).
2. Optionally subsample train (``max_train_samples``; stratified, seeded, recorded in the JSON).
3. Fine-tune; the Trainer evaluates on VALIDATION each epoch, early-stops on validation
   macro-F1 and reloads the best-on-validation checkpoint at the end.
4. Score the selected model on validation, then on test exactly once, and write
   ``<report_dir>/berturk_<task>_<tag>_seed<N>.json``.

Checkpoints go to ``--output-dir`` / ``$TCLS_OUTPUT_DIR`` / ``training.output_dir`` (in that
order), which should be outside the repo (e.g. Google Drive). torch/transformers are imported
lazily so that config parsing and tests work without them.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tcls.data import LABEL, SOURCE, TEXT, build_label_mapping, stratified_subsample
from tcls.metrics import compute_metrics, focus_label_metrics, source_breakdown
from tcls.utils import Config, get_logger, load_config, set_seed

logger = get_logger(__name__)

TRUNCATIONS = ("head", "head_tail")
BEST_METRIC = "macro_f1"  # Trainer logs it as "eval_macro_f1" (validation only)


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class SmokeConfig:
    train_samples: int = 200
    eval_samples: int = 100
    max_length: int = 64
    epochs: float = 1.0
    batch_size: int = 16


@dataclass(frozen=True)
class TrainConfig:
    tag: str
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 4.0
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    # "auto" = fp16 iff CUDA is available. bf16 is never used (Colab T4 has no bf16).
    fp16: bool | str = "auto"
    early_stopping_patience: int = 1
    class_weighted_loss: bool = False
    max_train_samples: int | None = None
    truncation: str = "head"
    head_tokens: int = 128
    output_dir: str = "checkpoints"
    save_total_limit: int = 2
    logging_steps: int = 50
    report_dir: str = "reports"
    focus_label: str | None = None
    smoke: SmokeConfig = field(default_factory=SmokeConfig)

    def __post_init__(self) -> None:
        if not self.tag or any(c in self.tag for c in "/\\ "):
            raise ValueError(f"tag must be a non-empty name without spaces/slashes: {self.tag!r}")
        if self.truncation not in TRUNCATIONS:
            raise ValueError(f"truncation must be one of {TRUNCATIONS}, got {self.truncation!r}")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.fp16 not in (True, False, "auto"):
            raise ValueError(f"fp16 must be true, false or 'auto', got {self.fp16!r}")
        if self.max_train_samples is not None and self.max_train_samples <= 0:
            raise ValueError("max_train_samples must be positive or null")
        for name in (
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "early_stopping_patience",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")


def parse_train_config(raw: Mapping[str, Any]) -> TrainConfig:
    if "training" not in raw:
        raise ValueError("Config has no 'training' section")
    t = dict(raw["training"])
    smoke = SmokeConfig(**(t.pop("smoke", None) or {}))
    unknown = set(t) - {f for f in TrainConfig.__dataclass_fields__ if f != "smoke"}
    if unknown:
        raise ValueError(f"Unknown training keys: {sorted(unknown)}")
    tcfg = TrainConfig(**t, smoke=smoke)
    max_length = int(raw["model"]["max_length"])
    if tcfg.truncation == "head_tail" and not 0 < tcfg.head_tokens < max_length - 2:
        raise ValueError(f"head_tokens must be in (0, max_length - 2), got {tcfg.head_tokens}")
    return tcfg


def resolve_output_dir(tcfg: TrainConfig, cli_value: str | None) -> str:
    return cli_value or os.environ.get("TCLS_OUTPUT_DIR") or tcfg.output_dir


def run_name(task: str, tag: str, seed: int, smoke: bool) -> str:
    return f"berturk_{task}_{tag}_seed{seed}" + ("_smoke" if smoke else "")


# --------------------------------------------------------------------------- data


@dataclass
class RunData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    label2id: dict[str, int]
    id2label: dict[int, str]
    info: dict[str, Any]


def load_processed(processed_dir: str | Path) -> dict[str, pd.DataFrame]:
    d = Path(processed_dir)
    names = ("train", "validation", "test")
    missing = [s for s in names if not (d / f"{s}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {d}; run `python -m tcls.data --config ...` first"
        )
    return {s: pd.read_parquet(d / f"{s}.parquet") for s in names}


def prepare_run_data(
    splits: Mapping[str, pd.DataFrame], tcfg: TrainConfig, seed: int, smoke: bool
) -> RunData:
    """Apply smoke / max_train_samples subsampling. Label ids come from the FULL train split."""
    full_train = splits["train"]
    label2id, id2label = build_label_mapping(full_train[LABEL].tolist())
    info: dict[str, Any] = {
        "full_sizes": {k: len(v) for k, v in splits.items()},
        "max_train_samples": tcfg.max_train_samples,
    }
    if smoke:
        s = tcfg.smoke
        train = stratified_subsample(full_train, s.train_samples, seed)
        validation = stratified_subsample(splits["validation"], s.eval_samples, seed)
        test = stratified_subsample(splits["test"], s.eval_samples, seed)
        info["subsample"] = (
            f"SMOKE: stratified {s.train_samples} train / {s.eval_samples} validation / "
            f"{s.eval_samples} test (seed={seed}); metrics are NOT results"
        )
    else:
        train = stratified_subsample(full_train, tcfg.max_train_samples, seed)
        validation, test = splits["validation"], splits["test"]
        info["subsample"] = (
            f"train: stratified {len(train)} of {len(full_train)} (seed={seed})"
            if len(train) < len(full_train)
            else "none (full train split)"
        )
    info["used_sizes"] = {"train": len(train), "validation": len(validation), "test": len(test)}
    return RunData(train, validation, test, label2id, id2label, info)


def class_weights(labels: Sequence[str], label2id: Mapping[str, int]) -> list[float]:
    """Inverse-frequency weights ``n / (K * n_c)`` (mean weight over samples is 1)."""
    counts = pd.Series(list(labels)).value_counts()
    n, k = len(labels), len(label2id)
    weights = [0.0] * k
    for name, idx in label2id.items():
        c = int(counts.get(name, 0))
        if c == 0:
            raise ValueError(f"Label {name!r} has no training examples")
        weights[idx] = n / (k * c)
    return weights


def encode_texts(
    texts: Sequence[str], tokenizer: Any, max_length: int, truncation: str, head_tokens: int
) -> dict[str, list[list[int]]]:
    """Tokenize. ``head`` keeps the first tokens; ``head_tail`` keeps ``head_tokens`` from the
    start plus the rest of the budget from the end (Sun et al., 2019)."""
    if truncation == "head":
        enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        return {"input_ids": list(enc["input_ids"]), "attention_mask": list(enc["attention_mask"])}
    prefix, suffix = special_token_layout(tokenizer)
    budget = max_length - len(prefix) - len(suffix)
    if not 0 < head_tokens < budget:
        raise ValueError(f"head_tokens={head_tokens} must be in (0, {budget}) for {max_length=}")
    raw = tokenizer(list(texts), add_special_tokens=False, truncation=False)["input_ids"]
    input_ids: list[list[int]] = []
    for ids in raw:
        ids = list(ids)
        if len(ids) > budget:
            ids = ids[:head_tokens] + ids[len(ids) - (budget - head_tokens) :]
        input_ids.append([*prefix, *ids, *suffix])
    return {"input_ids": input_ids, "attention_mask": [[1] * len(x) for x in input_ids]}


def special_token_layout(
    tokenizer: Any, probe: str = "merhaba dünya"
) -> tuple[list[int], list[int]]:
    """Special tokens the tokenizer adds before/after a single sequence (e.g. [CLS] / [SEP]).

    Found by diffing a probe encoded with and without special tokens, which works across
    tokenizer versions (transformers 5 dropped ``build_inputs_with_special_tokens``)."""
    full = list(tokenizer([probe])["input_ids"][0])
    body = list(tokenizer([probe], add_special_tokens=False)["input_ids"][0])
    for i in range(len(full) - len(body) + 1):
        if full[i : i + len(body)] == body:
            return full[:i], full[i + len(body) :]
    raise ValueError("Could not locate the text tokens inside the special-token encoding")


# --------------------------------------------------------------------------- selection


def best_validation_epoch(log_history: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Best epoch by validation macro-F1 from Trainer logs. Only ``eval_*`` keys are read."""
    key = f"eval_{BEST_METRIC}"
    evals = [h for h in log_history if key in h]
    if not evals:
        return None
    best = max(evals, key=lambda h: float(h[key]))  # first max wins on ties
    return {"epoch": best.get("epoch"), "step": best.get("step"), "val_macro_f1": float(best[key])}


def validation_curve(log_history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    key = f"eval_{BEST_METRIC}"
    return [
        {"epoch": h.get("epoch"), "val_macro_f1": float(h[key]), "val_loss": h.get("eval_loss")}
        for h in log_history
        if key in h
    ]


def logits_to_labels(logits: Any, id2label: Mapping[int, str]) -> list[str]:
    return [id2label[int(i)] for i in np.asarray(logits).argmax(axis=-1)]


# --------------------------------------------------------------------------- dependencies


@dataclass
class Deps:
    """Factories for heavy objects; tests replace them with fakes."""

    tokenizer: Callable[[str], Any]
    model: Callable[[str, int, dict[int, str], dict[str, int]], Any]
    training_args: Callable[..., Any]
    trainer: Callable[..., Any]
    early_stopping: Callable[[int], Any]
    collator: Callable[[Any], Any]
    dataset: Callable[[dict[str, list[Any]]], Any]
    weighted_loss: Callable[[list[float]], Any]
    device: Callable[[], dict[str, Any]]


def _weighted_loss(weights: list[float]) -> Callable[..., Any]:
    import torch
    import torch.nn.functional as F  # noqa: N812

    w = torch.tensor(weights, dtype=torch.float32)

    def loss_fn(outputs: Any, labels: Any, num_items_in_batch: Any = None) -> Any:
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            labels.view(-1),
            weight=w.to(logits.device),
            reduction="sum",
        )
        # The Trainer does not rescale for gradient accumulation when compute_loss_func is
        # set, so normalise by the number of labels in the whole accumulated batch.
        denom = num_items_in_batch if num_items_in_batch is not None else labels.numel()
        return loss / denom

    return loss_fn


def _device_info() -> dict[str, Any]:
    import torch

    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "n_gpu": torch.cuda.device_count(),
        }
    return {"device": "cpu", "gpu_name": None, "n_gpu": 0, "cpu": platform.processor()}


def default_deps() -> Deps:
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    return Deps(
        tokenizer=lambda name: AutoTokenizer.from_pretrained(name),
        model=lambda name, k, id2label, label2id: (
            AutoModelForSequenceClassification.from_pretrained(
                name, num_labels=k, id2label=id2label, label2id=label2id
            )
        ),
        training_args=TrainingArguments,
        trainer=Trainer,
        early_stopping=lambda p: EarlyStoppingCallback(early_stopping_patience=p),
        collator=lambda tok: DataCollatorWithPadding(tokenizer=tok),
        dataset=Dataset.from_dict,
        weighted_loss=_weighted_loss,
        device=_device_info,
    )


# --------------------------------------------------------------------------- run


def train_and_evaluate(
    cfg: Config,
    tcfg: TrainConfig,
    seed: int,
    *,
    smoke: bool = False,
    output_dir: str,
    splits: Mapping[str, pd.DataFrame] | None = None,
    deps: Deps | None = None,
) -> dict[str, Any]:
    deps = deps or default_deps()
    set_seed(seed)
    data = prepare_run_data(
        splits if splits is not None else load_processed(cfg.data.processed_dir),
        tcfg,
        seed,
        smoke,
    )
    device = deps.device()
    max_length = tcfg.smoke.max_length if smoke else cfg.model.max_length
    # Smoke shrinks max_length, so scale head+tail's head down with it (recorded below).
    head_tokens = min(tcfg.head_tokens, max_length // 4) if smoke else tcfg.head_tokens
    epochs = tcfg.smoke.epochs if smoke else tcfg.num_train_epochs
    train_bs = tcfg.smoke.batch_size if smoke else tcfg.per_device_train_batch_size
    eval_bs = tcfg.smoke.batch_size if smoke else tcfg.per_device_eval_batch_size
    fp16 = device["device"] == "cuda" if tcfg.fp16 == "auto" else bool(tcfg.fp16)
    name = run_name(cfg.task, tcfg.tag, seed, smoke)

    tokenizer = deps.tokenizer(cfg.model.name)

    def to_dataset(df: pd.DataFrame) -> Any:
        enc = encode_texts(df[TEXT].tolist(), tokenizer, max_length, tcfg.truncation, head_tokens)
        return deps.dataset({**enc, "labels": [data.label2id[x] for x in df[LABEL]]})

    train_ds, val_ds, test_ds = (to_dataset(d) for d in (data.train, data.validation, data.test))
    model = deps.model(cfg.model.name, len(data.label2id), data.id2label, data.label2id)

    def metric_fn(eval_pred: Any) -> dict[str, float]:
        logits, label_ids = eval_pred
        pred = np.asarray(logits).argmax(axis=-1)
        gold = np.asarray(label_ids)
        from sklearn.metrics import accuracy_score, f1_score

        return {
            BEST_METRIC: float(f1_score(gold, pred, average="macro")),
            "accuracy": float(accuracy_score(gold, pred)),
        }

    args = deps.training_args(
        output_dir=str(Path(output_dir) / name),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=BEST_METRIC,
        greater_is_better=True,
        save_total_limit=tcfg.save_total_limit,
        learning_rate=tcfg.learning_rate,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=1 if smoke else tcfg.gradient_accumulation_steps,
        num_train_epochs=epochs,
        weight_decay=tcfg.weight_decay,
        warmup_steps=tcfg.warmup_ratio,  # float < 1 = ratio of total steps (transformers 5)
        fp16=fp16,
        bf16=False,
        seed=seed,
        data_seed=seed,
        logging_steps=tcfg.logging_steps,
        report_to="none",
    )
    weights = class_weights(data.train[LABEL].tolist(), data.label2id)
    trainer = deps.trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,  # model selection sees validation only
        processing_class=tokenizer,
        data_collator=deps.collator(tokenizer),
        compute_metrics=metric_fn,
        callbacks=[deps.early_stopping(tcfg.early_stopping_patience)],
        compute_loss_func=deps.weighted_loss(weights) if tcfg.class_weighted_loss else None,
    )

    t0 = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - t0
    log_history = list(trainer.state.log_history)
    best = best_validation_epoch(log_history)

    # Best-on-validation weights are loaded (load_best_model_at_end). Validation first...
    val_out = trainer.predict(val_ds)
    val_pred = logits_to_labels(val_out.predictions, data.id2label)
    labels = list(data.label2id)
    validation = compute_metrics(data.validation[LABEL].tolist(), val_pred, labels)
    # ...then test, exactly once.
    t1 = time.perf_counter()
    test_out = trainer.predict(test_ds)
    test_pred = logits_to_labels(test_out.predictions, data.id2label)
    test = compute_metrics(data.test[LABEL].tolist(), test_pred, labels)

    result: dict[str, Any] = {
        "run": name,
        "smoke": smoke,
        "task": cfg.task,
        "tag": tcfg.tag,
        "seed": seed,
        "model": cfg.model.name,
        "dataset_id": cfg.data.dataset_id,
        "hyperparameters": {
            **{k: v for k, v in asdict(tcfg).items() if k not in ("smoke", "output_dir")},
            "max_length": max_length,
            "head_tokens": head_tokens,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": train_bs,
            "per_device_eval_batch_size": eval_bs,
            "effective_train_batch_size": train_bs
            * (1 if smoke else tcfg.gradient_accumulation_steps)
            * max(1, int(device["n_gpu"])),
            "fp16": fp16,
            "class_weights": dict(zip(labels, weights, strict=True))
            if tcfg.class_weighted_loss
            else None,
        },
        "data": data.info,
        "labels": labels,
        "device": device,
        "train_seconds": round(train_seconds, 1),
        "test_predict_seconds": round(time.perf_counter() - t1, 1),
        "selection": {
            "metric": f"validation {BEST_METRIC}",
            "best": best,
            "best_checkpoint": getattr(trainer.state, "best_model_checkpoint", None),
            "curve": validation_curve(log_history),
            "epochs_run": log_history[-1].get("epoch") if log_history else None,
        },
        "validation": validation,
        "test": test,
        "versions": _versions(),
    }
    if tcfg.focus_label is not None:
        result["test_focus_label"] = {
            "label": tcfg.focus_label,
            **focus_label_metrics(test, tcfg.focus_label),
        }
    if SOURCE in data.test.columns:
        result["test_by_source"] = source_breakdown(
            data.test[SOURCE].astype(str).tolist(), data.test[LABEL].tolist(), test_pred, labels
        )
    return result


def _versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {"python": platform.python_version()}
    for pkg in ("torch", "transformers", "datasets", "scikit-learn"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def report_path(report_dir: str | Path, task: str, tag: str, seed: int, smoke: bool) -> Path:
    return Path(report_dir) / f"{run_name(task, tag, seed, smoke)}.json"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BERTurk (validation-selected).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true", help="Tiny CPU-friendly pipeline check")
    parser.add_argument("--output-dir", default=None, help="Checkpoint root (outside the repo)")
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Where the run JSON goes (default: training.report_dir; data/smoke_reports for smoke)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Do nothing if this run's JSON exists"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    tcfg = parse_train_config(cfg.raw)
    report_dir = args.report_dir or ("data/smoke_reports" if args.smoke else tcfg.report_dir)
    out = report_path(report_dir, cfg.task, tcfg.tag, args.seed, args.smoke)
    if args.skip_existing and out.exists():
        logger.info("Skipping %s (exists)", out)
        return

    result = train_and_evaluate(
        cfg,
        tcfg,
        args.seed,
        smoke=args.smoke,
        output_dir=resolve_output_dir(tcfg, args.output_dir),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "%s: val macro-F1=%.4f test macro-F1=%.4f -> %s%s",
        result["run"],
        result["validation"]["macro_f1"],
        result["test"]["macro_f1"],
        out,
        " (SMOKE: pipeline check only, not a result)" if args.smoke else "",
    )


if __name__ == "__main__":
    main()
