"""Fixtures for the MCP server tests.

A tiny randomly initialised BERT (2 layers, hidden 32) with a hand-made WordPiece vocab is
saved to a temp dir, so the tests run offline in seconds and exercise the real
``Classifier.load`` path. Its predictions are meaningless; the tests check behaviour and
schemas, not accuracy.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("mcp") is None:
    # `uv sync` without the `serve` extra: skip this directory instead of failing.
    collect_ignore_glob = ["test_*.py"]
else:
    from mcp import Client

    from tcls_mcp.config import Settings
    from tcls_mcp.model import Classifier
    from tcls_mcp.server import build_server

LABELS = ["Negative", "Notr", "Positive"]
_WORDS = """
ürün harika çok memnun kaldım kargo geç geldi bozuk çıktı berbat iyi kötü fiyat
güzel hızlı teslimat paket hasarlı tavsiye ederim etmem bu bir ve ama da de ile
için kalite sipariş iade satıcı ilgili değil gibi en
"""
WORDS = _WORDS.split()
CHARS = list("abcçdefgğhıijklmnoöprsştuüvyzqwxABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ0123456789")
PUNCT = list(".,!?;:'\"()-")


def _make_tiny_model(path: Path) -> None:
    import torch

    # transformers' stubs are incomplete for these classes; use it untyped here.
    tf: Any = importlib.import_module("transformers")

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS, *CHARS, *PUNCT]
    vocab += [f"##{c}" for c in CHARS]
    vocab_file = path / "vocab.txt"
    path.mkdir(parents=True, exist_ok=True)
    vocab_file.write_text("\n".join(dict.fromkeys(vocab)) + "\n", encoding="utf-8")
    tok = tf.BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=False)

    torch.manual_seed(0)
    config = tf.BertConfig(
        vocab_size=tok.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={lab: i for i, lab in enumerate(LABELS)},
    )
    tf.BertForSequenceClassification(config).save_pretrained(path)
    tok.save_pretrained(path)
    vocab_file.unlink()  # force the fast tokenizer.json path, like real checkpoints


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("tiny_berturk")
    _make_tiny_model(path)
    return path


@pytest.fixture(scope="session")
def report_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A run JSON shaped like tcls.train's output (smoke run)."""
    report: dict[str, Any] = {
        "run": "berturk_sentiment_tiny_seed42_smoke",
        "smoke": True,
        "task": "sentiment",
        "model": "dbmdz/bert-base-turkish-cased",
        "dataset_id": "winvoker/turkish-sentiment-analysis-dataset",
        "hyperparameters": {"max_length": 64},
        "data": {
            "full_sizes": {"train": 394642, "validation": 43850, "test": 48924},
            "used_sizes": {"train": 200, "validation": 100, "test": 100},
            "subsample": "SMOKE",
        },
        "labels": LABELS,
        "test": {
            "accuracy": 0.77,
            "macro_f1": 0.55,
            "per_class": {lab: {"f1": 0.5} for lab in LABELS},
        },
    }
    path = tmp_path_factory.mktemp("reports") / "run.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def classifier(model_dir: Path) -> Classifier:
    return Classifier.load(model_dir, device="cpu")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def make_settings(model_dir: Path, report: Path | None = None, **kw: Any) -> Settings:
    return Settings(model_path=model_dir, device="cpu", model_report=report, **kw)


@pytest.fixture
async def client(
    model_dir: Path, report_path: Path, classifier: Classifier
) -> AsyncIterator[Client]:
    # Reuse the session-scoped classifier: the tests also prove the lifespan only calls
    # the factory once per server, not per tool call.
    server = build_server(make_settings(model_dir, report_path), lambda _: classifier)
    async with Client(server) as c:
        yield c
