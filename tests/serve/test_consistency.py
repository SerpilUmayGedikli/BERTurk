"""classify_batch must give the same answer as classify_text for every text."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from mcp import Client

from tcls_mcp.card import build_card
from tcls_mcp.explain import explain
from tcls_mcp.model import Classifier

from .conftest import LABELS

pytestmark = pytest.mark.anyio

TEXTS = [
    "Ürün harika, çok memnun kaldım!",
    "kargo geç geldi",
    "Berbat bir satıcı, paket hasarlı geldi ve iade de etmediler.",
    "bu",
    "fiyat iyi ama kalite kötü " * 30,  # long enough to need lots of padding in the batch
    "ürün " * 600,  # truncated
]


async def test_batch_equals_single(client: Client) -> None:
    batch = await client.call_tool("classify_batch", {"texts": TEXTS})
    assert batch.structured_content is not None
    results = batch.structured_content["results"]
    for text, b in zip(TEXTS, results, strict=True):
        single = await client.call_tool("classify_text", {"text": text})
        s = single.structured_content
        assert s is not None
        assert s["label"] == b["label"]
        assert (s["truncated"], s["input_tokens"], s["used_tokens"]) == (
            b["truncated"],
            b["input_tokens"],
            b["used_tokens"],
        )
        # Padding changes float summation order, so allow tiny numeric differences.
        np.testing.assert_allclose(
            [s["probabilities"][k] for k in LABELS],
            [b["probabilities"][k] for k in LABELS],
            atol=1e-5,
        )


def test_order_is_preserved(classifier: Classifier) -> None:
    forward = classifier.classify(TEXTS)
    backward = classifier.classify(TEXTS[::-1])[::-1]
    assert [r.label for r in forward] == [r.label for r in backward]


def test_inference_chunking_does_not_change_results(classifier: Classifier) -> None:
    ids = [classifier.encode(t).input_ids for t in TEXTS]
    np.testing.assert_allclose(
        classifier.probabilities(ids, batch_size=64),
        classifier.probabilities(ids, batch_size=2),
        atol=1e-5,
    )


def test_occlusion_contribution_matches_definition(classifier: Classifier) -> None:
    text = "kargo çok geç geldi ve ürün bozuk"
    exp = explain(classifier, text)
    top = exp.top_tokens[0]
    masked = text[: top.start] + classifier.tokenizer.mask_token + text[top.end :]
    p_masked = classifier.classify([masked])[0].probabilities[exp.label]
    assert top.contribution == pytest.approx(exp.confidence - p_masked, abs=1e-4)


def test_card_without_report(model_dir: Path) -> None:
    card = build_card(model_dir / "checkpoint-1", LABELS, 512, "cpu", None)
    assert card.status == "unknown" and card.test_metrics is None
    assert card.name == model_dir.name
    assert any("MODEL_REPORT" in w for w in card.warnings)


def test_card_from_report(model_dir: Path, report_path: Path) -> None:
    card = build_card(model_dir, LABELS, 512, "cpu", report_path)
    assert card.status == "smoke"
    assert card.training_data["license"] == "CC BY-SA 4.0"
    assert any("max_length=64" in w for w in card.warnings)
    assert card.test_metrics is not None and card.test_metrics.test_size == 100
