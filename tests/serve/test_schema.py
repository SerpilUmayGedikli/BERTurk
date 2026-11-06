"""Tool listing, input/output schemas and the model://card resource."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from mcp import Client

from tcls_mcp.model import Classifier
from tcls_mcp.schemas import BatchClassificationResult, ClassificationResult, Explanation
from tcls_mcp.server import build_server

from .conftest import LABELS, make_settings

pytestmark = pytest.mark.anyio

TOOLS = {"classify_text", "classify_batch", "explain_prediction"}


async def test_tools_are_listed_with_descriptions_and_schemas(client: Client) -> None:
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == TOOLS
    for tool in tools.values():
        assert tool.description and len(tool.description) > 80
        assert tool.output_schema is not None
        assert tool.annotations is not None and tool.annotations.read_only_hint
        props = tool.input_schema["properties"]
        assert "ctx" not in props  # Context is injected, not part of the schema
        for name, prop in props.items():
            assert prop.get("description"), f"{tool.name}.{name} has no description"
    assert tools["classify_batch"].input_schema["properties"]["texts"]["type"] == "array"


async def test_classify_text_output_matches_schema(client: Client) -> None:
    r = await client.call_tool("classify_text", {"text": "Ürün harika, çok memnun kaldım!"})
    assert not r.is_error
    res = ClassificationResult.model_validate(r.structured_content)
    assert list(res.probabilities) == LABELS
    assert math.isclose(sum(res.probabilities.values()), 1.0, abs_tol=1e-4)
    assert res.label == max(res.probabilities, key=lambda k: res.probabilities[k])
    assert res.confidence == res.probabilities[res.label]
    assert res.truncated is False and res.warning is None
    assert res.used_tokens == res.input_tokens


async def test_classify_batch_output_matches_schema(client: Client) -> None:
    texts = ["kargo geç geldi", "ürün çok güzel", "bu bir ürün"]
    r = await client.call_tool("classify_batch", {"texts": texts})
    assert not r.is_error
    res = BatchClassificationResult.model_validate(r.structured_content)
    assert res.count == len(res.results) == 3
    assert set(res.label_counts) == set(LABELS)
    assert sum(res.label_counts.values()) == 3
    assert res.truncated_count == 0


async def test_explain_prediction_output(client: Client) -> None:
    text = "Kargo çok geç geldi, ürün de bozuk çıktı."
    r = await client.call_tool("explain_prediction", {"text": text})
    assert not r.is_error
    res = Explanation.model_validate(r.structured_content)
    assert res.method == "occlusion"
    assert 1 <= len(res.top_tokens) <= 5
    assert res.words_evaluated == 10  # 8 words + "," + "."
    contribs = [t.contribution for t in res.top_tokens]
    assert contribs == sorted(contribs, reverse=True)
    for t in res.top_tokens:
        assert text[t.start : t.end] == t.token
    # The explanation is about the same prediction classify_text makes.
    c = await client.call_tool("classify_text", {"text": text})
    assert c.structured_content is not None
    assert res.label == c.structured_content["label"]


async def test_model_card_resource(client: Client) -> None:
    resources = (await client.list_resources()).resources
    assert [str(r.uri) for r in resources] == ["model://card"]
    content = (await client.read_resource("model://card")).contents[0]
    card = json.loads(content.text)  # type: ignore[union-attr]
    assert card["labels"] == LABELS
    assert card["status"] == "smoke"
    assert card["task"] == "sentiment"
    assert card["test_metrics"]["macro_f1"] == 0.55
    assert any("SMOKE" in w for w in card["warnings"])


async def test_model_is_loaded_once(model_dir: Path, classifier: Classifier) -> None:
    calls = 0

    def factory(_: object) -> Classifier:
        nonlocal calls
        calls += 1
        return classifier

    async with Client(build_server(make_settings(model_dir), factory)) as c:
        for _ in range(3):
            await c.call_tool("classify_text", {"text": "iyi"})
        await c.call_tool("classify_batch", {"texts": ["iyi", "kötü"]})
    assert calls == 1
