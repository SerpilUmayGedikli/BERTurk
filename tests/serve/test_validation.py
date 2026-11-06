"""Invalid inputs return clear tool errors, and the server keeps working afterwards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from tcls_mcp.config import MAX_TEXT_CHARS, ConfigError, Settings
from tcls_mcp.model import Classifier
from tcls_mcp.server import build_server

from .conftest import make_settings

pytestmark = pytest.mark.anyio


async def _error(client: Client, tool: str, args: dict[str, Any]) -> str:
    r = await client.call_tool(tool, args)
    assert r.is_error, f"{tool}({args!r:.60}) should fail"
    text: str = r.content[0].text  # type: ignore[union-attr]
    return text


@pytest.mark.parametrize("tool", ["classify_text", "explain_prediction"])
@pytest.mark.parametrize(
    ("text", "expected"),
    [("", "is empty"), ("   \n\t ", "only whitespace"), ("a" * (MAX_TEXT_CHARS + 1), "limit")],
)
async def test_bad_text(client: Client, tool: str, text: str, expected: str) -> None:
    assert expected in await _error(client, tool, {"text": text})


async def test_wrong_type_is_rejected_by_schema(client: Client) -> None:
    await _error(client, "classify_text", {"text": 123})
    await _error(client, "classify_text", {})
    await _error(client, "classify_batch", {"texts": "not a list"})


async def test_empty_batch(client: Client) -> None:
    assert "empty list" in await _error(client, "classify_batch", {"texts": []})


async def test_batch_over_limit(client: Client) -> None:
    msg = await _error(client, "classify_batch", {"texts": ["iyi"] * 65})
    assert "65 items" in msg and "limit is 64" in msg


async def test_bad_item_in_batch_names_its_index(client: Client) -> None:
    msg = await _error(client, "classify_batch", {"texts": ["iyi", "kötü", "  "]})
    assert "texts[2]" in msg


async def test_max_batch_env_lowers_the_limit(model_dir: Path, classifier: Classifier) -> None:
    server = build_server(make_settings(model_dir, max_batch=8), lambda _: classifier)
    async with Client(server) as c:
        assert "limit is 8" in await _error(c, "classify_batch", {"texts": ["iyi"] * 9})
        r = await c.call_tool("classify_batch", {"texts": ["iyi"] * 8})
        assert not r.is_error


async def test_long_text_is_truncated_and_flagged(client: Client) -> None:
    text = "ürün " * 700
    r = await client.call_tool("classify_text", {"text": text})
    assert not r.is_error and r.structured_content is not None
    out = r.structured_content
    assert out["truncated"] is True
    assert out["input_tokens"] == 702 and out["used_tokens"] == 512
    assert "702 tokens" in out["warning"]

    b = await client.call_tool("classify_batch", {"texts": [text, "iyi"]})
    assert b.structured_content is not None
    assert b.structured_content["truncated_count"] == 1

    e = await client.call_tool("explain_prediction", {"text": text})
    assert e.structured_content is not None
    assert e.structured_content["truncated"] is True
    assert e.structured_content["words_evaluated"] == 510


async def test_server_survives_errors(client: Client) -> None:
    for args in ({"text": ""}, {"text": " "}, {"text": 1}):
        await _error(client, "classify_text", args)
    r = await client.call_tool("classify_text", {"text": "çok iyi"})
    assert not r.is_error


def test_settings_from_env() -> None:
    s = Settings.from_env({"MODEL_PATH": "m", "MAX_BATCH": "16", "DEVICE": "CPU"})
    assert (s.max_batch, s.device, s.model_report) == (16, "cpu", None)
    defaults = Settings.from_env({"MODEL_PATH": "m"})
    assert (defaults.max_batch, defaults.device) == (64, "auto")


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"MODEL_PATH": "  "},
        {"MODEL_PATH": "m", "MAX_BATCH": "0"},
        {"MODEL_PATH": "m", "MAX_BATCH": "65"},
        {"MODEL_PATH": "m", "MAX_BATCH": "abc"},
        {"MODEL_PATH": "m", "DEVICE": "tpu"},
    ],
)
def test_bad_settings(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_missing_model_dir(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"config\.json"):
        Classifier.load(tmp_path, device="cpu")
