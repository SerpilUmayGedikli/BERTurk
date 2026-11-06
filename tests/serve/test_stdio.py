"""End-to-end over a real stdio subprocess: proves logs never corrupt stdout (the protocol)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

pytestmark = pytest.mark.anyio


async def test_stdio_roundtrip(model_dir: Path, tmp_path: Path) -> None:
    stderr_file = tmp_path / "stderr.log"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "tcls_mcp"],
        env={**os.environ, "MODEL_PATH": str(model_dir), "DEVICE": "cpu", "MAX_BATCH": "8"},
    )
    with stderr_file.open("w", encoding="utf-8") as errlog:
        async with (
            stdio_client(params, errlog=errlog) as (read, write),
            ClientSession(read, write) as c,
        ):
            await c.initialize()
            r = await c.call_tool("classify_text", {"text": "çok iyi"})
            assert not r.is_error
            bad = await c.call_tool("classify_batch", {"texts": ["iyi"] * 9})
            assert bad.is_error
    log = stderr_file.read_text(encoding="utf-8")
    # Every call is logged with its duration and status, on stderr.
    assert "tool=classify_text status=ok ms=" in log
    assert "tool=classify_batch status=invalid_input" in log
    assert "model loaded" in log
