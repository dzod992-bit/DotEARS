from __future__ import annotations

import json
import sys
from datetime import timedelta

import pytest

mcp = pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


@pytest.mark.integration
async def test_real_stdio_agent_call_and_scope(media_files, tmp_path):
    server = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "dotears",
            "serve",
            "--media-root",
            str(media_files["root"]),
            "--output",
            str(tmp_path / "library"),
        ],
    )
    async with stdio_client(server) as (reader, writer):
        async with ClientSession(
            reader, writer, read_timeout_seconds=timedelta(seconds=120)
        ) as client:
            await client.initialize()
            tools = await client.list_tools()
            assert {t.name for t in tools.tools} == {"analyze_video", "check_audio_environment"}
            health = await client.call_tool("check_audio_environment", {})
            assert not health.isError
            result = await client.call_tool("analyze_video", {"source": str(media_files["video"])})
            assert not result.isError
            payload = json.loads(result.content[0].text)
            assert payload["events"] and payload["audio"]["present"]
            for bad in (str(tmp_path / "outside.mp4"), "https://example.com/video"):
                denied = await client.call_tool("analyze_video", {"source": bad})
                assert denied.isError
