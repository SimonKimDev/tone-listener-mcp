"""Exercise the actual stdio server process, including structured results and tool errors."""

import asyncio
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_stdio_tools(tmp_path):
    path = tmp_path / "tone.wav"
    sf.write(path, 0.25 * np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000), 16000)

    async def exercise():
        environment = {**os.environ, "GEMINI_API_KEY": ""}
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "tone_listener_mcp.server"],
            env=environment,
            cwd=Path(__file__).resolve().parents[1],
        )
        async with (
            asyncio.timeout(30),
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            tools = await session.list_tools()
            assert {tool.name for tool in tools.tools} == {
                "server_info", "audio_metrics", "analyze_audio", "compare_audio"
            }
            info = await session.call_tool("server_info")
            assert not info.is_error
            assert info.structured_content["name"] == "tone-listener-mcp"
            result = await session.call_tool("audio_metrics", {"path": str(path)})
            assert not result.is_error
            assert abs(result.structured_content["spectrum"]["centroid_hz"] - 1000) < 5
            analysis = await session.call_tool("analyze_audio", {"path": str(path)})
            assert not analysis.is_error
            assert analysis.structured_content["listening_unavailable"]
            comparison = await session.call_tool(
                "compare_audio", {"path_a": str(path), "path_b": str(path)}
            )
            assert not comparison.is_error
            assert comparison.structured_content["loudness_match"]["gain_db_applied_to_b"] == 0
            invalid = await session.call_tool(
                "audio_metrics", {"path": str(tmp_path / "missing.wav")}
            )
            assert invalid.is_error

    asyncio.run(exercise())
