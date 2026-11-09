"""Exercise the actual stdio server process, including structured results and tool errors."""

import asyncio
import os
import shutil
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
        executable = shutil.which("tone-listener-mcp", path=str(Path(sys.executable).parent))
        assert executable is not None, "Install the project to test the console entry point"
        params = StdioServerParameters(
            command=executable,
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
                "server_info", "audio_metrics", "analyze_audio", "compare_audio",
                "analyze_youtube", "compare_youtube_reference",
            }
            schemas = {tool.name: tool.output_schema for tool in tools.tools}
            compare_schema = schemas["compare_audio"]
            assert {"loudness_match", "deltas_b_minus_a", "eq_direction_hints", "listening"} <= set(
                compare_schema["properties"]
            )
            assert "EqHint" in compare_schema["$defs"] and "Adjustment" in compare_schema["$defs"]
            assert "spectrum" in schemas["audio_metrics"]["properties"]
            assert "source" in schemas["analyze_youtube"]["properties"]
            info = await session.call_tool("server_info")
            assert not info.is_error
            assert info.structured_content["name"] == "tone-listener-mcp"
            assert info.structured_content["primary_tool"] == "analyze_youtube"
            assert info.structured_content["youtube_local_extraction"] is False
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
            youtube = await session.call_tool(
                "analyze_youtube", {"url": "https://youtu.be/9hE5-98ZeCg", "start": 12, "end": 20}
            )
            assert not youtube.is_error
            assert youtube.structured_content["metrics"] is None
            assert youtube.structured_content["listening_unavailable"]
            assert youtube.structured_content["input_transport"] == "youtube_url"
            mixed = await session.call_tool(
                "compare_youtube_reference",
                {"url": "https://youtu.be/9hE5-98ZeCg", "path_b": str(path), "start": 1, "end": 5},
            )
            assert not mixed.is_error
            assert mixed.structured_content["available_analyses"]["rms_matching"] is False
            assert mixed.structured_content["listening_unavailable"]
            invalid_url = await session.call_tool(
                "analyze_youtube", {"url": "https://example.invalid/video"}
            )
            assert invalid_url.is_error
            invalid = await session.call_tool(
                "audio_metrics", {"path": str(tmp_path / "missing.wav")}
            )
            assert invalid.is_error

    asyncio.run(exercise())
