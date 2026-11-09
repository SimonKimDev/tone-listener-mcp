"""Smoke-test an installed tone-listener-mcp executable over real MCP stdio.

Usage: python scripts/smoke_installed.py /path/to/venv/bin/tone-listener-mcp

The executable may be an absolute or caller-relative path, or a command name on PATH.
Runs without GEMINI_API_KEY or network: lists tools, calls server_info and measures a
generated sine wave with audio_metrics. The server runs in a temporary directory, not
the caller's checkout, and the audio is passed relative to that directory.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED_TOOLS = {
    "server_info", "analyze_youtube", "audio_metrics", "analyze_audio", "compare_audio",
    "compare_youtube_reference",
}


def resolve_executable(executable: str) -> str:
    """Resolve paths against the caller's cwd and names via PATH before the cwd changes."""
    found = shutil.which(executable)
    if found is None:
        kind = "path" if os.path.dirname(executable) else "command on PATH"
        raise SystemExit(f"Executable {kind} not found or not executable: {executable}")
    return os.path.abspath(found)


async def smoke(executable: str, audio: Path) -> None:
    environment = {**os.environ, "GEMINI_API_KEY": ""}
    params = StdioServerParameters(command=executable, env=environment, cwd=audio.parent)
    async with (
        asyncio.timeout(60),
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = {tool.name for tool in (await session.list_tools()).tools}
        if tools != EXPECTED_TOOLS:
            raise SystemExit(f"Unexpected tools: {sorted(tools)}")
        info = await session.call_tool("server_info")
        if info.is_error or info.structured_content["listening_available"] is not False:
            raise SystemExit(f"server_info failed: {info.structured_content}")
        # A relative path only resolves if the server really runs in the audio directory.
        metrics = await session.call_tool("audio_metrics", {"path": audio.name})
        if metrics.is_error or abs(metrics.structured_content["spectrum"]["centroid_hz"] - 1000) > 5:
            raise SystemExit("audio_metrics failed")
        print(f"ok: {info.structured_content['name']} {info.structured_content['version']}, "
              f"{len(tools)} tools")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("executable")
    executable = resolve_executable(parser.parse_args().executable)
    with tempfile.TemporaryDirectory() as directory:
        audio = Path(directory) / "sine.wav"
        rate = 44_100
        sf.write(audio, 0.5 * np.sin(2 * np.pi * 1000 * np.arange(rate) / rate), rate)
        asyncio.run(smoke(executable, audio))


if __name__ == "__main__":
    main()
