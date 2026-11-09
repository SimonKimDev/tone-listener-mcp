"""MCP server entry point."""

from __future__ import annotations

from typing import Any, Literal

from mcp.server.mcpserver import MCPServer

from tone_listener_mcp import __version__
from tone_listener_mcp.audio_io import load_audio
from tone_listener_mcp.comparison import compare
from tone_listener_mcp.listener import listen, listen_youtube
from tone_listener_mcp.metrics import measure

mcp = MCPServer(
    name="tone-listener",
    version=__version__,
    instructions=(
        "Start with analyze_youtube for a public YouTube guitar-tone reference. "
        "Use local audio_metrics/analyze_audio/compare_audio when waveform measurements are needed."
    ),
)


@mcp.tool(structured_output=True)
def server_info() -> dict[str, Any]:
    """Return basic information about this server."""
    return {
        "name": "tone-listener-mcp",
        "version": __version__,
        "primary_tool": "analyze_youtube",
        "youtube_local_extraction": False,
    }


@mcp.tool(structured_output=True)
def audio_metrics(
    path: str, start: float | None = None, end: float | None = None
) -> dict[str, Any]:
    """Measure local audio levels, spectrum and stereo. Times are seconds; end is exclusive."""
    return measure(load_audio(path, start, end))


@mcp.tool(structured_output=True)
def analyze_audio(
    path: str,
    start: float | None = None,
    end: float | None = None,
    question: str | None = None,
) -> dict[str, Any]:
    """Measure a clip and optionally upload it to Gemini for guitar listening."""
    clip = load_audio(path, start, end)
    return {"metrics": measure(clip), **listen(clip, question)}


@mcp.tool(structured_output=True)
def analyze_youtube(
    url: str,
    start: float | None = None,
    end: float | None = None,
    question: str | None = None,
) -> dict[str, Any]:
    """Listen to a public YouTube clip via Gemini. Does not extract a WAV or measure local DSP."""
    return listen_youtube(url, start, end, question)


@mcp.tool(structured_output=True)
def compare_audio(
    path_a: str,
    path_b: str,
    start_a: float | None = None,
    end_a: float | None = None,
    start_b: float | None = None,
    end_b: float | None = None,
    focus: str | None = None,
    matching: Literal["active", "whole"] = "active",
) -> dict[str, Any]:
    """Compare B to reference A with RMS matching and optional Gemini A/B listening.

    matching="active" (default) matches gated, non-silent regions so silence padding does
    not change the applied gain; "whole" matches whole-clip RMS including silence and tails.
    """
    return compare(
        load_audio(path_a, start_a, end_a),
        load_audio(path_b, start_b, end_b),
        focus,
        matching,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
