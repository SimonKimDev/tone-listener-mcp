"""MCP server entry point."""

from __future__ import annotations

import logging
from typing import Literal

import anyio.from_thread
from mcp.server.mcpserver import Context, MCPServer

from tone_listener_mcp import __version__
from tone_listener_mcp.audio_io import load_audio
from tone_listener_mcp.comparison import YouTubeReference, compare, compare_reference
from tone_listener_mcp.listener import Progress, listen, listen_youtube
from tone_listener_mcp.metrics import measure
from tone_listener_mcp.schemas import (
    AnalyzeAudioResponse,
    AnalyzeYouTubeResponse,
    AudioMetrics,
    CompareResponse,
    CompareYouTubeResponse,
    ServerInfo,
)

logger = logging.getLogger(__name__)

mcp = MCPServer(
    name="tone-listener",
    version=__version__,
    instructions=(
        "Start with analyze_youtube for a public YouTube guitar-tone reference. "
        "Use compare_youtube_reference to compare that reference with a local recording. "
        "Use local audio_metrics/analyze_audio/compare_audio when waveform measurements are needed."
    ),
)


def _progress_reporter(ctx: Context | None) -> Progress | None:
    """Forward listening stages as MCP progress notifications when the client asked for them."""
    if ctx is None:
        return None

    def report(step: float, total: float | None, message: str) -> None:
        try:
            anyio.from_thread.run(ctx.report_progress, step, total, message)
        except Exception:
            logger.debug("Progress notification failed", exc_info=True)

    return report


@mcp.tool(structured_output=True)
def server_info() -> ServerInfo:
    """Return basic information about this server."""
    return ServerInfo.model_validate({
        "name": "tone-listener-mcp",
        "version": __version__,
        "primary_tool": "analyze_youtube",
        "youtube_local_extraction": False,
    })


@mcp.tool(structured_output=True)
def audio_metrics(
    path: str, start: float | None = None, end: float | None = None
) -> AudioMetrics:
    """Measure local audio levels, spectrum and stereo. Times are seconds; end is exclusive."""
    return AudioMetrics.model_validate(measure(load_audio(path, start, end)))


@mcp.tool(structured_output=True)
def analyze_audio(
    path: str,
    start: float | None = None,
    end: float | None = None,
    question: str | None = None,
    ctx: Context | None = None,
) -> AnalyzeAudioResponse:
    """Measure a clip and optionally upload it to Gemini for guitar listening."""
    clip = load_audio(path, start, end)
    return AnalyzeAudioResponse.model_validate(
        {"metrics": measure(clip), **listen(clip, question, _progress_reporter(ctx))}
    )


@mcp.tool(structured_output=True)
def analyze_youtube(
    url: str,
    start: float | None = None,
    end: float | None = None,
    question: str | None = None,
    ctx: Context | None = None,
) -> AnalyzeYouTubeResponse:
    """Listen to a public YouTube clip via Gemini. Does not extract a WAV or measure local DSP."""
    return AnalyzeYouTubeResponse.model_validate(
        listen_youtube(url, start, end, question, _progress_reporter(ctx))
    )


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
    ctx: Context | None = None,
) -> CompareResponse:
    """Compare B to reference A with RMS matching and optional Gemini A/B listening.

    matching="active" (default) matches gated, non-silent regions so silence padding does
    not change the applied gain; "whole" matches whole-clip RMS including silence and tails.
    """
    return CompareResponse.model_validate(compare(
        load_audio(path_a, start_a, end_a),
        load_audio(path_b, start_b, end_b),
        focus,
        matching,
        _progress_reporter(ctx),
    ))


@mcp.tool(structured_output=True)
def compare_youtube_reference(
    url: str,
    path_b: str,
    start: float | None = None,
    end: float | None = None,
    start_b: float | None = None,
    end_b: float | None = None,
    focus: str | None = None,
    ctx: Context | None = None,
) -> CompareYouTubeResponse:
    """Compare a local candidate B with a public YouTube reference interval A via Gemini.

    Provides semantic listening differences and prioritized adjustments only: RMS matching
    and spectral deltas need a local reference (see available_analyses). start/end select
    the YouTube interval; start_b/end_b select the local file.
    """
    reference = YouTubeReference(url, start, end)
    candidate = load_audio(path_b, start_b, end_b)
    return CompareYouTubeResponse.model_validate(compare_reference(
        reference, candidate, focus, progress=_progress_reporter(ctx)
    ))


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
