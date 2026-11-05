"""MCP server entry point."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from tone_listener_mcp import __version__
from tone_listener_mcp.audio_io import load_audio
from tone_listener_mcp.comparison import compare
from tone_listener_mcp.listener import listen
from tone_listener_mcp.metrics import measure

mcp = MCPServer(
    name="tone-listener",
    version=__version__,
    instructions="Tools for listening to and comparing guitar tone recordings.",
)


@mcp.tool()
def server_info() -> dict:
    """Return basic information about this server."""
    return {"name": "tone-listener-mcp", "version": __version__}


@mcp.tool()
def audio_metrics(path: str, start: float | None = None, end: float | None = None) -> dict:
    """Measure local audio levels, spectrum and stereo. Times are seconds; end is exclusive."""
    return measure(load_audio(path, start, end))


@mcp.tool()
def analyze_audio(
    path: str,
    start: float | None = None,
    end: float | None = None,
    question: str | None = None,
) -> dict:
    """Measure a clip and optionally upload it to Gemini for guitar listening."""
    clip = load_audio(path, start, end)
    return {"metrics": measure(clip), **listen(clip, question)}


@mcp.tool()
def compare_audio(
    path_a: str,
    path_b: str,
    start_a: float | None = None,
    end_a: float | None = None,
    start_b: float | None = None,
    end_b: float | None = None,
    focus: str | None = None,
) -> dict:
    """Compare B to reference A with RMS matching and optional Gemini A/B listening."""
    return compare(
        load_audio(path_a, start_a, end_a),
        load_audio(path_b, start_b, end_b),
        focus,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
