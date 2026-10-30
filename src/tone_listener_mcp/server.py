"""MCP server entry point."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from tone_listener_mcp import __version__

mcp = MCPServer(
    name="tone-listener",
    version=__version__,
    instructions="Tools for listening to and comparing guitar tone recordings.",
)


@mcp.tool()
def server_info() -> dict:
    """Return basic information about this server."""
    return {"name": "tone-listener-mcp", "version": __version__}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
