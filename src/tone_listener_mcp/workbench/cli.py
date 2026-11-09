"""CLI entry point for the loopback-only tone workbench."""

from __future__ import annotations

import argparse
import os
import re

import uvicorn

from tone_listener_mcp.config import load_default_credentials
from tone_listener_mcp.workbench.api import create_app
from tone_listener_mcp.workbench.privacy import SafeLogFormatter, redact_paths
from tone_listener_mcp.workbench.storage import LocalStorage, StorageError


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Tone Listener Workbench.")
    parser.add_argument("--port", type=int, default=8765, help="Local loopback port (default: 8765)")
    parser.add_argument("--model", help="Explicit Gemini model selection, without automatic fallback")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    try:
        load_default_credentials()
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Could not load local credentials: {redact_paths(str(exc))}\n")
    if args.model is not None:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", args.model):
            parser.error("--model must be a plain model identifier")
        os.environ["TONE_LISTENER_MODEL"] = args.model
    port = args.port
    try:
        storage = LocalStorage()
        storage.sanitize_saved_projects()
    except (OSError, StorageError):
        parser.exit(2, "Could not initialize or sanitize private workbench storage.\n")
    app = create_app(
        storage=storage,
        allowed_hosts={
            f"127.0.0.1:{port}",
            f"localhost:{port}",
        }
    )
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False,
        log_config={
            "version": 1, "disable_existing_loggers": False,
            "formatters": {"safe": {"()": SafeLogFormatter, "format": "%(levelname)s: %(message)s"}},
            "handlers": {"safe": {"class": "logging.StreamHandler", "formatter": "safe", "stream": "ext://sys.stderr"}},
            "loggers": {
                "uvicorn": {"handlers": ["safe"], "level": "INFO", "propagate": False},
                "tone_listener_mcp": {"handlers": ["safe"], "level": "INFO", "propagate": False},
            },
        },
    )
