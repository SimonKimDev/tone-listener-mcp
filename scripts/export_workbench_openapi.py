"""Export the local workbench's shared Pydantic API schemas for TypeScript."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tone_listener_mcp.workbench.api import create_app
from tone_listener_mcp.workbench.storage import LocalStorage

destination = Path(__file__).resolve().parents[1] / "web" / "src" / "api" / "openapi.json"
destination.parent.mkdir(parents=True, exist_ok=True)
with TemporaryDirectory(prefix="tone-workbench-openapi-") as directory:
    app = create_app(storage=LocalStorage(Path(directory)))
    try:
        schema = app.openapi()
    finally:
        app.state.jobs.close()
destination.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
