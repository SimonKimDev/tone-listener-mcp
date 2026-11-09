"""Shared, non-executing local environment-file loader."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def load_credentials(path: Path) -> None:
    """Load literal allowlisted KEY=value entries from a user-owned mode-0600 file."""
    info = path.stat()
    if stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
        raise ValueError("Credential file must be owned by the current user with mode 0600")
    allowed = {"GEMINI_API_KEY", "TONE_LISTENER_MODEL", "TONE_LISTENER_MAX_SECONDS"}
    entries = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ")
        name, separator, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not separator or name not in allowed or name in entries:
            raise ValueError(f"Invalid or duplicate credential entry on line {number}")
        if value[:1] in ("'", '"') or value[-1:] in ("'", '"'):
            if len(value) < 2 or value[-1] != value[0] or value[0] not in ("'", '"'):
                raise ValueError(f"Unclosed quote in credential entry on line {number}")
            value = value[1:-1]
        if any(character in value for character in ("$", "`", "\x00")):
            raise ValueError(f"Shell expansion is not supported in credential entry on line {number}")
        entries[name] = value
    if not entries.get("GEMINI_API_KEY"):
        raise ValueError("Credential file must contain a nonempty GEMINI_API_KEY")
    os.environ.update(entries)


def load_default_credentials() -> bool:
    """Load the conventional local config only when no API key is already configured."""
    if os.environ.get("GEMINI_API_KEY"):
        return False
    path = Path.home() / ".config" / "tone-listener-mcp" / "env"
    if not path.exists():
        return False
    load_credentials(path)
    return True
