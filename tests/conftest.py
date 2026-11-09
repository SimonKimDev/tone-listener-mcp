from unittest.mock import MagicMock

import pytest

from tone_listener_mcp.listener import clear_listening_cache


@pytest.fixture(autouse=True)
def prevent_live_gemini(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for name in (
        "TONE_LISTENER_MODEL",
        "TONE_LISTENER_MAX_SECONDS",
        "TONE_LISTENER_REQUEST_DEADLINE_SECONDS",
        "TONE_LISTENER_MAX_CONCURRENT_REQUESTS",
        "TONE_LISTENER_CACHE_ENTRIES",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "tone_listener_mcp.listener.genai.Client",
        MagicMock(side_effect=AssertionError("Tests must mock all Gemini clients")),
    )
    clear_listening_cache()
    yield
    clear_listening_cache()
