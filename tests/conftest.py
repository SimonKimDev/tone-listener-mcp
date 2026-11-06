from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def prevent_live_gemini(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        "tone_listener_mcp.listener.genai.Client",
        MagicMock(side_effect=AssertionError("Tests must mock all Gemini clients")),
    )
