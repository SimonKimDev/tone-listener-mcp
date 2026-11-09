"""Total request deadline over the real google-genai HTTP transport (issue #4).

A loopback HTTP server stands in for Gemini, so the SDK, httpx2 and sockets all run for real
without credentials or provider calls.
"""

import json
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from google.genai import types
from google.genai.client import Client as RealClient
from test_request_controls import PAYLOAD, USAGE

from tone_listener_mcp import listener
from tone_listener_mcp.audio_io import AudioClip
from tone_listener_mcp.listener import (
    _inference_slots,
    listen,
    listen_comparison,
    listen_youtube,
)

YOUTUBE = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


class Disconnected(Exception):
    pass


class FakeGemini(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.lock = threading.Lock()
        self.requests: list[tuple[str, str]] = []
        self.active = 0
        self.uploads = 0
        self.stopping = threading.Event()
        self.slot_free_during_requests: list[bool] = []
        self.slot_limit = 1
        # Behaviour hooks, replaced per test.
        self.generate = [("json", 0.0)]
        self.upload_delay = 0.0
        self.slow_delete: str | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/"

    def wait_idle(self, timeout: float = 2.0) -> bool:
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            with self.lock:
                if self.active == 0:
                    return True
            time.sleep(0.02)
        return False

    def count(self, fragment: str) -> int:
        with self.lock:
            return sum(fragment in path for _, path in self.requests)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: FakeGemini

    def log_message(self, *args) -> None:
        pass

    def pause(self, seconds: float) -> None:
        """Sleep while watching for the client closing the connection."""
        until = time.monotonic() + seconds
        while time.monotonic() < until and not self.server.stopping.is_set():
            readable, _, _ = select.select([self.connection], [], [], 0.02)
            if readable and self.connection.recv(1, socket.MSG_PEEK) == b"":
                raise Disconnected

    def reply(self, status: int, body: bytes, headers: dict | None = None,
              chunks: int = 1, interval: float = 0.0, length: int | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(length if length is not None else len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        size = max(1, -(-len(body) // chunks))
        for offset in range(0, len(body), size):
            self.wfile.write(body[offset:offset + size])
            self.wfile.flush()
            if offset + size < len(body):
                self.pause(interval)

    def drip_forever(self) -> None:
        self.reply(200, b"", length=10**6)
        while not self.server.stopping.is_set():
            self.pause(0.2)
            self.wfile.write(b" ")
            self.wfile.flush()

    def handle_one(self) -> None:
        server = self.server
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        with server.lock:
            server.requests.append((self.command, self.path))
        slots = _inference_slots(server.slot_limit)
        free = slots.acquire(blocking=False)
        if free:
            slots.release()
        server.slot_free_during_requests.append(free)
        if self.path.startswith("/upload/"):
            self.pause(server.upload_delay)
            self.reply(200, b"{}", {"x-goog-upload-url": f"{server.url}session"})
        elif self.path.startswith("/session"):
            self.pause(server.upload_delay)
            with server.lock:
                name = f"files/a{server.uploads}"
                server.uploads += 1
            file = {"name": name, "uri": f"https://example.invalid/{name}",
                    "state": "ACTIVE", "mimeType": "audio/wav"}
            self.reply(200, json.dumps({"file": file}).encode(),
                       {"x-goog-upload-status": "final"})
        elif "generateContent" in self.path:
            behaviour = server.generate.pop(0) if len(server.generate) > 1 else server.generate[0]
            kind, delay = behaviour
            self.pause(delay)
            if kind == "503":
                self.reply(503, json.dumps({"error": {"code": 503, "message": "busy"}}).encode())
            elif kind == "drip":
                self.drip_forever()
            else:
                body = json.dumps({
                    "candidates": [{"content": {"role": "model",
                                                "parts": [{"text": json.dumps(PAYLOAD)}]}}],
                    "usageMetadata": {"promptTokenCount": 1, "totalTokenCount": 2},
                }).encode()
                # "slow" spreads one valid body over nine chunks 0.3 s apart (~2.4 s total).
                self.reply(200, body, chunks=9 if kind == "slow" else 1,
                           interval=0.3 if kind == "slow" else 0.0)
        elif self.command == "DELETE":
            if server.slow_delete is not None and self.path.endswith(server.slow_delete):
                self.drip_forever()
            else:
                self.reply(200, b"{}")
        else:
            self.reply(404, b"{}")

    def handle_request(self) -> None:
        with self.server.lock:
            self.server.active += 1
        try:
            self.handle_one()
        except (Disconnected, BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        finally:
            with self.server.lock:
                self.server.active -= 1

    do_GET = do_POST = do_DELETE = handle_request


@pytest.fixture
def gemini(monkeypatch):
    server = FakeGemini()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy",
                 "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "loopback-only")
    monkeypatch.setenv("TONE_LISTENER_MAX_CONCURRENT_REQUESTS", "1")
    monkeypatch.setenv("TONE_LISTENER_REQUEST_DEADLINE_SECONDS", "1")

    def client(**kwargs):
        options = kwargs["http_options"]
        assert options.httpx_client is not None, "deadline-enforcing HTTP client must be used"
        options.base_url = server.url
        return RealClient(**kwargs)

    monkeypatch.setattr("tone_listener_mcp.listener.genai.Client", client)
    yield server
    server.stopping.set()
    server.shutdown()
    server.server_close()


@pytest.fixture
def audio():
    return AudioClip(np.ones((1, 16000), dtype=np.float32) * 0.1, 16000, "synthetic")


def assert_released(server: FakeGemini) -> None:
    assert server.wait_idle(), "a provider request outlived the listening call"
    slots = _inference_slots(1)
    assert slots.acquire(blocking=False), "inference slot was not released"
    slots.release()


def test_slow_response_body_hits_total_deadline_and_is_not_cached(gemini):
    gemini.generate = [("slow", 0.0)]
    started = time.monotonic()
    result = listen_youtube(YOUTUBE, 0, 1)
    elapsed = time.monotonic() - started
    assert elapsed < 1.5
    assert result["listening"] is None
    assert result["failure"]["category"] == "deadline"
    assert result["listening_failure_stage"] == "generate"
    assert result["diagnostics"]["elapsed_seconds"] < 1.5
    assert gemini.count("generateContent") == 1
    assert gemini.count("upload") == 0 and gemini.count("DELETE") == 0
    assert gemini.slot_free_during_requests == [False]
    assert_released(gemini)
    gemini.generate = [("json", 0.0)]
    retried = listen_youtube(YOUTUBE, 0, 1)
    assert retried["failure"] is None and retried["listening"] is not None
    assert retried["diagnostics"]["cache"] == "miss"
    assert gemini.count("generateContent") == 2


def test_cumulative_upload_requests_hit_deadline_and_cleanup_runs(gemini, audio):
    # Each request answers well within any per-call timeout, but four of them do not fit.
    gemini.upload_delay = 0.3
    started = time.monotonic()
    result = listen_comparison(audio, audio)
    elapsed = time.monotonic() - started
    assert elapsed < 1.5
    assert result["failure"]["category"] == "deadline"
    assert result["listening_failure_stage"] == "upload"
    assert gemini.count("generateContent") == 0
    deleted = [path for method, path in gemini.requests if method == "DELETE"]
    assert deleted == ["/v1beta/files/a0"] or deleted == ["/v1/files/a0"]
    assert result["cleanup_warnings"] == []
    assert_released(gemini)


def test_youtube_retries_share_the_total_deadline(gemini, monkeypatch):
    monkeypatch.setenv("TONE_LISTENER_REQUEST_DEADLINE_SECONDS", "3")
    gemini.generate = [("503", 0.1), ("drip", 0.0)]
    started = time.monotonic()
    result = listen_youtube(YOUTUBE, 0, 1)
    elapsed = time.monotonic() - started
    assert gemini.count("generateContent") == 2
    assert 2.9 < elapsed < 3.5
    assert result["failure"]["category"] == "deadline"
    assert_released(gemini)


def test_youtube_retry_is_skipped_without_budget(gemini, monkeypatch):
    monkeypatch.setenv("TONE_LISTENER_REQUEST_DEADLINE_SECONDS", "2")
    gemini.generate = [("503", 0.5), ("json", 0.0)]
    started = time.monotonic()
    result = listen_youtube(YOUTUBE, 0, 1)
    assert time.monotonic() - started < 1.5
    assert gemini.count("generateContent") == 1
    assert result["failure"]["category"] == "capacity"
    assert_released(gemini)


def test_cleanup_has_its_own_bounded_budget(gemini, audio, monkeypatch):
    monkeypatch.setenv("TONE_LISTENER_REQUEST_DEADLINE_SECONDS", "5")
    monkeypatch.setattr(listener, "CLEANUP_TIMEOUT_SECONDS", 0.5)
    gemini.generate = [("503", 0.0)]
    gemini.slow_delete = "files/a0"
    started = time.monotonic()
    result = listen_comparison(audio, audio)
    elapsed = time.monotonic() - started
    assert result["failure"]["category"] == "capacity"
    deleted = [path.rsplit("/", 1)[1] for method, path in gemini.requests if method == "DELETE"]
    assert deleted == ["a0", "a1"]
    assert result["cleanup_warnings"] == ["Remote upload cleanup failed (DeadlineExceeded)"]
    assert elapsed < 1.5
    assert_released(gemini)


def test_success_completing_after_deadline_is_rejected_and_not_cached(monkeypatch, audio):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("TONE_LISTENER_REQUEST_DEADLINE_SECONDS", "1")
    mocked = MagicMock()
    mocked.__enter__.return_value = mocked
    mocked.files.upload.return_value = SimpleNamespace(
        name="files/test", uri="https://example.invalid/audio", state=types.FileState.ACTIVE
    )

    def late(**kwargs):
        time.sleep(1.1)
        return SimpleNamespace(text=json.dumps(PAYLOAD), usage_metadata=USAGE)

    mocked.models.generate_content.side_effect = late
    monkeypatch.setattr("tone_listener_mcp.listener.genai.Client", lambda **kwargs: mocked)
    result = listen(audio)
    assert result["listening"] is None
    assert result["failure"]["category"] == "deadline"
    mocked.files.delete.assert_called_once()
    mocked.models.generate_content.side_effect = None
    mocked.models.generate_content.return_value = SimpleNamespace(
        text=json.dumps(PAYLOAD), usage_metadata=USAGE
    )
    assert listen(audio)["diagnostics"]["cache"] == "miss"
