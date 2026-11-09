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
from typing import ClassVar
from unittest.mock import MagicMock

import httpcore2
import numpy as np
import pytest
from google.genai import types
from google.genai.client import Client as RealClient
from test_request_controls import PAYLOAD, USAGE

from tone_listener_mcp import listener
from tone_listener_mcp.audio_io import AudioClip
from tone_listener_mcp.listener import (
    DeadlineExceeded,
    _Deadline,
    _deadline_http_client,
    _DeadlineBackend,
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
        self.drip_uploads_from = 10**9
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
                index = server.uploads
                server.uploads += 1
            name = f"files/a{index}"
            file = {"name": name, "uri": f"https://example.invalid/{name}",
                    "state": "ACTIVE", "mimeType": "audio/wav"}
            # A dripping finalize never trips a per-read inactivity timeout (~3 s in total).
            drip = index >= server.drip_uploads_from
            self.reply(200, json.dumps({"file": file}).encode(),
                       {"x-goog-upload-status": "final"},
                       chunks=12 if drip else 1, interval=0.25 if drip else 0.0)
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
                # "slow" spreads one valid body over nine chunks 0.6 s apart (~4.8 s total).
                self.reply(200, body, chunks=9 if kind == "slow" else 1,
                           interval=0.6 if kind == "slow" else 0.0)
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
    assert 1 <= elapsed < 2.5
    assert result["listening"] is None
    assert result["failure"]["category"] == "deadline"
    assert result["listening_failure_stage"] == "generate"
    assert result["diagnostics"]["elapsed_seconds"] < 2.5
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
    # No read is ever idle long enough for an inactivity timeout, yet the four upload
    # requests together need ~3.6 s against a 1 s deadline.
    gemini.upload_delay = 0.2
    gemini.drip_uploads_from = 1
    started = time.monotonic()
    result = listen_comparison(audio, audio)
    elapsed = time.monotonic() - started
    assert 1 <= elapsed < 2.5
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
    assert 2.9 < elapsed < 4.5
    assert result["failure"]["category"] == "deadline"
    assert_released(gemini)


def test_youtube_retry_is_skipped_without_budget(gemini, monkeypatch):
    monkeypatch.setenv("TONE_LISTENER_REQUEST_DEADLINE_SECONDS", "2")
    gemini.generate = [("503", 0.5), ("json", 0.0)]
    started = time.monotonic()
    result = listen_youtube(YOUTUBE, 0, 1)
    assert time.monotonic() - started < 3.5
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
    assert 0.5 <= elapsed < 2.5
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


class StubBackend:
    """Network backend stand-in: no sockets, scripted outcome per address."""

    def __init__(self, outcomes: dict) -> None:
        self.outcomes = outcomes
        self.attempts: list[tuple[str, float]] = []

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.attempts.append((host, timeout))
        outcome = self.outcomes[host]
        if outcome == "blackhole":
            time.sleep(timeout)
            raise httpcore2.ConnectTimeout("timed out")
        if outcome == "refused":
            raise httpcore2.ConnectError("refused")
        return outcome


@pytest.fixture
def resolve(monkeypatch):
    def install(*addresses: str) -> None:
        def getaddrinfo(host, port, *args, **kwargs):
            assert host == "gemini.test" and kwargs.get("type") == socket.SOCK_STREAM
            return [
                (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (address, port, 0, 0))
                if ":" in address else (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))
                for address in addresses
            ]

        monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)

    return install


def test_blackholed_addresses_share_one_connect_deadline(resolve):
    # create_connection would give each of the eight addresses the full 0.5 s (4 s in total).
    addresses = [f"192.0.2.{n}" for n in range(1, 9)]
    resolve(*addresses)
    stub = StubBackend(dict.fromkeys(addresses, "blackhole"))
    backend = _DeadlineBackend(stub, _Deadline(time.monotonic() + 0.8))
    started = time.monotonic()
    with pytest.raises(DeadlineExceeded):
        backend.connect_tcp("gemini.test", 443, timeout=0.5)
    elapsed = time.monotonic() - started
    assert 0.75 < elapsed < 0.8 + 1.5
    assert [host for host, _ in stub.attempts] == addresses[:2]
    assert stub.attempts[0][1] == 0.5
    assert stub.attempts[1][1] <= 0.3


def test_connect_falls_through_to_a_later_address(resolve):
    resolve("2001:db8::1", "192.0.2.1", "192.0.2.2")
    stream = object()
    stub = StubBackend({"2001:db8::1": "refused", "192.0.2.1": "blackhole", "192.0.2.2": stream})
    backend = _DeadlineBackend(stub, _Deadline(time.monotonic() + 5))
    connected = backend.connect_tcp("gemini.test", 443, timeout=0.2)
    assert isinstance(connected, listener._DeadlineStream) and connected._stream is stream
    assert [host for host, _ in stub.attempts] == ["2001:db8::1", "192.0.2.1", "192.0.2.2"]


def test_connect_reraises_the_last_error_when_every_address_fails(resolve):
    resolve("192.0.2.1", "192.0.2.2")
    stub = StubBackend({"192.0.2.1": "refused", "192.0.2.2": "refused"})
    backend = _DeadlineBackend(stub, _Deadline(time.monotonic() + 5))
    with pytest.raises(httpcore2.ConnectError):
        backend.connect_tcp("gemini.test", 443, timeout=1)
    assert len(stub.attempts) == 2


def test_resolution_failure_keeps_the_backend_error(monkeypatch):
    def fail(*args, **kwargs):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", fail)
    stub = StubBackend({"gemini.test": "refused"})
    backend = _DeadlineBackend(stub, _Deadline(time.monotonic() + 5))
    with pytest.raises(httpcore2.ConnectError):
        backend.connect_tcp("gemini.test", 443, timeout=1)
    assert [host for host, _ in stub.attempts] == ["gemini.test"]


def test_missing_pool_internals_fail_closed_as_local_io(monkeypatch, audio):
    class Changed:
        _transport = SimpleNamespace()
        _mounts: ClassVar[dict] = {}
        closed = False

        def __init__(self, **kwargs) -> None:
            pass

        def close(self) -> None:
            Changed.closed = True

    module = listener.httpx2 if listener.httpx2 is not None else listener.httpx
    monkeypatch.setattr(module, "Client", Changed)
    with pytest.raises(RuntimeError):
        _deadline_http_client(_Deadline(time.monotonic() + 1))
    assert Changed.closed

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    created = MagicMock()
    monkeypatch.setattr("tone_listener_mcp.listener.genai.Client", created)
    result = listen(audio)
    assert result["listening"] is None
    assert result["failure"]["category"] == "local_io"
    assert result["listening_failure_stage"] == "create_client"
    created.assert_not_called()
