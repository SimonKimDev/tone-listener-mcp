"""Total request deadline over the real google-genai HTTP transport (issue #4).

A loopback HTTP server stands in for Gemini, so the SDK, httpx2 and sockets all run for real
without credentials or provider calls.
"""

import json
import math
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

# Loopback tests use the real clock. On a coarse clock (Windows Python 3.12 ticks at 1/64 s)
# a socket wait clamped to the budget can end while the clock still reads up to one tick
# short of the deadline, so lower bounds allow a tick of slack; upper bounds allow a slow
# runner 1.5 s over the deadline.
TICK_SLACK = 0.1

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
    assert 1 - TICK_SLACK <= elapsed < 2.5
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
    # requests together need ~3.4 s against a 1 s deadline. The first file is ready after
    # ~0.2 s, leaving a slow runner most of the budget to finish it.
    gemini.upload_delay = 0.1
    gemini.drip_uploads_from = 1
    started = time.monotonic()
    result = listen_comparison(audio, audio)
    elapsed = time.monotonic() - started
    assert 1 - TICK_SLACK <= elapsed < 2.5
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
    assert 3 - TICK_SLACK <= elapsed < 4.5
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
    # The drip never trips the 0.5 s per-read timeout, so only the cleanup budget ends the
    # delete. A lagging clock can surface the clamped socket timeout instead of the deadline.
    assert result["cleanup_warnings"] in (
        ["Remote upload cleanup failed (DeadlineExceeded)"],
        ["Remote upload cleanup failed (ReadTimeout)"],
    )
    assert 0.5 - TICK_SLACK <= elapsed < 2.5
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


class FakeClock:
    """Monotonic clock for the listener module only; reads can be quantized to ``tick``.

    Windows Python 3.12 ``time.monotonic`` advances in 1/64 s steps while socket waits are
    precise, so a read just after a clamped wait can still lag behind the deadline.
    """

    def __init__(self, tick: float | None = None, offset: float = 0.0,
                 origin: float = 1000.0) -> None:
        self.tick = tick
        self.now = origin + offset * (tick or 0.0)

    def monotonic(self) -> float:
        if self.tick is None:
            return self.now
        return math.floor(self.now / self.tick) * self.tick

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    def install(tick: float | None = None, offset: float = 0.0,
                origin: float = 1000.0) -> FakeClock:
        fake = FakeClock(tick, offset, origin)
        monkeypatch.setattr(listener, "time", SimpleNamespace(
            monotonic=fake.monotonic, sleep=fake.sleep
        ))
        return fake

    return install


class StubBackend:
    """Network backend stand-in: no sockets, scripted outcome per address."""

    def __init__(self, outcomes: dict, clock: FakeClock | None = None) -> None:
        self.outcomes = outcomes
        self.clock = clock
        # (host, timeout handed to the backend, clock reading when the attempt started)
        self.attempts: list[tuple[str, float, float | None]] = []

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.attempts.append((host, timeout, self.clock.monotonic() if self.clock else None))
        outcome = self.outcomes[host]
        if outcome == "blackhole":
            # A black-holed connect waits out its whole timeout.
            self.clock.sleep(timeout)
            raise httpcore2.ConnectTimeout("timed out")
        if outcome == "refused":
            raise httpcore2.ConnectError("refused")
        return outcome

    @property
    def hosts(self) -> list[str]:
        return [host for host, _, _ in self.attempts]

    @property
    def timeouts(self) -> list[float]:
        return [timeout for _, timeout, _ in self.attempts]


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


BLACKHOLED = [f"192.0.2.{n}" for n in range(1, 9)]
CONNECT_TIMEOUT = 0.5
CONNECT_BUDGET = 0.8
# Budgets are float differences (0.8 - 0.5 == 0.30000000000000004, and larger at big clock
# origins), so every comparison with a decimal literal allows this much rounding.
EPS = 1e-9
# 1/64 s is the Windows Python 3.12 monotonic tick; offsets place the start inside a tick.
# Origin 100000 reproduces the reviewed Windows failure (second timeout 0.3000000000029104).
QUANTIZED_CLOCKS = [
    pytest.param(1 / 64, offset, origin, id=f"tick64-{offset}-at{origin:g}")
    for origin in (1000.0, 100000.0) for offset in (0, 0.25, 0.5, 0.75)
] + [
    pytest.param(0.05, offset, 1000.0, id=f"tick50ms-{offset}") for offset in (0, 0.5)
]


@pytest.mark.parametrize("origin", [1000.0, 100000.0])
def test_blackholed_addresses_share_one_connect_deadline(resolve, clock, origin):
    # create_connection would give each of the eight addresses the full 0.5 s (4 s in total).
    fake = clock(origin=origin)
    resolve(*BLACKHOLED)
    stub = StubBackend(dict.fromkeys(BLACKHOLED, "blackhole"), fake)
    started = fake.now
    backend = _DeadlineBackend(stub, _Deadline(fake.monotonic() + CONNECT_BUDGET))
    with pytest.raises(DeadlineExceeded):
        backend.connect_tcp("gemini.test", 443, timeout=CONNECT_TIMEOUT)
    assert stub.hosts == BLACKHOLED[:2]
    assert stub.timeouts == [
        CONNECT_TIMEOUT, pytest.approx(CONNECT_BUDGET - CONNECT_TIMEOUT, abs=EPS)
    ]
    assert fake.now - started == pytest.approx(CONNECT_BUDGET, abs=EPS)


@pytest.mark.parametrize(("tick", "offset", "origin"), QUANTIZED_CLOCKS)
def test_blackholed_addresses_stay_within_deadline_on_a_coarse_clock(resolve, clock, tick,
                                                                   offset, origin):
    # A lagging clock may allow extra tiny attempts, but never more than one tick of budget.
    fake = clock(tick, offset, origin)
    resolve(*BLACKHOLED)
    stub = StubBackend(dict.fromkeys(BLACKHOLED, "blackhole"), fake)
    deadline = _Deadline(fake.monotonic() + CONNECT_BUDGET)
    with pytest.raises(DeadlineExceeded):
        _DeadlineBackend(stub, deadline).connect_tcp("gemini.test", 443, timeout=CONNECT_TIMEOUT)
    assert len(stub.attempts) >= 2
    assert stub.hosts == BLACKHOLED[:len(stub.attempts)]
    assert stub.timeouts[0] == CONNECT_TIMEOUT
    for _, timeout, read in stub.attempts:
        # Never more than the budget the clock showed when the attempt started.
        assert 0 < timeout <= deadline.at - read + EPS
    assert sum(stub.timeouts) <= CONNECT_BUDGET + tick + EPS


@pytest.mark.parametrize(("tick", "offset", "origin"),
                         [pytest.param(None, 0, 1000.0, id="ideal"), *QUANTIZED_CLOCKS])
def test_connect_falls_through_to_a_later_address(resolve, clock, tick, offset, origin):
    fake = clock(tick, offset, origin)
    resolve("2001:db8::1", "192.0.2.1", "192.0.2.2")
    stream = object()
    stub = StubBackend(
        {"2001:db8::1": "refused", "192.0.2.1": "blackhole", "192.0.2.2": stream}, fake
    )
    backend = _DeadlineBackend(stub, _Deadline(fake.monotonic() + CONNECT_BUDGET))
    connected = backend.connect_tcp("gemini.test", 443, timeout=CONNECT_TIMEOUT)
    assert isinstance(connected, listener._DeadlineStream) and connected._stream is stream
    assert stub.hosts == ["2001:db8::1", "192.0.2.1", "192.0.2.2"]
    assert stub.timeouts[:2] == [CONNECT_TIMEOUT, CONNECT_TIMEOUT]
    assert 0 < stub.timeouts[2] <= CONNECT_BUDGET - CONNECT_TIMEOUT + (tick or 0) + EPS


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
    assert stub.hosts == ["gemini.test"]


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


@pytest.fixture
def mocked_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    mocked = MagicMock()
    mocked.__enter__.return_value = mocked
    mocked.files.upload.return_value = SimpleNamespace(
        name="files/test", uri="https://example.invalid/audio", state=types.FileState.ACTIVE
    )
    mocked.models.generate_content.return_value = SimpleNamespace(
        text=json.dumps(PAYLOAD), usage_metadata=USAGE
    )
    monkeypatch.setattr("tone_listener_mcp.listener.genai.Client", lambda **kwargs: mocked)
    return mocked


def slow_preparation(monkeypatch, fake: FakeClock) -> dict:
    """Make audio preparation take ``delay["seconds"]`` of fake time per clip."""
    delay = {"seconds": 0.0}
    prepare = listener.prepare_upload

    def slow(clip):
        fake.sleep(delay["seconds"])
        return prepare(clip)

    monkeypatch.setattr(listener, "prepare_upload", slow)
    return delay


def test_cache_hit_is_not_returned_after_the_deadline(monkeypatch, clock, audio, mocked_gemini):
    fake = clock()
    delay = slow_preparation(monkeypatch, fake)
    monkeypatch.setenv("TONE_LISTENER_REQUEST_DEADLINE_SECONDS", "3")
    warm = listen(audio)
    assert warm["failure"] is None and warm["diagnostics"]["cache"] == "miss"
    monkeypatch.setenv("TONE_LISTENER_REQUEST_DEADLINE_SECONDS", "1")

    delay["seconds"] = 0.6
    hit = listen(audio)
    assert hit["failure"] is None and hit["listening"] == warm["listening"]
    assert hit["diagnostics"]["cache"] == "hit"

    delay["seconds"] = 1.1
    rejected = listen(audio)
    assert rejected["listening"] is None
    assert rejected["failure"]["category"] == "deadline"
    assert rejected["listening_failure_stage"] == "queue"
    assert rejected["diagnostics"]["cache"] == "hit"
    assert rejected["diagnostics"]["usage"] is None
    assert rejected["diagnostics"]["elapsed_seconds"] >= 1

    delay["seconds"] = 0.0
    served = listen(audio)
    assert served["failure"] is None and served["listening"] == warm["listening"]
    assert served["diagnostics"]["cache"] == "hit"
    assert mocked_gemini.models.generate_content.call_count == 1

    # A cold cache with the same spent budget fails identically, without any provider call.
    listener.clear_listening_cache()
    delay["seconds"] = 1.1
    cold = listen(audio)
    assert cold["diagnostics"]["cache"] == "miss"
    for field in ("listening", "failure", "listening_unavailable", "listening_failure_stage"):
        assert cold[field] == rejected[field]
    assert mocked_gemini.files.upload.call_count == 1
    assert mocked_gemini.models.generate_content.call_count == 1
