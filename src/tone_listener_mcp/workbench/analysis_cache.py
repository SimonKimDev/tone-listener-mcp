"""Private reuse of successful interval analyses and coalescing of identical in-flight calls.

Only the workbench analysis job uses this module. The opt-in reference benchmark calls the
listener directly so its independent reference and repeat requests are never served from here.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from tone_listener_mcp import audio_io, listener
from tone_listener_mcp.listener import ToneProfile
from tone_listener_mcp.workbench.models import Interval

CACHE_FORMAT = 1
MAX_ENTRIES = 500


@lru_cache(maxsize=1)
def analysis_semantics_sha256() -> str:
    """Fingerprint of everything besides input/model that shapes a profile request.

    Prompt, system instructions, response schema, generation config and audio preparation all
    live in these sources, so any edit to them makes earlier results unreusable.
    """
    digest = hashlib.sha256()
    digest.update(f"workbench-analysis-cache-v{CACHE_FORMAT}\n".encode())
    for module in (listener, audio_io):
        digest.update(Path(module.__file__).read_bytes())
    digest.update(json.dumps(ToneProfile.model_json_schema(), sort_keys=True).encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class AnalysisKey:
    source_kind: Literal["youtube", "audio_file"]
    source_identity: str  # canonical YouTube video ID or SHA-256 of the uploaded file bytes
    start_seconds: float
    end_seconds: float
    model: str
    semantics_sha256: str

    @property
    def digest(self) -> str:
        payload = json.dumps({
            "format": CACHE_FORMAT,
            "source_kind": self.source_kind,
            "source_identity": self.source_identity,
            # float.hex keeps exact bounds; 1.0 and 1.0000000001 are different requests.
            "start": float(self.start_seconds).hex(),
            "end": float(self.end_seconds).hex(),
            "model": self.model,
            "semantics": self.semantics_sha256,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


def make_key(
    source_kind: Literal["youtube", "audio_file"],
    source_identity: str,
    interval: Interval,
    model: str,
) -> AnalysisKey:
    return AnalysisKey(
        source_kind=source_kind,
        source_identity=source_identity,
        start_seconds=interval.start_seconds,
        end_seconds=interval.end_seconds,
        model=model,
        semantics_sha256=analysis_semantics_sha256(),
    )


@dataclass(frozen=True)
class CachedAnalysis:
    profile: dict
    model: str
    analyzed_at: str


class AnalysisCache:
    def __init__(self, root: Path) -> None:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(root, 0o700)
        self.root = root
        self._lock = threading.RLock()
        self._in_flight: dict[str, Future[dict]] = {}

    def _path(self, key: AnalysisKey) -> Path:
        return self.root / f"{key.digest}.json"

    def get(self, key: AnalysisKey) -> CachedAnalysis | None:
        path = self._path(key)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("format") != CACHE_FORMAT or data.get("key") != key.digest:
                return None
            profile = ToneProfile.model_validate(data["profile"]).model_dump(mode="json")
            return CachedAnalysis(profile=profile, model=str(data["model"]), analyzed_at=str(data["analyzed_at"]))
        except (OSError, ValueError, KeyError, TypeError, ValidationError):
            return None

    def put(self, key: AnalysisKey, profile: dict, model: str) -> None:
        validated = ToneProfile.model_validate(profile).model_dump(mode="json")
        payload = json.dumps({
            "format": CACHE_FORMAT,
            "key": key.digest,
            "model": model,
            "analyzed_at": datetime.now(UTC).isoformat(),
            "profile": validated,
        }, sort_keys=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.root, prefix=".entry-", suffix=".tmp", delete=False,
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        os.chmod(temporary, 0o600)
        os.replace(temporary, self._path(key))
        self._prune()

    def clear(self) -> int:
        removed = 0
        for path in self.root.glob("*.json"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    def _prune(self) -> None:
        entries = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime)
        for path in entries[:-MAX_ENTRIES] if len(entries) > MAX_ENTRIES else []:
            path.unlink(missing_ok=True)

    def in_flight(self, key: AnalysisKey) -> bool:
        with self._lock:
            return key.digest in self._in_flight

    def run_shared(
        self,
        key: AnalysisKey,
        request: Callable[[], dict],
        cancelled: threading.Event,
        *,
        may_start: bool,
        use_cache: bool = True,
    ) -> tuple[Literal["cache", "fresh", "shared", "blocked", "cancelled"], dict | CachedAnalysis | None]:
        """Run ``request`` once per key; concurrent callers wait for the same outcome.

        ``may_start`` is checked atomically: a caller without remaining confirmed request budget
        can join existing work but never starts a new provider request ("blocked"). A caller that
        is cancelled while waiting stops waiting ("cancelled"); the shared request keeps running
        for its other consumers. Failures are shared with waiters but are never cached.

        With ``use_cache`` the cache is re-read under the same lock before starting: a leader
        stores its success before leaving the in-flight map, so a caller that just missed both
        the cache and the in-flight entry still reuses that result ("cache") instead of
        starting a duplicate request.
        """
        with self._lock:
            future = self._in_flight.get(key.digest)
            leader = future is None
            if leader:
                if use_cache:
                    cached = self.get(key)
                    if cached is not None:
                        return "cache", cached
                if not may_start:
                    return "blocked", None
                future = Future()
                self._in_flight[key.digest] = future
        if leader:
            try:
                result = request()
            except BaseException as exc:
                with self._lock:
                    self._in_flight.pop(key.digest, None)
                future.set_exception(exc)
                raise
            with self._lock:
                self._in_flight.pop(key.digest, None)
            future.set_result(result)
            return "fresh", result
        while True:
            if cancelled.is_set():
                return "cancelled", None
            try:
                return "shared", future.result(timeout=0.05)
            except TimeoutError:
                if future.done():
                    return "shared", future.result()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
