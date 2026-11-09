"""Bounded in-process jobs with explicit progress and cooperative cancellation."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

from tone_listener_mcp.workbench.models import JobStatus


class JobQueueFull(RuntimeError):
    """The bounded local job queue is full."""


class JobFailure(RuntimeError):
    """A safe, user-facing job failure."""


@dataclass
class _Job:
    id: str
    kind: Literal["analysis", "preview", "refinement"]
    status: Literal["queued", "running", "completed", "partial", "failed", "cancelled", "stale"] = "queued"
    progress: float = 0.0
    message: str = "Waiting for a local worker."
    result: dict[str, object] | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: Future | None = None
    on_abandon: Callable[[], None] | None = None
    updated_at: float = field(default_factory=time.time)

    def public(self) -> JobStatus:
        return JobStatus(
            id=self.id,
            kind=self.kind,
            status=self.status,
            progress=self.progress,
            message=self.message,
            result=self.result,
            error=self.error,
        )


JobFunction = Callable[
    [Callable[[float, str], None], threading.Event],
    dict[str, object],
]


class JobManager:
    def __init__(self, workers: int = 2, max_active: int = 4) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="tone-workbench",
        )
        self._max_active = max_active
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.RLock()

    def submit(
        self,
        kind: Literal["analysis", "preview", "refinement"],
        work: JobFunction,
        on_abandon: Callable[[], None] | None = None,
    ) -> JobStatus:
        """``on_abandon`` runs once if the job is cancelled or shut down before ``work`` ever starts."""
        with self._lock:
            active = sum(job.status in ("queued", "running") for job in self._jobs.values())
            if active >= self._max_active:
                raise JobQueueFull("Local work queue is full; wait for an active job to finish.")
            job = _Job(id=str(uuid.uuid4()), kind=kind)
            job.on_abandon = on_abandon
            self._jobs[job.id] = job
            job.future = self._executor.submit(self._run, job, work)
            self._trim_history()
            return job.public()

    def get(self, identifier: str) -> JobStatus | None:
        with self._lock:
            job = self._jobs.get(identifier)
            return job.public() if job else None

    def cancel(self, identifier: str) -> JobStatus | None:
        with self._lock:
            job = self._jobs.get(identifier)
            if not job or job.status not in ("queued", "running"):
                return job.public() if job else None
            job.cancel_event.set()
            abandoned = None
            if job.status == "queued" and job.future and job.future.cancel():
                job.status = "cancelled"
                job.message = "Cancelled before a worker started."
                abandoned = self._take_abandon(job)
            else:
                job.message = "Cancellation requested; the active provider call will finish before the job stops."
            job.updated_at = time.time()
            public = job.public()
        if abandoned:
            abandoned()
        return public

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        abandoned = []
        with self._lock:
            for job in self._jobs.values():
                if job.status == "queued" and job.future is not None and job.future.cancelled():
                    job.status = "cancelled"
                    job.message = "Cancelled by shutdown before a worker started."
                    callback = self._take_abandon(job)
                    if callback:
                        abandoned.append(callback)
        for callback in abandoned:
            callback()

    @staticmethod
    def _take_abandon(job: _Job) -> Callable[[], None] | None:
        callback, job.on_abandon = job.on_abandon, None
        return callback

    def _run(self, job: _Job, work: JobFunction) -> None:
        with self._lock:
            if job.cancel_event.is_set():
                job.status = "cancelled"
                job.message = "Cancelled before a worker started."
                job.updated_at = time.time()
                abandoned = self._take_abandon(job)
            else:
                abandoned = None
                # Once work starts it owns its cleanup.
                job.on_abandon = None
                job.status = "running"
                job.message = "Starting."
                job.updated_at = time.time()
        if job.status == "cancelled":
            if abandoned:
                abandoned()
            return

        def update(progress: float, message: str) -> None:
            with self._lock:
                if job.status == "running":
                    job.progress = min(1.0, max(0.0, progress))
                    job.message = message[:200]
                    job.updated_at = time.time()

        try:
            result = work(update, job.cancel_event)
            result_status = result.pop("_job_status", "completed")
            if result_status not in ("completed", "partial", "stale"):
                raise ValueError("Worker returned an unsupported terminal status.")
            with self._lock:
                cleanup = result.pop("_cancel_cleanup", None)
                if job.cancel_event.is_set():
                    if callable(cleanup):
                        cleanup()
                    job.status = "cancelled"
                    job.result = None
                    job.message = "Cancelled; no stale result was published."
                    job.updated_at = time.time()
                    return
                job.status = result_status
                job.progress = 1.0
                job.message = {
                    "completed": "Finished.",
                    "partial": "Finished with one or more explicit limitations.",
                    "stale": "Result discarded because the project changed during rendering.",
                }[result_status]
                job.result = result
                job.updated_at = time.time()
        except Exception as exc:  # noqa: BLE001 - mark every worker failure terminal instead of stranding it.
            if job.cancel_event.is_set():
                with self._lock:
                    job.status = "cancelled"
                    job.error = None
                    job.message = "Cancelled; no stale result was published."
                    job.updated_at = time.time()
                return
            error = str(exc).strip() if isinstance(exc, JobFailure) else ""
            if not error:
                error = f"Operation failed ({type(exc).__name__}); details were withheld."
            from .privacy import redact_paths
            error = redact_paths(error)[:500]
            with self._lock:
                job.status = "failed"
                job.error = error
                job.message = "Failed; no result was published."
                job.updated_at = time.time()

    def _trim_history(self) -> None:
        completed = sorted(
            (
                job for job in self._jobs.values()
                if job.status not in ("queued", "running")
            ),
            key=lambda job: job.updated_at,
        )
        for job in completed[:-100]:
            self._jobs.pop(job.id, None)
