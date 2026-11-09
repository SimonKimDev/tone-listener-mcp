import threading
import time

from tone_listener_mcp.workbench.jobs import JobManager


def _wait(manager, job_id, status):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        job = manager.get(job_id)
        if job and job.status == status:
            return job
        time.sleep(0.005)
    raise AssertionError(f"job did not reach {status}")


def test_job_manager_reports_progress_and_terminal_result():
    manager = JobManager(workers=1, max_active=2)
    try:
        started = threading.Event()

        def work(progress, cancelled):
            started.set()
            progress(0.4, "Rendering sample.")
            return {"value": 7}

        job = manager.submit("preview", work)
        assert started.wait(1)
        complete = _wait(manager, job.id, "completed")
        assert complete.progress == 1
        assert complete.result == {"value": 7}
    finally:
        manager.close()


def test_job_manager_cancels_running_work_and_runs_discard_cleanup():
    manager = JobManager(workers=1, max_active=2)
    try:
        started = threading.Event()
        release = threading.Event()
        cleaned = threading.Event()

        def work(progress, cancelled):
            started.set()
            assert release.wait(1)
            return {
                "value": "must not publish",
                "_cancel_cleanup": cleaned.set,
            }

        job = manager.submit("preview", work)
        assert started.wait(1)
        manager.cancel(job.id)
        release.set()
        cancelled = _wait(manager, job.id, "cancelled")
        assert cancelled.result is None
        assert cleaned.is_set()
    finally:
        manager.close()


def test_job_manager_cancels_a_job_before_its_worker_starts():
    manager = JobManager(workers=1, max_active=2)
    try:
        started = threading.Event()
        release = threading.Event()

        def blocker(progress, cancelled):
            started.set()
            assert release.wait(1)
            return {}

        first = manager.submit("analysis", blocker)
        assert started.wait(1)
        second = manager.submit("analysis", lambda progress, cancelled: {})
        result = manager.cancel(second.id)
        assert result is not None
        assert result.status == "cancelled"
        release.set()
        _wait(manager, first.id, "completed")
    finally:
        manager.close()


def test_job_manager_sanitizes_unexpected_exceptions():
    manager = JobManager(workers=1, max_active=1)
    try:
        job = manager.submit(
            "analysis",
            lambda progress, cancelled: (_ for _ in ()).throw(
                RuntimeError("/private/user/path")
            ),
        )
        failed = _wait(manager, job.id, "failed")
        assert failed.error == "Operation failed (RuntimeError); details were withheld."
        assert "/private/user/path" not in failed.error
    finally:
        manager.close()
