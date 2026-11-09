import threading
from concurrent.futures import Future

import pytest

from tests.test_workbench_recipes import profile
from tone_listener_mcp.workbench.analysis_cache import AnalysisCache, AnalysisKey
from tone_listener_mcp.workbench.evaluation import compare_profiles
from tone_listener_mcp.workbench.models import SourceRef
from tone_listener_mcp.workbench.sources import automatic_intervals
from tone_listener_mcp.workbench.storage import PreviewCache


def test_v1_evaluator_accepts_matching_profiles_and_rejects_large_differences():
    assert compare_profiles(profile(), profile())["outcome"] == "pass"
    different = profile(gain="high-gain", brightness="dark", mids="scooped")
    assert compare_profiles(profile(), different)["outcome"] == "fail"


@pytest.mark.parametrize("duration", [31.0, 40.0, 60.1])
def test_automatic_intervals_pass_source_overlap_validation(duration):
    source = SourceRef(
        kind="audio_file",
        display_name="Uploaded audio",
        duration_seconds=duration,
        selected_intervals=automatic_intervals(duration),
    )

    assert len(source.selected_intervals) == 3


def test_preview_cache_pruning_accounts_for_file_sizes_before_removal(tmp_path):
    cache_dir = tmp_path / "previews"
    cache_dir.mkdir()
    preview_id = "00000000-0000-4000-8000-000000000001"
    audio = cache_dir / f"{preview_id}.wav"
    sidecar = cache_dir / f"{preview_id}.json"
    audio.write_bytes(b"audio")
    sidecar.write_text("{}")

    PreviewCache(cache_dir, max_bytes=1)

    assert not audio.exists()
    assert not sidecar.exists()


def test_shared_analysis_propagates_provider_timeout_errors(tmp_path):
    cache = AnalysisCache(tmp_path / "analysis")
    key = AnalysisKey(
        source_kind="youtube",
        source_identity="401iS5McuoU",
        start_seconds=0,
        end_seconds=10,
        model="offline-test",
        semantics_sha256="test",
    )
    future = Future()
    future.set_exception(TimeoutError("provider timed out"))
    cache._in_flight[key.digest] = future

    cancelled = threading.Event()
    timer = threading.Timer(0.05, cancelled.set)
    timer.daemon = True
    timer.start()

    with pytest.raises(TimeoutError, match="provider timed out"):
        cache.run_shared(key, dict, cancelled, may_start=False)
