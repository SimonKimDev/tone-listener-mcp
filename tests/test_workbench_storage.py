import hashlib
import os
import stat

import numpy as np
import pytest
import soundfile as sf

from tone_listener_mcp.workbench.models import (
    AnalysisSegment,
    ChainDraft,
    ChainNode,
    Interval,
    PreviewSnapshot,
    SourceRef,
    TargetBinding,
    ToneIntent,
    ToneProject,
)
from tone_listener_mcp.workbench.storage import LocalStorage, PreviewCache


def _project():
    return ToneProject(
        name="Local test draft",
        source=SourceRef(
            kind="youtube",
            url="https://www.youtube.com/watch?v=401iS5McuoU",
            display_name="YouTube video 401iS5McuoU",
            selected_intervals=[Interval(start_seconds=0, end_seconds=10)],
        ),
        analysis=[
            AnalysisSegment(
                interval=Interval(start_seconds=0, end_seconds=10),
                model="offline-test",
                unavailable="No provider requested in this test.",
            )
        ],
        intent=ToneIntent(summary="Unanalyzed local test project."),
        chain=ChainDraft(
            nodes=[
                ChainNode(id="input", kind="input", title="Input"),
                ChainNode(id="gain", kind="gain", title="Gain", params={"gainDb": 0}),
                ChainNode(id="output", kind="output", title="Output"),
            ]
        ),
        target=TargetBinding(adapter_id="fake", engine_version="test"),
        created_at="2025-01-01T00:00:00+00:00",
        updated_at="2025-01-01T00:00:00+00:00",
    )


def test_local_project_is_saved_atomically_with_private_permissions(tmp_path):
    storage = LocalStorage(tmp_path / "workbench")
    project = _project()
    storage.save_project(project)
    path = storage.projects / f"{project.id}.json"

    assert storage.load_project(project.id) == project
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(storage.root.stat().st_mode) == 0o700
    assert os.access(path, os.R_OK)


def test_project_compare_and_save_cannot_overwrite_concurrent_revision(tmp_path):
    import pytest

    from tone_listener_mcp.workbench.storage import ProjectConflict

    storage = LocalStorage(tmp_path)
    original = _project()
    storage.save_project(original)
    concurrent = original.model_copy(update={"revision": 2, "name": "Concurrent edit"})
    storage.save_project(concurrent, expected_revision=1)
    stale = original.model_copy(update={"revision": 2, "name": "Stale edit"})
    with pytest.raises(ProjectConflict):
        storage.save_project(stale, expected_revision=1)
    assert storage.load_project(original.id).name == "Concurrent edit"


def test_old_persisted_metrics_paths_are_sanitized_on_load(tmp_path):
    import json

    storage = LocalStorage(tmp_path)
    original = _project()
    data = original.model_dump(mode="json")
    data["analysis"][0]["metrics"] = {"clip": {"source": "/private/upload.bin", "channels": 1}}
    (storage.projects / f"{original.id}.json").write_text(json.dumps(data))
    loaded = storage.load_project(original.id)
    assert loaded.analysis[0].metrics["clip"] == {"source": "Selected audio interval", "channels": 1}


def test_upload_tokens_resolve_only_app_owned_files_and_release(tmp_path):
    storage = LocalStorage(tmp_path / "workbench")
    upload_id, path = storage.new_upload_path()
    path.write_bytes(b"private local audio")
    uploaded = storage.register_upload(
        upload_id, path, "riff.wav", 2.0, 44_100, 1
    )

    assert storage.get_upload(upload_id) == uploaded
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    storage.release_upload(upload_id)
    assert storage.get_upload(upload_id) is None
    assert not path.exists()


@pytest.mark.parametrize("sample", ["riff-1", "lead-2"])
def test_preview_cache_checks_audio_hash_and_round_trips_metadata(tmp_path, sample):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(mode=0o700)
    cache = PreviewCache(cache_dir)
    samples = np.zeros((128, 2), dtype=np.float32)
    audio_path = cache_dir / "00000000-0000-4000-8000-000000000001.wav"
    sf.write(audio_path, samples, 44_100, subtype="PCM_24")
    digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    preview = PreviewSnapshot(
        id="00000000-0000-4000-8000-000000000001",
        project_id=_project().id,
        project_revision=1,
        chain_revision=1,
        chain_sha256="a" * 64,
        sample_name=sample,
        source_sample_sha256="b" * 64,
        engine_version="test",
        sample_rate=44_100,
        dsp_conditions={"sample_rate": 44_100},
        audio_sha256=digest,
        duration_seconds=128 / 44_100,
        rms=0.1,
        peak=0.1,
        created_at="2025-01-01T00:00:00+00:00",
    )
    cached = cache.add(preview, sample)

    assert cached.snapshot == preview
    assert cache.get(preview.id) == cached
    assert cache.for_project(preview.project_id) == [cached]

    audio_path.write_bytes(b"changed")
    assert cache.get(preview.id) is None
