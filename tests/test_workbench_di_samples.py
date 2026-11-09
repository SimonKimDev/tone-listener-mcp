import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from tone_listener_mcp.workbench.models import DEFAULT_DI_SAMPLES, DI_SAMPLES, PreviewSnapshot
from tone_listener_mcp.workbench.renderer import MAX_RENDER_SECONDS, SAMPLE_RATE

DI_ROOT = Path(__file__).resolve().parents[1] / "src" / "tone_listener_mcp" / "workbench" / "assets" / "di"


def _provenance() -> dict:
    return json.loads((DI_ROOT / "provenance.json").read_text())


def test_riffs_stay_default_and_original_provenance_is_unchanged() -> None:
    assert DEFAULT_DI_SAMPLES == ("riff-1", "riff-2") and DI_SAMPLES[:2] == DEFAULT_DI_SAMPLES
    provenance = _provenance()
    assert provenance["license"] == "CC0-1.0"
    assert [item["file"] for item in provenance["samples"]] == ["riff-1.wav", "riff-2.wav"]


@pytest.mark.parametrize("name", DI_SAMPLES)
def test_every_bundled_di_is_renderable_and_hash_recorded(name: str) -> None:
    path = DI_ROOT / f"{name}.wav"
    records = {item["file"]: item for item in _provenance()["samples"]}
    for source in _provenance().get("additional_sources", []):
        records.update({item["file"]: item for item in source["samples"]})
    assert hashlib.sha256(path.read_bytes()).hexdigest() == records[f"{name}.wav"]["sha256"]
    audio, rate = sf.read(path, always_2d=True)
    assert rate == SAMPLE_RATE and audio.shape[1] == 1
    assert 0 < len(audio) <= SAMPLE_RATE * MAX_RENDER_SECONDS
    assert np.isfinite(audio).all() and np.max(np.abs(audio)) < 1.0


def test_lead_dis_carry_cc_by_attribution_and_unamplified_processing() -> None:
    (source,) = _provenance()["additional_sources"]
    assert source["license"] == "CC-BY-4.0" and source["record_id"] == "14963133"
    assert source["source_file_sha256"].startswith("033489e2") and source["source_file_sha256"].endswith("dbfe2")
    assert "Pedroza" in source["citation"] and "10.1109/ICASSP49660.2025.10887996" in source["citation"]
    assert source["recording_context"]["pickup"] == "Bridge-Neck"
    assert "No normalization or gain" in source["processing"]
    files = {item["file"]: item for item in source["samples"]}
    assert files["lead-1.wav"]["upstream_member"].endswith("directinput_08.wav")
    assert files["lead-2.wav"]["upstream_member"].endswith("directinput_10.wav")
    for item in files.values():
        assert 12 <= item["duration_seconds"] <= 16
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert "Guitar-TECHS" in readme and "CC BY 4.0" in readme


def test_snapshot_accepts_leads_and_rejects_unknown_samples() -> None:
    base = {"project_id": "p", "project_revision": 1, "chain_revision": 1, "chain_sha256": "a" * 64,
                "source_sample_sha256": "b" * 64, "engine_version": "1.7.3", "sample_rate": 44100,
                "dsp_conditions": {}, "audio_sha256": "c" * 64, "duration_seconds": 1.0, "peak": 0.5, "rms": 0.1,
                "created_at": "2026-01-01T00:00:00Z"}
    assert PreviewSnapshot(**base, sample_name="lead-2").sample_name == "lead-2"
    assert PreviewSnapshot(**base, sample_name="riff-1").sample_name == "riff-1"
    with pytest.raises(ValidationError):
        PreviewSnapshot(**base, sample_name="riff-3")
