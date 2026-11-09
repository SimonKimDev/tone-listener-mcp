import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_local_search as evidence


def test_headroom_uses_every_channel_not_the_mono_downmix() -> None:
    samples = np.zeros((8, 2), dtype=np.float32)
    samples[0] = [1.3, -1.3]  # cancels to 0 in a mono downmix
    samples[1] = [1.2, 0.2]  # mono 0.7
    samples[2, 1] = -1.1
    report = evidence.headroom(samples)
    assert report["channels"] == 2
    assert report["peak"] == pytest.approx(1.3)
    assert report["peak_per_channel"] == pytest.approx([1.3, 1.3])
    assert report["samples_over_full_scale"] == 4
    assert report["frames_with_any_channel_over_full_scale"] == 3
    mono = samples.mean(axis=1)
    assert float(np.max(np.abs(mono))) < 1.0
    tonal = evidence.mono_tonal(mono)
    assert "mono downmix" in tonal["basis"] and "peak" not in tonal


def test_artifact_is_byte_copy_of_snapshot_wav(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    data = b"RIFF-native-snapshot-bytes"
    (cache / "snap.wav").write_bytes(data)
    snapshot = SimpleNamespace(id="snap", audio_sha256=hashlib.sha256(data).hexdigest())
    artifact = evidence.deliver_artifact(cache, snapshot, tmp_path, "current-riff-1.wav")
    assert (tmp_path / "current-riff-1.wav").read_bytes() == data
    assert artifact["sha256"] == artifact["snapshot_audio_sha256"] == snapshot.audio_sha256


def test_artifact_refuses_mismatched_snapshot(tmp_path: Path) -> None:
    (tmp_path / "snap.wav").write_bytes(b"changed")
    snapshot = SimpleNamespace(id="snap", audio_sha256="0" * 64)
    with pytest.raises(RuntimeError):
        evidence.deliver_artifact(tmp_path, snapshot, tmp_path, "x.wav")
    assert not (tmp_path / "x.wav").exists()


def test_di_identity_matches_provenance() -> None:
    identity = evidence.di_identity()
    assert set(identity) == {"riff-1", "riff-2"}
    assert all(item["license"] == "CC0-1.0" and len(item["sha256"]) == 64 for item in identity.values())


def test_di_identity_covers_leads_with_their_own_license() -> None:
    identity = evidence.di_identity(("riff-1", "lead-1", "lead-2"))
    assert identity["riff-1"]["license"] == "CC0-1.0"
    assert identity["lead-1"]["license"] == "CC-BY-4.0" and identity["lead-2"]["upstream_file"].endswith("_10.wav")
    assert identity["lead-1"]["source_file_sha256"].startswith("033489e2")


def test_tonal_features_report_band_energy_ratios() -> None:
    t = np.arange(44_100) / 44_100
    low = evidence.mono_tonal(np.sin(2 * np.pi * 100 * t))
    high = evidence.mono_tonal(np.sin(2 * np.pi * 4000 * t))
    assert low["energy_below_200hz_ratio_db"] > -0.1 and low["energy_above_2khz_ratio_db"] < -60
    assert high["energy_above_2khz_ratio_db"] > -0.1 and high["energy_below_200hz_ratio_db"] < -60
