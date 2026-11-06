import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tone_listener_mcp.audio_io import load_audio
from tone_listener_mcp.comparison import compare
from tone_listener_mcp.metrics import measure


def test_known_variants(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_validation_set.py"
    subprocess.run(
        [sys.executable, str(script), "--output", str(tmp_path), "--seconds", "1"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    recordings = {path.stem: load_audio(str(path)) for path in tmp_path.glob("*.wav")}
    assert len(recordings) == 8
    reference = measure(recordings["reference"])
    quiet = compare(recordings["reference"], recordings["gain_quiet"])
    assert quiet["loudness_match"]["gain_db_applied_to_b"] == pytest.approx(6.0206, abs=0.01)
    loud = compare(recordings["reference"], recordings["gain_loud"])
    assert loud["loudness_match"]["gain_db_applied_to_b"] == pytest.approx(-3.5218, abs=0.01)
    assert measure(recordings["eq_dark"])["spectrum"]["centroid_hz"] < (
        reference["spectrum"]["centroid_hz"]
    )
    assert measure(recordings["eq_bright"])["spectrum"]["centroid_hz"] > (
        reference["spectrum"]["centroid_hz"]
    )
    assert measure(recordings["drive_heavy"])["level"]["crest_factor_db"] < (
        measure(recordings["drive_light"])["level"]["crest_factor_db"]
    )
    assert recordings["reverb"].duration > recordings["reference"].duration
    assert all(np.max(np.abs(recording.samples)) <= 1 for recording in recordings.values())
    retry = subprocess.run(
        [sys.executable, str(script), "--output", str(tmp_path), "--seconds", "1"],
        check=False, capture_output=True, text=True, timeout=30,
    )
    assert retry.returncode != 0
    assert "Refusing to overwrite" in retry.stderr
