import json

import numpy as np
import pytest

from tone_listener_mcp.audio_io import AudioClip
from tone_listener_mcp.metrics import measure


def sine(rate=48000, frequency=1000):
    return 0.5 * np.sin(2 * np.pi * frequency * np.arange(rate) / rate)


def clip(samples, rate=48000):
    return AudioClip(np.atleast_2d(samples).astype(np.float32), rate, "synthetic")


def test_sine_levels():
    result = measure(clip(sine()))["level"]
    assert result["peak_dbfs"] == pytest.approx(-6.0206, abs=0.01)
    assert result["rms_dbfs"] == pytest.approx(-9.0309, abs=0.01)
    assert result["crest_factor_db"] == pytest.approx(3.0103, abs=0.01)
    assert result["short_term_rms_range_db"] == pytest.approx(0, abs=0.001)


def test_silence_is_valid_json():
    result = measure(clip(np.zeros(48000)))
    assert result["level"]["rms_dbfs"] is None
    assert result["level"]["crest_factor_db"] is None
    assert result["level"]["silent_window_fraction"] == 1
    json.dumps(result, allow_nan=False)


def test_dynamics():
    samples = sine()
    samples[:24000] *= 0.1
    result = measure(clip(samples))["level"]
    assert result["short_term_rms_range_db"] == pytest.approx(20, abs=0.01)


def test_sine_spectrum():
    result = measure(clip(sine()))["spectrum"]
    assert result["centroid_hz"] == pytest.approx(1000, abs=2)
    assert result["rolloff_85_hz"] == pytest.approx(1000, abs=15)
    assert result["bands"]["800_2000"]["relative_db"] == pytest.approx(0, abs=0.01)
    assert result["bands"]["2000_5000"]["relative_db"] < -60


def test_white_noise_is_flatter():
    rng = np.random.default_rng(42)
    noise = measure(clip(rng.normal(0, 0.1, 48000)))["spectrum"]
    tone = measure(clip(sine()))["spectrum"]
    assert noise["flatness"] > 0.8
    assert noise["flatness"] > tone["flatness"]


def test_low_rate_and_single_sample():
    result = measure(clip(np.ones(1) * 0.1, rate=8000))
    assert not result["spectrum"]["bands"]["above_8000"]["available"]
    assert result["spectrum"]["bands"]["above_8000"]["relative_db"] is None
    json.dumps(result, allow_nan=False)
