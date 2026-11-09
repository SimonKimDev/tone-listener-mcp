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


def test_robust_range_ignores_silence_gaps():
    samples = sine()
    samples[:24000] *= 0.1
    gapped = np.concatenate((np.zeros(48000), samples, np.zeros(48000)))
    result = measure(clip(gapped))["level"]
    assert result["short_term_rms_range_db"] == pytest.approx(20, abs=0.01)
    assert result["active_window_fraction"] == pytest.approx(1 / 3)
    assert result["silent_window_fraction"] == pytest.approx(2 / 3)


def test_decaying_tail_range_is_finite_and_bounded():
    samples = sine() * np.exp(-np.arange(48000) / 48000 * 50)
    result = measure(clip(samples))["level"]
    assert result["short_term_rms_range_db"] is not None
    assert 0 < result["short_term_rms_range_db"] < 90
    assert result["active_window_fraction"] < 1
    json.dumps(result, allow_nan=False)


def test_too_few_active_windows():
    result = measure(clip(sine()[:2400]))["level"]
    assert result["short_term_rms_range_db"] is None


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


def test_stereo_and_antiphase():
    tone = sine()
    duplicated = measure(clip(np.stack((tone, tone))))
    assert duplicated["stereo"]["lr_correlation"] == pytest.approx(1)
    assert duplicated["stereo"]["side_mid_energy_ratio_db"] is None
    antiphase = measure(clip(np.stack((tone, -tone))))
    assert antiphase["stereo"]["lr_correlation"] == pytest.approx(-1)
    assert antiphase["spectrum"]["centroid_hz"] == pytest.approx(1000, abs=2)
    assert measure(clip(tone))["stereo"] is None
    json.dumps(antiphase, allow_nan=False)


def test_gated_level_ignores_silence_padding():
    tone = sine()
    padded = measure(clip(np.concatenate((tone, np.zeros(96000)))))["level"]
    plain = measure(clip(tone))["level"]
    assert padded["gated_rms_dbfs"] == pytest.approx(plain["gated_rms_dbfs"], abs=1e-6)
    assert padded["gated_rms_dbfs"] == pytest.approx(-9.0309, abs=0.01)
    assert padded["gated_crest_factor_db"] == pytest.approx(3.0103, abs=0.01)
    assert padded["gated_duration_seconds"] == pytest.approx(1, abs=0.002)
    assert padded["rms_dbfs"] < plain["rms_dbfs"] - 4


def test_gated_level_undefined_for_silence():
    result = measure(clip(np.zeros(48000)))["level"]
    assert result["gated_rms_dbfs"] is None
    assert result["gated_crest_factor_db"] is None
    assert result["gated_duration_seconds"] == 0


def test_gated_level_keeps_quiet_material_within_gate():
    rate = 48000
    t = np.arange(rate // 2) / rate
    loud, quiet = 0.5 * np.sin(2 * np.pi * 440 * t), 0.125 * np.sin(2 * np.pi * 440 * t)
    tail = 0.02 * np.sin(2 * np.pi * 440 * t)
    level = measure(clip(np.concatenate((loud, quiet, tail)), rate))["level"]
    # The -12 dB passage stays active; the -28 dB tail is below the relative gate.
    assert level["gated_duration_seconds"] == pytest.approx(1.0, abs=0.06)
    expected = 10 * np.log10((0.5**2 + 0.125**2) / 4)
    assert level["gated_rms_dbfs"] == pytest.approx(expected, abs=0.05)


def test_gated_level_keeps_continuous_ambience():
    rate = 48000
    rng = np.random.default_rng(3)
    t = np.arange(rate) / rate
    note = 0.3 * np.sin(2 * np.pi * 220 * t) * (t < 0.5)
    ambience = measure(clip(note + rng.normal(0, 0.03, rate), rate))["level"]
    # About 17 dB below the note: within the relative gate, so the room stays active.
    # Its end loses window * gate / level (about 6 dB above the gate: 12.5 ms).
    assert ambience["gated_duration_seconds"] == pytest.approx(1.0 - 0.0125, abs=0.003)
    hiss = measure(clip(note + rng.normal(0, 0.01, rate), rate))["level"]
    assert hiss["gated_duration_seconds"] == pytest.approx(0.5, abs=0.002)


def test_gated_level_defined_for_clicks_passing_only_together():
    samples = np.zeros(48000)
    # Each click alone is below -70 dBFS per 50 ms window; together they pass.
    samples[[20000, 20100]] = 0.012
    level = measure(clip(samples))["level"]
    assert level["gated_rms_dbfs"] == pytest.approx(10 * np.log10(2 * 0.012**2 / 2400), abs=0.01)
    assert level["gated_duration_seconds"] == 0
