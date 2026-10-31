import numpy as np
import pytest
import soundfile as sf

from tone_listener_mcp.audio_io import load_audio


def test_load_and_clip(tmp_path):
    path = tmp_path / "stereo.wav"
    samples = np.column_stack((np.linspace(-0.5, 0.5, 48000), np.zeros(48000)))
    sf.write(path, samples, 48000, subtype="FLOAT")
    clip = load_audio(str(path), 0.25, 0.75)
    assert clip.samples.shape == (2, 24000)
    assert clip.samples.dtype == np.float32
    assert clip.duration == 0.5
    assert clip.start_seconds == 0.25
    np.testing.assert_allclose(clip.samples[0], samples[12000:36000, 0], atol=1e-7)


@pytest.mark.parametrize("start,end", [(-1, None), (0, 0), (2, None), (float("nan"), None)])
def test_invalid_intervals(tmp_path, start, end):
    path = tmp_path / "test.wav"
    sf.write(path, np.ones(100), 100)
    with pytest.raises(ValueError):
        load_audio(str(path), start, end)


def test_missing_file():
    with pytest.raises(ValueError, match="does not exist"):
        load_audio("/nonexistent/tone.wav")


def test_duration_cap(tmp_path, monkeypatch):
    path = tmp_path / "test.flac"
    sf.write(path, np.zeros(16000), 16000)
    monkeypatch.setenv("TONE_LISTENER_MAX_SECONDS", "0.5")
    with pytest.raises(ValueError, match="exceeds"):
        load_audio(str(path))
    assert load_audio(str(path), end=0.5).duration == 0.5
    monkeypatch.setenv("TONE_LISTENER_MAX_SECONDS", "nan")
    with pytest.raises(ValueError, match="positive finite"):
        load_audio(str(path))
