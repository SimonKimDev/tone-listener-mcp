import numpy as np
import pytest
import soundfile as sf

from tone_listener_mcp.audio_io import AudioClip, load_audio, upload_wav


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


def test_upload_conversion_and_cleanup(tmp_path):
    path = tmp_path / "stereo.wav"
    sf.write(path, np.column_stack((np.ones(48000) * 0.4, np.zeros(48000))), 48000)
    clip = load_audio(str(path))
    with upload_wav(clip) as temporary:
        mono, rate = sf.read(temporary)
        assert rate == 16000
        assert mono.shape == (16000,)
        assert np.mean(mono[100:-100]) == pytest.approx(0.2, abs=1e-4)
    assert not temporary.exists()


def test_upload_cleanup_on_error(tmp_path):
    path = tmp_path / "mono.wav"
    sf.write(path, np.zeros(8000), 8000)
    with pytest.raises(RuntimeError), upload_wav(load_audio(str(path))) as temporary:
        raise RuntimeError("upload failed")
    assert not temporary.exists()


@pytest.mark.parametrize("frames,rate", [(15, 44100), (1001, 8000), (27, 48000)])
def test_exact_end_of_file(tmp_path, frames, rate):
    path = tmp_path / "small.wav"
    data = np.arange(frames, dtype=np.float32) / frames
    sf.write(path, data, rate, subtype="FLOAT")
    for end in (None, frames / rate, frames / rate + 1):
        loaded = load_audio(str(path), end=end)
        np.testing.assert_array_equal(loaded.samples[0], data)


def test_multichannel_and_high_sample_rate_rejected(tmp_path):
    path = tmp_path / "unsupported.wav"
    sf.write(path, np.zeros((20, 3)), 48000)
    with pytest.raises(ValueError, match="mono and stereo"):
        load_audio(str(path))
    sf.write(path, np.zeros(20), 384000)
    with pytest.raises(ValueError, match="Sample rate"):
        load_audio(str(path))


def test_decoded_size_bound_before_read(tmp_path, monkeypatch):
    path = tmp_path / "bounded.wav"
    sf.write(path, np.zeros(100), 16000)
    monkeypatch.setattr("tone_listener_mcp.audio_io.MAX_DECODED_BYTES", 200)
    with pytest.raises(ValueError, match="128 MiB"):
        load_audio(str(path))
    assert load_audio(str(path), end=0.001).samples.shape[1] == 16


def test_antiphase_upload_rejected():
    tone = np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000).astype(np.float32)
    audio = AudioClip(np.stack((tone, -tone)), 16000, "synthetic")
    with pytest.raises(ValueError, match="Stereo downmix cancels"), upload_wav(audio):
        pytest.fail("A cancelled downmix must not be uploaded")
