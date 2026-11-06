# Controlled validation

Generate recordings you own rather than committing copyrighted reference audio:

```sh
.venv/bin/python scripts/make_validation_set.py --output /tmp/tone-validation --seconds 4
# Or use the same DI phrase for every variant:
.venv/bin/python scripts/make_validation_set.py --output /tmp/my-di-validation \
  --source /absolute/path/to/di.wav --seconds 4
```

The script creates a synthetic plucked phrase by default, or processes mono/stereo
DI audio. It refuses to overwrite existing WAVs. All variants receive the same
headroom reduction if needed, preserving known gain ratios. Nothing is uploaded.

| Variant | Known change | Expected direction |
|---|---|---|
| `gain_quiet` / `gain_loud` | Linear gain 0.5 / 1.5 | RMS changes -6.02 / +3.52 dB; matched tone stays the same |
| `drive_light` / `drive_heavy` | `tanh` soft clipping with input multipliers 2 / 8 | More saturation, lower crest factor; not an amp simulation |
| `eq_dark` / `eq_bright` | Reduce / increase signal above a low-pass split | Lower / higher centroid and upper-band relative energy |
| `reverb` | Four decaying reflections, 35-173 ms | More ambience and a longer tail; deliberately simple, not a room model |

Call `audio_metrics` on the variants, then `compare_audio` with `reference.wav`
as A and a variant as B. The gain variants should require only loudness matching,
not tonal EQ correction. Drive/EQ cases should track the directions above.
Do not interpret spectral changes alone as proof of a specific effect.

With a Gemini key configured, `analyze_audio` and `compare_audio` also upload clips.
Check whether listening judgments agree with known changes; inspect evidence and
uncertainties instead of treating confidence as calibrated accuracy. Do not expect
bit-identical listening responses. Test the same DI and phrase as well as synthetic
signals: a synthetic phrase is useful for DSP regressions, not proof of musical
tone-recognition quality. A live listening check is opt-in and incurs API usage.

Run `.venv/bin/ruff check . && .venv/bin/pytest -q` for the offline suite.
Tests cover bounded decoding, level/spectrum/stereo metrics, linear-gain invariance,
known EQ/drive directions, provider mocks and an actual stdio process listing and
calling all four MCP tools. No API key or network is needed.
