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

## Blinded live evaluation (opt-in)

Run from the repository root. Keep the key outside the checkout in
`~/.config/tone-listener-mcp/env`, readable only by your user (`chmod 600`).
That file contains `GEMINI_API_KEY=...`, optionally `TONE_LISTENER_MODEL=...`.
Never paste the key into reports, commit it, or enable shell tracing.

```sh
.venv/bin/python -m scripts.eval_listening \
  --env-file ~/.config/tone-listener-mcp/env \
  --split tuning --repeats 3 --report .eval/iteration-1-tuning.json
# Freeze code/prompt choices before evaluating the held-out split:
.venv/bin/python -m scripts.eval_listening \
  --env-file ~/.config/tone-listener-mcp/env \
  --split held_out --repeats 3 --report .eval/iteration-1-held-out.json
```

`--env-file` parses only literal allowed `KEY=value` entries with optional outer
quotes and comments. It requires mode 0600 and current-user ownership. It never
sources/executes the file or expands shell expressions. Exported variables remain
supported when the option is omitted. Keys never enter reports or CLI arguments.

Each split makes at most **45 generation calls**: six pairs, both A/B orders,
three repeats (36 comparisons), plus reference/light-drive/heavy-drive analysis
three times (9 analyses). Each comparison uploads two files; each analysis
uploads one. Rate-limit pacing defaults to one second between requests;
`--pause` increases it. Both splits cost at most 90 generation calls. Evaluate
tuning first, inspect failures and only rerun when an evidence-based change is
worth the additional cost. Stop after at most five iterations or diminishing
returns. CI never invokes this harness with credentials.

Pairs cover brighter/darker EQ, light versus heavy drive, added reflections,
and quieter/louder gain variants expected to have the same tone after matching.
Audio is padded to equal durations, RMS-matched, presented with neutral clip
labels, randomly ordered, and tested in both orders. Neither filenames nor
ground truth are supplied to Gemini. Tuning uses seed 42; held-out uses seed
90210, different note/decay patterns, EQ split/strength, drive and reflection
settings. These choices are fixed before live testing; do not tune on held-out
failures and then call the same set independent validation.

JSON reports include individual predictions/responses, per-pair direction
accuracy, swap consistency and strict gain-category ordering, plus model,
revision and source hashes. `uncertain` and unavailable responses do not score
as correct. Gain ordering requires three strictly increasing known gain
categories; this is a deliberately stringent synthetic benchmark, not a claim
that every small drive increment must change a musical category. Gain-only
tone equality requires all four comparison axes to say `similar`.

Reports must be outside the repo or under gitignored `.eval/`; existing reports
are never overwritten. Without a key the harness writes `status: "skipped"`
and performs **zero API calls**. A failed API response stops the budget early
with `status: "blocked"` and a nonzero exit status. Partial results are not a
completed benchmark, and skipped runs are not evidence of listening accuracy.
If the configured model is unavailable, inspect model availability with your
Google account and select an audio/structured-output capable model using
`TONE_LISTENER_MODEL`; do not silently substitute models mid-evaluation.
The backend omits sampling overrides as required by the
[Gemini 3.8 migration guide](https://aistudio.google.com/docs/whats-new-gemini-3.8#migrate-to-gemini-3-8-flash).
No setting guarantees repeatable responses.

## Offline feedback findings (2026-11-08)

The original 50 ms max/min power range was mathematically defined, but its
semantics were unhelpful with silent gaps or tiny numerical tails. The existing
`short_term_rms_range_db` field now reports **p90-p10 of window dBFS above
-90 dBFS**, requiring at least three active windows. This is an intentional
statistic/compatibility change; all-window RMS percentiles are unchanged.
The result explicitly reports `short_term_range_method` and
`active_window_fraction` so consumers can interpret coverage.

On the original 4-second synthetic set (rounded dB):

| Variant | Previous max/min range | Active-window p90-p10 range |
|---|---:|---:|
| Reference / quieter / louder | Undefined because of silence | 19.99 |
| Light drive | Undefined because of silence | 19.54 |
| Heavy drive | Undefined because of silence | 15.83 |
| Dark EQ | 48.35 | 21.25 |
| Bright EQ | 45.69 | 20.99 |
| Simple reflections | 313.28 | 24.13 |

These are **different statistics**, not an improvement in musical recognition
accuracy. Added regressions cover silence gaps, decaying tails, insufficient
active windows, scoring/swap invariants, seeded held-out reproducibility and a
fully mocked blinded run. Band hints now always explain that saturation
harmonics, playing and ambience can affect spectral energy; they propose an EQ
direction only conditionally. No fragile saturation or reverb detector was added.
Weak spectral changes from reflections remain a limitation of aggregate DSP
metrics, not proof that ambience is absent.

Upload preprocessing also needed a concrete correctness fix: matching
original-resolution channel power did not ensure equal RMS after stereo
downmix/resampling. With A = mono `x` and B = stereo `[x, -0.5*x]`, the previous
uploads left B 10 dB quieter. Comparisons now match the prepared 16 kHz mono
signals, then apply common peak attenuation. The additional upload-only gain
and silent/sub-floor edge case are explicitly reported; local metrics are
unchanged. Mocked upload tests read **both actual WAVs** and verify RMS equality
within 1e-6 relative tolerance for that stereo case, 12 kHz band loss during
resampling and peak-safety cases, with separate antiphase/silence regressions.
These tests verify preprocessing bytes, not live model quality.

**Live status (2026-11-08):** the first credential-free harness run was skipped.
After a credential was provided, real audio probes reached the API but returned
HTTP 401: Files upload initiation failed before transfer, and a separate supported
inline-WAV `generateContent` request also failed authentication. SDK and REST
metadata probes using the same literal credential were also rejected in this
environment. No key or provider error body is stored in reports. This is a
live-access blocker, not evidence of bad listening quality or a model-direction
failure. The full benchmark was not run against a failing credential.

Tuning accuracy, held-out accuracy, gain-category ordering and live order
consistency are **not measured**. No live prompt-tuning iteration has occurred.
Synthetic harmonic plucks also remain a gap even after future live runs: real
owned DI/processed guitar recordings and full-mix cases need separate evaluation
before claiming practical guitar-tone reliability.
