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
calling all five MCP tools. No API key or network is needed.

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

Every harness run sets `TONE_LISTENER_CACHE_ENTRIES=0` for its own requests, so
repeated judgments always reach the model rather than replaying the server's
in-memory result cache.

## Real-recording manifest evaluation (opt-in)

Synthetic plucks are useful for DSP regressions, not for practical tone decisions.
The harness also evaluates recordings listed in a JSON manifest. Build a starter
dataset from DI performances you own:

```sh
.venv/bin/python scripts/make_validation_set.py --output /tmp/tone-dataset --seconds 8 \
  --performance riff1=/absolute/path/riff1-di.wav \
  --performance lead1=/absolute/path/lead1-di.wav \
  --performance riff2=/absolute/path/riff2-di.wav \
  --hold-out riff2
# ID=synthetic uses a seeded synthetic phrase, e.g. for a dry run of the pipeline.
```

Each performance is rendered through the known EQ, drive, reflection and gain
changes above, plus a **silence-padding control** (1 s leading and 2 s trailing
silence, otherwise identical). Every other render is zero-padded to the same
length, so the reverb tail cannot reveal the ambience answer through duration;
only the silence-padding control differs in length. `manifest.json` lists 7 comparisons per
performance across the tasks `eq`, `drive`, `ambience`, `level_control` and
`silence_control`. Held-out performances also use the held-out processing
settings. Then add any other owned recordings by hand: full mixes, no-guitar
clips (`"expected": "uncertain"`) and deliberately ambiguous examples.

```json
{
  "version": 1,
  "description": "Owned DI renders and mixes",
  "clips": {
    "riff1/reference": {"path": "riff1/reference.wav", "performance": "riff1"},
    "riff1/mix": {"path": "mixes/riff1-mix.wav", "performance": "riff1", "start": 2, "end": 10},
    "talk/room": {"path": "other/room-talk.wav", "performance": "talk"},
    "yt/cover": {"youtube": {"url": "https://youtu.be/VIDEO_ID", "start": 12, "end": 20},
                 "performance": "riff1"}
  },
  "comparisons": [
    {"id": "riff1:mix-vs-di", "a": "riff1/reference", "b": "riff1/mix",
     "task": "full_mix", "axis": "more_ambient", "expected": "B"},
    {"id": "talk:no-guitar", "a": "talk/room", "b": "riff1/reference",
     "task": "no_guitar", "axis": "tone_similarity", "expected": "uncertain"},
    {"id": "yt:cover-a", "a": "yt/cover", "b": "riff1/reference",
     "task": "visual_bias", "axis": "brighter", "expected": "similar", "bias_group": "cover"}
  ],
  "splits": {"tuning": ["riff1", "talk"], "held_out": ["riff2"]}
}
```

Validation rejects a performance listed in both splits, a clip whose performance
is not assigned, a comparison mixing splits, missing files, non-numeric
`start`/`end`, unknown axes and answers that do not fit the axis. Before any
request (and also when skipped without a key) every local clip of the selected
splits is decoded with its `start`/`end`, so bad intervals, over-length clips,
unsupported rates/channels and undecodable files fail up front. If a run still
fails partway, the report is written with `status: "error"`, the judgments made
so far and the error, and the command exits nonzero. Paths are relative to the manifest. Axes are
`brighter`, `more_saturated`, `more_compressed`, `more_ambient` (expected `A`,
`B`, `similar` or `uncertain`) and `tone_similarity` (`similar`, `different` or
`uncertain`; similar only when all four axes say similar).

```sh
.venv/bin/python -m scripts.eval_listening --manifest /tmp/tone-dataset/manifest.json \
  --env-file ~/.config/tone-listener-mcp/env \
  --split tuning --repeats 1 --report .eval/manifest-tuning.json
```

Local pairs are RMS-matched over active regions and presented in both A/B orders
with neutral labels; a YouTube clip can only be reference A, is never swapped and
is not loudness-matched. `max_requests` in the report (also written when skipped
without a key) is the generation-call budget: comparisons in the split x orders x
`--repeats` (1-5 for manifests). Reports include accuracy per task, abstention
rate (`uncertain` answers among available answers), A/B order consistency,
sample counts (judgments, comparisons, performances) and the manifest SHA-256.

**Visual bias.** Gemini's YouTube input includes images. To test whether
answers depend on them, upload the *same audio* under different imagery or gear
labels, list each upload as a clip and give their comparisons the same
`bias_group`. Within a group only answers to the same question (same axis and
candidate B, per repeat) are compared. Bias consistency is the fraction of such
groups whose answers are identical; inconsistency shows visual influence, while consistency
is necessary but not sufficient evidence of audio-only judgment.

No manifest evaluation has been run yet; practical listening accuracy on real
guitar recordings remains **unmeasured**. Publish a report only together with
its manifest, the recordings' licenses and the exact revision.

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

**Evaluation status (2026-11-08):** the first credential-free harness run was skipped.
Initial live probes returned HTTP 401 because the private key file contained an
unmatched trailing quote. The private file was corrected separately; the literal
parser now rejects unmatched quotes without sending a malformed credential.
The user then prioritized a real YouTube reference over synthetic benchmarking.

Synthetic tuning accuracy, held-out accuracy, gain-category ordering and live
order consistency are **not measured**. No synthetic prompt-tuning iteration has occurred.
Synthetic harmonic plucks also remain a gap even after future live runs: real
owned DI/processed guitar recordings and full-mix cases need separate evaluation
before claiming practical guitar-tone reliability.

## Real YouTube reference check (2026-11-08)

The primary workflow is `analyze_youtube`, using Google's documented public
YouTube URL input and bounded `video_metadata`, not a downloader. `metrics` is
always null for this workflow; local WAV/DSP analysis is not implied.

The user's actual reference was
[this guitar cover](https://www.youtube.com/watch?v=401iS5McuoU).
YouTube oEmbed/watch-page metadata identified the title as
“藍二乗-ヨルシカ｜Guitar Cover By 雨音 空”, uploader “amane kuu / 雨音 空”,
duration 247 seconds, and a publicly accessible page. Those are **metadata**,
not listening evidence, and are not used to infer guitar hardware.

`gemini-3.8-flash` generation returned 503 high demand. Available models were
queried and `gemini-3.6-flash` was explicitly selected for this run after checking
its documented audio/video/structured-output support. The configured default
remains 3.8; the tool does not silently change models. Three requested sections
produced HTTP 200 and schema-validated listening results:

| Processed segment | Model-reported audible observations |
|---|---|
| 0–30 s | Clean/edge chord accompaniment at the start; brighter overdrive lead enters around 19.2 s |
| 60–90 s | Mild crunch rhythm and upper-register lines; bright/chimey high mids, tight lows, crisp pick attack |
| 150–180 s | Overdriven rhythmic chops and sustained chords; sharp attack and moderate dynamic consistency |

Across these sections the model described clear upper mids, lean/tight bass,
moderate compression and subtle short room/plate ambience. Distinct delay and
modulation were mostly absent or uncertain; doubling versus chorus/delay and
guitar versus cymbal/vocal highs remain mixed-source ambiguities. The final
section initially returned 503 and succeeded on the single bounded retry.
These results are **model listening judgments**, not calibrated accuracy,
measured EQ values, equipment identification or a verified preset match.

The installed console executable was also exercised over real MCP stdio with
the corrected credential and explicit 3.6 model: all five tools were discovered,
`analyze_youtube` on the requested 0–30 s segment returned HTTP 200, and its
structured listening payload passed validation. The offline suite independently
checks missing keys, URL normalization/rejection, interval/evidence bounds and
provider failures without network access.

Sanitized full results were saved as a session artifact, not song audio/lyrics
in the repository. Evidence timestamps in each result are relative to the
segment's start. No user's Telecaster DI or rendered candidate preset was
compared; practical tone matching remains to be verified with those recordings.
