# Tool reference

Detailed semantics behind the [README](../README.md) quickstart. Response
fields are typed by the Pydantic models in `src/tone_listener_mcp/schemas.py`,
published as each tool's MCP `outputSchema`; responses may gain new keys in later versions.
Listening uses the Gemini SDK Files upload and `models.generate_content` APIs; the
default model follows the [Gemini audio documentation](https://ai.google.dev/gemini-api/docs/audio).
The YouTube workflow never downloads or extracts audio, isolates stems or computes
local DSP metrics.

## Inputs

Invalid files, URLs or intervals return an MCP tool error. Local-file times are
seconds; `start` defaults to zero, `end` is exclusive and defaults to EOF. An
endpoint past EOF is clamped to EOF. `~` is expanded, and original channels/sample
rate are retained for local metrics. Only mono/stereo audio up to 192 kHz is
accepted. Selected float32 decoded audio is limited to 128 MiB in addition to the
duration cap; shorten clips when either limit is exceeded.

## Local metrics

For a 0.5-peak, 1 kHz synthetic sine, selected fields look like this
(values rounded; this is an excerpt, not the complete response):

```json
{
  "level": {
    "peak_dbfs": -6.02,
    "rms_dbfs": -9.03,
    "crest_factor_db": 3.01,
    "short_term_rms_range_db": 0.0,
    "silent_window_fraction": 0.0
  },
  "spectrum": {
    "centroid_hz": 1000.0,
    "rolloff_85_hz": 1007.81,
    "bands": {
      "800_2000": {"available": true, "effective_range_hz": [800, 2000], "relative_db": 0.0}
    }
  },
  "stereo": null
}
```

Levels include 50 ms short-term RMS power percentiles (p10, p50, p90) over all
windows, plus a robust active-window range as a dynamics proxy. The range is
p90 minus p10 of window dBFS above -90 dBFS, requiring at least three active
windows. `short_term_range_method` describes this statistic and
`active_window_fraction` reports its coverage. These are **not LUFS** or a
perceptual loudness measurement. This replaces the original max/min range;
consumers should not compare old and new range values as the same statistic.
Spectrum uses Welch's method with a Hann window and up to 4096 frames, averaging
channel powers so antiphase stereo does not disappear. Frequency resolution and
Nyquist are reported. Band IDs are `below_80`, `80_250`, `250_800`, `800_2000`,
`2000_5000`, `5000_8000` and `above_8000` (Hz ranges). Each band's `effective_range_hz`
gives its lower and upper frequency bounds, with the upper bound capped at
Nyquist. The range is `null` when the band starts at or above Nyquist.
`relative_db` is band power relative to total spectral power, not an absolute
EQ control value.

Stereo reports L/R correlation and side/mid energy ratio dB, with separate mid
and side RMS levels. `null` means undefined or unavailable, including silence,
mono stereo metrics, zero side/mid energy, or bands above Nyquist. The robust
range is undefined with fewer than three active windows; silence gaps and
sub-floor tails are excluded. Inspect `silent_window_fraction` and
`active_window_fraction` as well. No NaN or Infinity is returned.

## Comparison

B is RMS-matched to A **before** spectral comparison and listening. By default
(`matching: "active"`) only gated, non-silent regions are matched. Unweighted 50 ms
windows are evaluated at every sample offset and kept when above -70 dBFS and within
20 dB of their mean; a sample is active when every window containing it is kept, and
the gated RMS is the RMS of active samples. Windows overhanging a clip edge count as
digital silence, so silence padding at any length or alignment does not change the
applied gain, gated crest factor or active seconds, even for staccato or very short
clips. The audio itself is not trimmed or retimed. Gaps shorter than 50 ms stay
active, and quiet passages or ambience within the gate are kept. At a boundary with
silence, material loses about `50 ms × gate / level` of measured activity: under 1 ms
for material near the clip's mean level, up to 50 ms just above the gate. In tests,
low-level noise padding moves the gain by less than 0.01 dB. The crest-factor delta
uses the gated RMS. Use `matching: "whole"` for whole-clip RMS that includes silence
and tails.
`loudness_match.activity` reports durations, gated active seconds and mismatch
flags; mismatches add a caution, because effect tails still affect spectra. The
Gemini uploads are matched with the same strategy. The response
retains original `metrics_a`/`metrics_b` and reports `matched_metrics_b`,
`loudness_match`, `deltas_b_minus_a` and `eq_direction_hints`. Linear gain alone
should not produce a tonal difference:

```json
{
  "loudness_match": {
    "applied": true,
    "gain_db_applied_to_b": 6.0206,
    "reason": null
  },
  "deltas_b_minus_a": {
    "relative_band_db": {"800_2000": 0.0},
    "centroid_hz": 0.0,
    "rolloff_85_hz": 0.0,
    "crest_factor_db": 0.0,
    "stereo": null
  }
}
```

This excerpt describes B at half A's amplitude. Deltas use **B minus A**;
positive band deltas mean relatively more energy in B. Band hints are non-causal
observations: saturation harmonics, playing and ambience can also change them.
They suggest an EQ direction **only if EQ is the cause**, not a distortion
diagnosis, exact knob settings or proof that EQ will recreate A. Hints need
evidence: `deltas_b_minus_a.band_evidence` marks each band `sufficient`,
`insufficient_energy` (below -50 dB relative in both A and B, where tiny absolute
changes give unstable ratios) or `unavailable` (above Nyquist, or every band when A
and B have different Nyquist frequencies). Raw deltas are still reported for
every band, but only `sufficient` bands with at least 1 dB of change produce
hints. Each hint is an object with `family: "eq"`, `band`, `band_label`,
`direction` (`cut`/`boost`), `delta_db`, `priority` (1 = largest difference),
`observation`, `alternative_explanation`, `listen_for` and a readable `text`.
Silence makes RMS
matching undefined and is explicitly reported. Stereo deltas require two stereo
clips. Gemini compares both uploads in one request and returns differences,
which clip is brighter/more saturated/more compressed/more ambient, suggested
adjustments and uncertainties. Each suggested adjustment is one experiment:

```json
{
  "family": "ambience",
  "direction": "reduce reverb amount",
  "priority": 1,
  "observation": "Sustained notes are similarly bright, but B has a longer audible tail",
  "alternative_explanation": "Room sound in the B recording",
  "listen_for": "Tail length after the final chord"
}
```

`family` is one of `gain`, `eq`, `compression`, `ambience`, `delay`,
`modulation`, `playing` or `other`.

When the clips have different Nyquist frequencies, their relative energies use
different normalization ranges. Relative band deltas are then `null` and the
EQ direction hints list is empty.

## Upload preprocessing

Local-file uploads are resampled to **16 kHz mono** (sample rate, not 16 kbps bitrate).
Stereo and fine high-frequency detail must come from local metrics, not listening
claims. Severe stereo downmix cancellation (more than 40 dB power loss) disables
listening with an explicit reason; provide a mono guitar stem instead.
For listening comparisons, a second RMS match is performed **after mono
conversion and resampling**, on the actual upload waveforms. Stereo cancellation
and high-frequency band loss can otherwise undo the original-resolution match.
`upload_loudness_match.additional_gain_db_applied_to_b` reports this additional
upload-only adjustment, separate from local `loudness_match`. Upload matching is
undefined if either prepared signal is silent or below -90 dBFS; the response
and provider prompt say so. Single-clip analysis has no upload A/B match.
If the prepared/matched signals exceed full scale, the same attenuation is
applied to both uploads and reported as `upload_gain_db`. Peak safety is checked
after resampling; original local measurements are unchanged.

## Listening

`analyze_youtube` and `analyze_audio` return a `listening` object with `summary`, `gain_level`,
`brightness`, `low_end`, `mids_character`, `pick_attack`, `compression_feel`,
`effects` (reverb/delay/modulation presence, amount and notes), `playing_notes`,
timestamped `evidence`, `uncertainties` and `confidence`.
Gain categories include clean, edge-of-breakup, crunch, overdrive, distortion,
high-gain and uncertain. Evidence timestamps are relative to the selected clip.

For YouTube, the result also reports the normalized `source` URL/interval,
`input_transport: "youtube_url"`, `metrics: null` and `metrics_unavailable`.
It uses the documented video URI/interval metadata through the Gemini API;
no downloader or local files are involved. A missing key returns a clear
YouTube-listening-unavailable reason, with no pretend local metrics.

For local-file analysis, no key gives an explicit local-only response:

```json
{
  "model": "gemini-3.8-flash",
  "listening": null,
  "listening_unavailable": "GEMINI_API_KEY is not set; local metrics only",
  "cleanup_warnings": [],
  "upload_gain_db": 0.0
}
```

This excerpt omits the returned local `metrics`. Provider errors and invalid
model responses similarly preserve metrics, with a sanitized failure reason
and `listening_failure_stage` (preparation, queue, client creation, upload, processing,
generation or validation) to distinguish actionable failures without exposing
API headers or response bodies.

## Failures, budgets, diagnostics and caching

`failure` is `null` on success, otherwise it includes
`{category, http_status, retryable, suggestion, retry_after_seconds, retry_after_invalid,
daily_request_limit, daily_token_limit}`.
Categories: `not_configured`, `audio_preparation`, `authentication` (401), `permission` (403),
`model_unavailable` (404), `invalid_request` (400), `quota` (429), `capacity` (5xx),
`provider_error`, `network`, `deadline`, `busy`, `upload_failed`, `invalid_response` and
`local_io`. A 429 with recognized structured quota IDs refines `quota` to
`daily_quota` or `rate_window`; daily quota takes precedence if both are reported.
`daily_quota` is not immediately retryable; `rate_window` is retryable.
Suggestions are fixed text; provider messages are never returned.

`retry_after_seconds` is the maximum valid delay from the HTTP `Retry-After` header
(seconds or an HTTP date) and Google `RetryInfo.retryDelay`, or `null` if neither
is valid. Past HTTP dates yield zero. `retry_after_invalid` flags a malformed
`Retry-After` header even when a valid `RetryInfo` delay exists. Daily request/token
limits are the minimum nonnegative integer values reported in matching daily
`QuotaFailure` violations, or `null` if unavailable. Only numeric values up to
12 digits are accepted. Unknown/malformed details do not override the generic
category. No raw headers, quota IDs, descriptions, URLs, provider messages or
payloads are returned. These hints do not trigger automatic quota retries.

One deadline (`TONE_LISTENER_REQUEST_DEADLINE_SECONDS`) bounds the total elapsed time of
waiting for an inference slot, each upload, upload processing (also capped at 30 s) and
generation. It is enforced on every socket connect, TLS handshake, read and write of every
provider request, including the YouTube 503 retry, so a slowly streamed response or a series of
individually quick requests still stops at the deadline with category `deadline`. When a host
resolves to several addresses, they are tried in order and share that one deadline, rather than
each getting a full connect timeout. A listening result is never accepted after the deadline,
whether it comes from the provider or the cache: a provider response that completes late is
neither returned nor cached, and a cache hit found after the budget is spent (for example by
slow audio preparation) fails with `deadline` at stage `queue`, exactly like a miss, while
the cached entry stays valid for later requests. The YouTube retry (one, after 1 s, on a 503
or a transient connect/timeout error) only covers the generate request, and only happens if at
least 1 s of budget would remain; uploads, upload status checks and deletes are not retried.
SDK HTTP retries are explicitly disabled; only that deadline-aware generate retry is allowed.
Remote upload cleanup always runs afterwards, even past the deadline, with its own 10 s total
budget per file, so cleanup can extend a call by at most 10 s per uploaded file (including
returning a result that was accepted before the deadline).
The deadline is enforced by wrapping the connection pools of a per-request httpx2 client passed
to google-genai, which relies on private httpx2/httpcore2 internals (`_transport`, `_mounts`,
`_pool._network_backend`). If those change in a new httpx2 release, listening fails closed with
category `local_io` at stage `create_client` instead of running without a deadline.
Not covered by the deadline:
- DNS resolution, which cannot be interrupted.
- The SDK's internal resumable-upload backoff (only used when the server omits the upload
  status), whose sleep can delay the `deadline` failure until the next socket operation.
- Within a very slowly draining upload, the deadline is rechecked every 64 KiB rather than on
  every send.

When all `TONE_LISTENER_MAX_CONCURRENT_REQUESTS` slots stay busy until the deadline, the
result fails with category `busy` at stage `queue` without uploading anything. A slot is held
until the request, including its cleanup, has fully finished, so no provider request outlives
the call.

Every listening result includes `diagnostics`: `elapsed_seconds`, `deadline_seconds`,
`max_concurrent_requests`, `prompt_version`, `schema_version` (response schema name plus a
hash of its JSON schema), `cache` (`hit`, `miss`, `disabled` or `not_used` when no request
was attempted; a `hit` with a `deadline` failure means a matching cached result was found but
not returned because the deadline had passed), provider token `usage` when reported, and `preprocessing` per uploaded clip
(original sample rate/channels/duration and the 16 kHz mono upload format).

Every result also includes root-level `request_counts`:
`{generate, upload, poll_upload, delete_upload}`. Counts are actual invocations of
`models.generate_content`, `files.upload`, `files.get` and `files.delete` within
this call, incremented immediately before invocation, including unsuccessful
attempts and cleanup. Cache hits, missing keys and failures before SDK invocation
have zero counts. Counts are not cached or inferred from token usage. These are
SDK-operation counts, not billing totals or wire-level HTTP counts: a resumable
upload can involve multiple HTTP exchanges inside one `files.upload` invocation.
Cleanup counts include failed delete attempts; cleanup errors remain separate
in `cleanup_warnings` and do not discard an otherwise valid profile.

Successful, validated results are cached in memory for the life of the server process.
The key covers the model, prompt version, schema version, system instructions, prompt
(including your question/focus), YouTube URL and interval, and the exact prepared audio
samples, so a changed file, interval, matching strategy or question is a new request.
A cache hit makes no API call and reports the original call's `usage`. Failures are never
cached. Restart the server or set `TONE_LISTENER_CACHE_ENTRIES=0` to force fresh listening.

Clients that send an MCP progress token receive progress notifications for audio
preparation, each upload, generation and validation (or "Served from cache").
**Confidence is model self-assessment, not calibrated accuracy.** Live listening
quality requires evaluation against recordings with known changes.

## Python library API

Import from `tone_listener_mcp.listener`; no MCP server is needed:

```python
listen_tone_profile(
    clip: AudioClip, progress: Progress | None = None, *,
    use_cache: bool = False, request_attempts: Literal[1, 2] | None = 1,
) -> dict

listen_youtube_tone_profile(
    url: str, start: float | None = None, end: float | None = None,
    progress: Progress | None = None, *,
    use_cache: bool = False, request_attempts: Literal[1, 2] | None = 1,
) -> dict

youtube_input(
    url: str, start: float | None = None, end: float | None = None,
) -> tuple[str, float, float, google.genai.types.Part]

failure_info(exc: Exception | None, stage: str | None = None) -> dict
```

`AudioClip` and `load_audio(path, start=None, end=None)` are available from
`tone_listener_mcp.audio_io`. `Progress` is an optional `(step, total, message)`
callback; callback errors do not fail listening.

Both profile functions return the standard listening envelope described above,
with a validated `ToneProfile.model_dump(mode="json")` in `listening`, or `null`
plus the usual sanitized `failure` and `listening_unavailable`. They include
diagnostics, request counts, cleanup warnings and upload metadata. The YouTube
variant additionally returns `source`, `metrics: null` and `metrics_unavailable`;
local profiles do not compute DSP metrics (use `metrics.measure` separately).
Use `ToneProfile.model_validate(result["listening"])` if a Pydantic instance is
preferred. There is no `failure_info` field alias: result metadata lives in `failure`.

`ToneProfile` has fixed categorical axes:

| Axis | Values |
|---|---|
| `guitar_dominance` | clear, mixed, uncertain |
| `gain` | clean, edge-of-breakup, crunch, overdrive, distortion, high-gain, uncertain |
| `brightness` | dark, slightly-dark, balanced, slightly-bright, bright, uncertain |
| `bass_level` | low, typical, high, uncertain |
| `bass_tightness` | loose, neutral, tight, uncertain |
| `mids` | scooped, neutral, present, forward, uncertain |
| `attack` | soft, rounded, balanced, sharp, uncertain |
| `compression` | dynamic, moderate, compressed, squashed, uncertain |
| `sustain` | short, typical, long, uncertain |
| `reverb`, `delay`, `modulation` | `ProfileEffect` objects (see below) |
| `modulation_type` | none, chorus, flanger, phaser, tremolo, other, uncertain (default) |

`ProfileEffect` has `presence` (absent, possible, present, uncertain) and
`prominence` (none, subtle, moderate, prominent, uncertain). An absent effect must
have none prominence; a present effect cannot have none prominence. Profiles also
require timestamped `evidence` and a list of `uncertainties`. Evidence timestamps
must be finite, nonnegative and no greater than the selected clip/interval duration,
relative to its start. Unknown categories, contradictory effects and out-of-range
evidence fail with the existing `invalid_response` type, not a coerced category.

`TONE_PROFILE_INSTRUCTIONS` defines these axes in audible, neutral terms and is used
instead of the prose listening instructions. The same source-specific safeguards
are appended: local audio is 16 kHz mono; YouTube imagery, titles and visible gear
are not tone evidence. Silence, inaccessible audio and mix ambiguity require
uncertainty. Profiles do not infer equipment, map effects to an application,
render audio or implement UI.

All six `listen*` functions accept keyword-only `use_cache` and `request_attempts`.
`use_cache=False` bypasses both reads and writes for this call without clearing or
changing the shared cache. `use_cache=True` permits caching only when
`TONE_LISTENER_CACHE_ENTRIES` is nonzero. `request_attempts` caps generate attempts
including the original: `1` prohibits the YouTube retry; `2` permits at most the
existing deadline-aware retry on a 503 or transient connection/timeout error.
Local-only generation always has at most one attempt. `None` retains those
transport defaults. Other values raise `ValueError` before I/O. Invalid
`use_cache` types raise `TypeError`.

Profile calls default to `use_cache=False, request_attempts=1`, so independent
evaluation repeats make independent calls. Existing `listen`, `listen_comparison`,
`listen_youtube` and `listen_youtube_comparison` keep `use_cache=True,
request_attempts=None`. Options do not relax deadlines, inference-slot limits,
upload processing limits, validation or bounded cleanup; attempt settings do not
change response semantics and therefore do not change cache keys. These library
options are not exposed as MCP tool arguments, and no new MCP tool is registered.

`youtube_input` performs the same URL/interval validation as listening without
network or file I/O. It returns the canonical URL, normalized start/end seconds
and SDK part carrying interval metadata; invalid inputs raise `ValueError`.
`failure_info` is the shared allowlisted failure formatter for callers with their
own SDK operations. `stage` distinguishes upload/processing `ValueError` from
invalid model output; `exc=None` represents missing configuration. It returns the
same fields as `failure` and never returns provider payloads or messages.

## YouTube reference vs local recording

`compare_youtube_reference` sends the YouTube interval (clip A) and your
selected local clip (clip B, uploaded as 16 kHz mono) to Gemini in one request
and returns the same `listening` comparison as `compare_audio`, including
prioritized `suggested_adjustments`. `available_analyses` states what a source
combination supports:

| Reference A | Candidate B | `semantic_listening` | `rms_matching` | `spectral_deltas` |
|---|---|---|---|---|
| Local file (`compare_audio`) | Local file | yes | yes | yes |
| YouTube interval (`compare_youtube_reference`) | Local file | yes | no | no |

Objective matching and deltas need the reference waveform locally, and the
YouTube reference is never downloaded or decoded; `unavailable_analyses`
explains why. The result reports `reference`, B's local `metrics_b`,
`input_transport: "youtube_url+files"` and cautions. A and B are not
loudness-matched, and the model is told that codec, bandwidth, channel and mix
differences are not tone differences. Semantic listening still requires a key.
