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
Nyquist are reported. Band IDs cover `<80`, `80-250`, `250-800`, `800-2000`,
`2000-5000`, `5000-8000` and `>8000` Hz. Each band's `effective_range_hz`
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
(`matching: "active"`) only gated, non-silent regions are matched: unweighted 50 ms
windows above -70 dBFS and within 20 dB of their mean. Silence padding therefore does
not change the applied gain, and the crest-factor delta uses the gated RMS. Use
`matching: "whole"` for whole-clip RMS that includes silence and tails.
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

`failure` is `null` on success, otherwise `{category, http_status, retryable, suggestion}`.
Categories: `not_configured`, `audio_preparation`, `authentication` (401), `permission` (403),
`model_unavailable` (404), `invalid_request` (400), `quota` (429), `capacity` (5xx),
`provider_error`, `network`, `deadline`, `busy`, `upload_failed`, `invalid_response` and
`local_io`. Suggestions are fixed text; provider messages are never returned.

One deadline (`TONE_LISTENER_REQUEST_DEADLINE_SECONDS`) bounds waiting for an inference slot,
each upload, upload processing (also capped at 30 s) and generation: every provider call's
HTTP timeout is the remaining budget. Remote upload cleanup always runs afterwards, even past
the deadline, with its own 10 s timeout per file, so a slow cleanup can extend a call slightly.
When all `TONE_LISTENER_MAX_CONCURRENT_REQUESTS` slots stay busy until the deadline, the
result fails with category `busy` at stage `queue` without uploading anything.

Every listening result includes `diagnostics`: `elapsed_seconds`, `deadline_seconds`,
`max_concurrent_requests`, `prompt_version`, `schema_version` (response schema name plus a
hash of its JSON schema), `cache` (`hit`, `miss`, `disabled` or `not_used` when no request
was attempted), provider token `usage` when reported, and `preprocessing` per uploaded clip
(original sample rate/channels/duration and the 16 kHz mono upload format).

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
