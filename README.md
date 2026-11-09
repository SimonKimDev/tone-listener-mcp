# tone-listener-mcp

Give your AI agent a **public YouTube guitar reference link** and ask how to
approach its tone. `analyze_youtube` sends a bounded segment directly to Gemini
for structured listening: gain, attack, compression, frequency character,
effects, timestamped evidence and uncertainty. No manual audio preparation is
needed for this primary workflow.

Use the observations to guide relative gain, EQ and effects adjustments in
software such as Soundshed Guitar, not to guess exact gear or knob settings.
The YouTube workflow **does not download/extract a WAV, isolate guitar stems,
or compute local DSP metrics**. It requires Gemini inference access.

Local-file measurement and RMS-matched comparison are also available when you
have recordings. These retain local metrics even without a Gemini key.

## YouTube quickstart

After installation and MCP registration below, ask your agent to call:

```json
{
  "url": "https://www.youtube.com/watch?v=401iS5McuoU",
  "start": 0,
  "end": 30,
  "question": "Describe the audible guitar gain, pick attack and effects without using video imagery."
}
```

The tool is **`analyze_youtube`**. Short public `youtu.be` links and YouTube
watch/shorts/embed/live video URLs are accepted and normalized to a single video.
Channel/playlist URLs, private/unlisted videos and arbitrary web URLs are not
supported. Tracking and URL timestamp parameters are discarded; specify
`start`/`end` explicitly. Omitting `end` selects at most
`TONE_LISTENER_MAX_SECONDS` (600 by default) after `start`, not an unlimited video.

Only audible evidence should support the response; the model is instructed not
to infer tone from images, titles, captions or visible equipment. Full mixes can
still confuse guitar with backing tracks, vocals, drums and mastering. Evidence
timestamps are relative to the selected segment. Public availability, model
capacity and account quota still affect success; failures are explicit rather
than invented listening results. See [validation](docs/validation.md) for the
real-link check and remaining limitations.

## Install

Requires Python **3.11 or newer**. Run from this repository:

```sh
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Runtime dependencies are MCP, NumPy, SciPy, SoundFile and google-genai, plus
Pydantic for response validation and HTTPX for typed provider-error handling.
No librosa or ffmpeg is required. SoundFile wheels bundle libsndfile, including
MP3 support. WAV, FLAC, OGG and MP3 are supported when the installed decoder can
read them. Convert M4A/AAC to WAV with your DAW or another converter first.

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Unset | Enable Gemini uploads and listening; omit for local-only operation |
| `TONE_LISTENER_MODEL` | `gemini-3.8-flash` | Audio-capable Gemini model available to your account |
| `TONE_LISTENER_MAX_SECONDS` | `600` | Maximum selected duration; positive finite seconds |

`.env.example` is a reference, **not automatically loaded**. Export variables in
your shell or set them in your MCP client's environment configuration. Never
commit a real key. Model availability, API access and billing depend on your
Google account. The default follows the current [Gemini audio documentation](https://ai.google.dev/gemini-api/docs/audio);
the backend uses the SDK Files upload and `models.generate_content` APIs.

The real YouTube reference check used `TONE_LISTENER_MODEL=gemini-3.6-flash`
because 3.8 was returning high-demand 503 responses. Set that environment value
to use the tested alternative, provided it is available to your account.
The tool reports the actual model and does not silently switch models.

```sh
export GEMINI_API_KEY="your-key"
.venv/bin/tone-listener-mcp
```

The server speaks MCP over stdio, not an interactive terminal interface.

## Register with an MCP client

Generic `mcpServers` configuration (replace the absolute path):

```json
{
  "mcpServers": {
    "tone-listener": {
      "command": "/absolute/path/to/tone-listener-mcp/.venv/bin/tone-listener-mcp",
      "env": {
        "GEMINI_API_KEY": "your-key"
      }
    }
  }
}
```

VS Code `.vscode/mcp.json` uses `servers` and supports a secret input:

```json
{
  "inputs": [
    {
      "type": "promptString",
      "id": "gemini-key",
      "description": "Gemini API key",
      "password": true
    }
  ],
  "servers": {
    "tone-listener": {
      "type": "stdio",
      "command": "/absolute/path/to/tone-listener-mcp/.venv/bin/tone-listener-mcp",
      "env": {
        "GEMINI_API_KEY": "${input:gemini-key}"
      }
    }
  }
}
```

For offline use, omit `GEMINI_API_KEY` and the VS Code secret input. Paths to audio
are resolved on the machine running this server, not necessarily the client machine.

## Tools

All tools return structured JSON as well as MCP-compatible content. Invalid files,
URLs or intervals return an MCP tool error. Local-file times are seconds; `start` defaults to zero,
`end` is exclusive and defaults to EOF. An endpoint past EOF is clamped to EOF.
`~` is expanded, and original channels/sample rate are retained for local metrics.
Only mono/stereo audio up to 192 kHz is accepted. Selected float32 decoded audio is
limited to 128 MiB in addition to the duration cap; shorten clips when either
limit is exceeded.

| Tool | Arguments | Result |
|---|---|---|
| `server_info` | None | Server name and version |
| `analyze_youtube` | `url`, optional `start`, `end`, `question` | Primary public YouTube listening; `metrics: null`, no local audio extraction |
| `audio_metrics` | `path`, optional `start`, `end` | Local clip metadata, level, spectrum and stereo metrics; never uploads |
| `analyze_audio` | `path`, optional `start`, `end`, `question` | Local `metrics` plus Gemini `listening`, or `listening_unavailable` |
| `compare_audio` | `path_a`, `path_b`, optional `start_a`, `end_a`, `start_b`, `end_b`, `focus` | Reference A vs candidate B, RMS matching, deltas, EQ directions and optional A/B listening |

Example requests to an agent's MCP tool interface:

```json
{"path": "/recordings/reference.wav", "start": 12, "end": 20}
```

```json
{
  "path_a": "/recordings/reference.wav",
  "path_b": "/recordings/candidate.wav",
  "focus": "How should I change B's brightness and drive to approach A?"
}
```

### Local metrics

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

### Listening

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
and `listening_failure_stage` (preparation, client creation, upload, processing,
generation or validation) to distinguish actionable failures without exposing
API headers or response bodies.
**Confidence is model self-assessment, not calibrated accuracy.** Live listening
quality requires evaluation against recordings with known changes.

### Comparison

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
hints. Silence makes RMS
matching undefined and is explicitly reported. Stereo deltas require two stereo
clips. Gemini compares both uploads in one request and returns differences,
which clip is brighter/more saturated/more compressed/more ambient, suggested
adjustments and uncertainties.

When the clips have different Nyquist frequencies, their relative energies use
different normalization ranges. Relative band deltas are then `null` and the
EQ direction hints list is empty.

## Privacy and limitations

With a key configured, `analyze_youtube` sends the public video URL and selected
interval to Google, which processes the source remotely. It does not extract a
local waveform or persist a media file. Public video processing is subject to
Google's supported YouTube-input rules, quotas and terms.

`analyze_audio` uploads the selected clip to Google,
and `compare_audio` uploads both selected clips. Only use recordings you have
permission to share. `audio_metrics` is always local. Temporary WAVs are deleted
on exit; uploaded files are deleted after the request, including most failure
paths. Cleanup failures are returned in `cleanup_warnings`. Remote handling and
retention are governed by Google's terms; deletion is not a zero-retention guarantee.
Local paths are included in local result metadata but not intentionally sent in
provider prompts. A user-supplied question/focus is sent to Google.

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

Full mixes can confuse cymbals with guitar fizz, doubling with delay and other
instruments with guitar ambience/compression. Isolate the guitar where possible.
The tool cannot identify exact amps, pedals, cabinets or knob settings. Spectral
differences may reflect notes, pickup position, playing or arrangement rather
than EQ. Gemini's descriptions may be wrong; use evidence and listening checks.

## Recording and validation

Use the **same DI and phrase** for A/B, avoid clipping, compare at matched
loudness and choose short, focused clips. Record dry and processed versions when
possible; start/end should cover equivalent musical material.

```sh
.venv/bin/ruff check .
.venv/bin/pytest -q
.venv/bin/python scripts/make_validation_set.py --output /tmp/tone-validation
```

See [controlled validation](docs/validation.md) for known gain, soft clipping,
EQ and simple reverb variants. Tests are synthetic/offline with mocked Gemini;
the stdio integration test spawns the real server and exercises all five tools.
GitHub Actions runs lint and tests on Python 3.11 and 3.12 for every pull
request (including forks), pushes to `main` and manual dispatch; the `ci-passed`
job is the required status check for merging into `main`. The opt-in `scripts.eval_listening` harness runs blinded, swapped A/B
judgments with separate tuning/held-out sets; see the validation guide for
request budgets and credential loading. Live Gemini checks require a key and
incur API usage; skipped runs do not measure listening accuracy.

## License

MIT. No reference recordings or copyrighted song audio are included.
