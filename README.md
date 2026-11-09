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

## Local tone workbench

The optional web workbench runs on this Mac and serves its built React UI and
FastAPI endpoints from one loopback origin. It does not deploy to a cloud
service or expose the Gemini key to the browser. With a mode-0600 local
credential file at `~/.config/tone-listener-mcp/env`, the web launcher loads
the allowlisted key/model settings automatically; an already configured
`GEMINI_API_KEY` environment variable takes precedence.

The actual preview requires macOS, the installed **Soundshed Guitar 1.7.3**
VST3, and a local build of the small Steinberg VST3 SDK host. Download the
Steinberg VST3 SDK separately, review its license, and keep it outside this
repository. From the repository root, build and launch:

```sh
python3 -m pip install -e '.[dev,web]'
npm --prefix web ci
npm --prefix web run build
cmake -S native/vst3-renderer -B native/vst3-renderer/build \
  -DVST3_SDK_ROOT=/absolute/path/to/vst3sdk -DCMAKE_BUILD_TYPE=Release
cmake --build native/vst3-renderer/build --target tone-vst3-renderer
tone-listener-web --port 8765 --model gemini-3.6-flash
```

Open `http://127.0.0.1:8765/`. The service binds only to `127.0.0.1`; Host and
Origin checks plus a local session token protect its API. Use the displayed
consent step before sending selected YouTube intervals or local audio clips to
Gemini. YouTube duration comes from the official embedded player; if unavailable,
enter bounded intervals yourself. Local audio is decoded with the existing
bounded decoder and kept in private app-owned storage.
Provider failures retain a sanitized category and `Retry-After`, not SDK payloads,
keys or account identifiers. Daily quota exhaustion requires restored quota or
waiting for the provider's stated window; the workbench never silently changes models.

Before analysis, a no-provider request check (`POST /api/analysis/preflight`)
shows which intervals reuse saved listening, which join an identical analysis
already running, and the maximum number of new Gemini requests. Saved results
are reused only for the same YouTube video ID or uploaded file content, exact
interval bounds, model, and prompt/schema/preparation code; any change is a miss.
Only successful profiles are saved (never failures, 429s or synthetic fixtures).
Identical concurrent analyses share one provider request and its outcome, and
cancelling one waiting draft does not cancel the shared request. The analysis
job carries the confirmed `max_new_requests`; if reusable results changed, the
server returns 409 for reconfirmation rather than sending more requests. Reused
intervals are labelled with their original analysis time. Tick "Listen again"
to bypass saved results explicitly. The reference benchmark calls the listener
directly and never uses this reuse layer, so its independent repeats are unchanged.
Filesystem paths are redacted from public JSON, preset descriptions and service
logs. Startup atomically sanitizes legacy project metadata without changing its
revision or DSP controls.

Projects are saved in the OS application-support directory. Preview jobs use
the installed plugin in a sandbox that denies reads of profile databases/settings,
home-directory writes and network access, while allowing the adapter-resolved
NAM/IR resource files. The adapter reads resource records read-only, and builds
one explicit single-preset host state; it does not capture user mixer settings.
At startup, a mandatory actual-DSP canary checks +6 dB utility gain
(tolerance +/-0.1 dB), bypass-to-dry, each NAM/IR contribution and order changes.
Preview, playback and export are refused without this validation; old previews
without the verified worker/plugin hashes cannot be exported.
The chain is one serial path;
resource IDs, parameter definitions, plugin state and `.soundshed.preset`
packaging remain inside the Soundshed adapter. Preset exports reference the
existing local NAM/IR library and do not bundle or redistribute those assets.
The starting recipe uses one explicitly selected categorical reference profile.
Clean/crunch/high-gain profiles select different curated capture candidates and
Soundshed EQ/cutoff controls. Capture labels are recommendations, not claims about
the source player's gear; unavailable candidates and ambiguous effects become
positional notes rather than silent substitutions. The interval selector can
regenerate the chain from another stored profile without another Gemini request.
Failed analysis creates an unresolved draft, not a universal Amane preset.
The official YouTube API is confined to an opaque sandboxed child frame; its
third-party script cannot read the app's session token or call its local API.
Playback uses bundled, separately licensed dry DI samples; monitor level and
dry/wet blend do not modify the saved preset. Riff 1/2 (the defaults, chordal
rhythm parts) are CC0 recordings from
[dry-guitar-samples](https://github.com/danieljfarrell/dry-guitar-samples).
Lead 1/2 (an upper-register lead line and a single-note pedal riff) are
excerpts of the **Guitar-TECHS** dataset by Hegel Pedroza, Wallace Abreu,
Ryan M. Corey and Iran R. Roman ([Zenodo record 14963133](https://zenodo.org/records/14963133)),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): Player 3
direct-input excerpts `directinput_08.wav` and `directinput_10.wav`, one channel
of the identical dual-mono pair, trimmed and resampled from 48 to 44.1 kHz
without normalization or gain (see `assets/di/provenance.json` and
`scripts/prepare_lead_dis.py`). Cite: Pedroza, Hegel, et al. "Guitar-TECHS: An
Electric Guitar Dataset Covering Techniques, Musical Excerpts, Chords and Scales
Using a Diverse Array of Hardware." ICASSP 2025, IEEE,
doi:10.1109/ICASSP49660.2025.10887996. Choose any sample in the preview and in
Refine/Fine-tune; each selected sample adds one disclosed judge request. Build output is placed
inside the Python package for the local launcher and is ignored by Git.
Previews are float WAVs, preserving DSP levels without PCM clipping. Up to ten
seconds of silence are processed for effect tails; a tail that is still audible
at that limit fails explicitly instead of saving truncated audio.

**Refine** (optional, paid) renders a bounded candidate set from the current
chain, uploads RMS-matched (-20 dBFS) copies with a fresh exact-input preview
and consent, and asks a relative judge which candidate sits closest to the
analysed reference interval. Choose the candidate set in the panel:

- **Explore** (default) swaps in alternative category recipes plus fixed
  pushed-clean and brighter/darker variants. Recipe names are alternatives, not
  a measured gain ordering: a "cleaner" recipe uses different gear and can be
  judged *more* saturated than the current chain.
- **Fine-tune** keeps the current gear, order, notes and analysis identity and
  only moves real parameters. Individual moves: the first enabled NAM capture's
  input trim (-6/-3/+3/+6 dB) or an upper-EQ tilt (parametric 2 kHz + 8 kHz,
  +/-3 dB each; capture treble when no EQ is present). Joint moves apply the
  +/-6 dB trim and the +/-3 dB tilt together (four sign combinations). Values are
  clamped to parameter bounds, no-op moves and identical chains are dropped (a
  joint move that clamps to a single-axis chain is listed once), and the
  wrong-direction control is kept. A full set is current + 10 moves + control,
  so each relative request uploads up to 12 clips.

Both modes show the candidate count, uploads per request and maximum request
count before consent.
The search mode is part of the plan identity, so changing it requires a new
preflight. A suggestion requires usable, control-passing evidence on every
planned DI; adopting a candidate is explicit and never edits your manual
settings silently. A parameter move is not itself a tonal improvement claim, and
a relative ranking is not a perceptual match. To refine further, adopt a
candidate and run another Fine-tune round with its own plan and consent.
Stored refinement renders and exported presets stay unchanged native output.
Refinement audition playback serves separate level-matched copies (equal RMS
across the candidate set with a shared peak ceiling), so a louder candidate is not
favoured by volume; the model uploads are separately RMS matched to -20 dBFS.
Neither copy is written back to the stored WAVs or presets.

Engine evidence for Fine-tune (no Gemini calls) renders each move with both DIs,
reporting full-channel output headroom separately from labelled mono-downmix tonal
measurements, and delivering byte copies of the renderer snapshot WAVs:

```sh
python3 scripts/verify_local_search.py --project path/to/project.json --output /private/tmp/local-search
```

Offline checks:

```sh
ruff check .
pytest -q
npm --prefix web run build
```

Opt-in **installed-engine** checks (no Gemini calls):

```sh
TONE_WORKBENCH_NATIVE_TESTS=1 pytest -q -m native_integration
# Or retain variant WAVs, a preset, hashes and the exact measurements:
python3 scripts/verify_workbench_renderer.py --live --output /private/tmp/tone-render-report
```

These checks restore a parsed exported preset into a fresh installed engine
instance and compare actual audio. They do **not** exercise the standalone
Soundshed app's ZIP importer UI or modify its preset library. Soundshed 1.7.3
ignores the legacy `mixer.masterGain` host-state field in this test; the explicit
global output gain and measured equal-power stereo pan law define preview level.
Standalone import remains **not verified**: the installed Mac app uses JUCE's
native shared profile location, and a sandboxed Foundation probe confirmed that
changing `HOME` does not redirect it. Home-wide write denial protects the live
profile and legacy container migration sources, but does not provide a writable
temporary profile. A second instance could also interact with the running app's
single-instance dispatch. No standalone launch/import was attempted without
proven profile and instance isolation; a protected native-import acceptance
environment is still required.

The opt-in live-reference benchmark is separate from offline CI. It freezes
categorical rubric, schema, prompts, model, recipe/engine hashes, complete grader
and benchmark source hashes, and source
intervals before its first paid request, then preserves all six requested cases:

```sh
python3 scripts/eval_workbench_references.py --live --model gemini-3.6-flash \
  --split development --start 0 --end 30 --output /private/tmp/tone-reference-report
python3 scripts/eval_workbench_references.py --live --model gemini-3.6-flash \
  --split held-out --start 0 --end 30 --output /private/tmp/tone-reference-report
```

The held-out step refuses changed recipe/model/rubric settings. It does not
reclassify tuned cases as untouched. Each supported candidate uses both real DI
phrases, independently judged with neutral filenames and no target/artist/gear
information. Opposite-saturation controls and repeated reference/preview
judgments calibrate the evaluator. API failures, uncertain attribution and
unsupported effects remain blocked/inconclusive, never passes. The report records
raw judgments, actual previews/presets, hashes, axis differences and inference
call counts; no credentials are saved. Same-model agreement is only a tone-profile
regression signal, not proof of perceptual identity or general validity.
An interrupted `running` case stops with an explicit recovery instruction rather
than being silently skipped. A current 429 pauses further benchmark requests and
retains unfinished cases; honor the recorded `Retry-After` before an explicitly
labelled new run. Raw reference reuse is opt-in with `--reuse-references DIR`,
requires matching frozen source/model/schema/prompts, and records prior calls
separately from new calls.

For bounded development of a stable lead interval, use a **new** output directory:

```sh
python3 scripts/eval_workbench_references.py --live --model gemini-3.6-flash \
  --split development --case case-01 --start 19.2 --end 30 \
  --output /private/tmp/tone-amane-lead-development
```

This is development, not a fresh untouched held-out claim. Do not change the rubric
to excuse mismatches or tune generation using held-out tonal evidence.
Bounded `--case` selection is development-only and cannot bypass held-out
prerequisites. Reference-reuse accounting includes only consumed judgments.

For interval-selection UI acceptance without paid calls:

```sh
python3 scripts/create_workbench_fixture.py
# Separate invented 429/Retry-After example for diagnostics UI:
python3 scripts/create_workbench_fixture.py --diagnostics
```

This creates a prominently labelled **offline synthetic fixture** with three
invented profiles. Regeneration makes no Gemini request. Its UI and exports never
present it as real listening or tone-match evidence; previews still use actual
licensed DI through the validated installed engine.

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
      "800_2000": {"available": true, "relative_db": 0.0}
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
`2000-5000`, `5000-8000` and `>8000` Hz. `relative_db` is band power relative
to total spectral power, not an absolute EQ control value.
Each band's `effective_range_hz` reports its Nyquist-clipped range. Comparisons
across different Nyquist frequencies omit relative band deltas and EQ direction
hints because their total-power denominators cover different bandwidths.

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

B is RMS-matched to A **before** spectral comparison and listening. The response
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
diagnosis, exact knob settings or proof that EQ will recreate A. Silence makes RMS
matching undefined and is explicitly reported. Stereo deltas require two stereo
clips. Gemini compares both uploads in one request and returns differences,
which clip is brighter/more saturated/more compressed/more ambient, suggested
adjustments and uncertainties.

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
GitHub Actions runs lint and tests on Python 3.11 and 3.12 on push or manual
dispatch. The opt-in `scripts.eval_listening` harness runs blinded, swapped A/B
judgments with separate tuning/held-out sets; see the validation guide for
request budgets and credential loading. Live Gemini checks require a key and
incur API usage; skipped runs do not measure listening accuracy.

## License

MIT. No reference recordings or copyrighted song audio are included.
