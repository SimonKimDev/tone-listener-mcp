# tone-listener-mcp

An MCP server that lets AI agents **listen to guitar tone**. Give your agent a
public YouTube reference and it describes gain, attack, compression, EQ
character and effects with timestamped evidence; give it your own recording and
it tells you what to change first. Use the answers to guide relative gain, EQ and
effects moves in software such as Soundshed Guitar, not to identify exact gear or
knob settings.

Listening uses Gemini (`GEMINI_API_KEY`). Local-file measurement and
loudness-matched A/B comparison work offline without a key.

## Install

Requires Python **3.11 or newer**. From a checkout of this repository:

macOS / Linux:

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

The executable is `.venv/bin/tone-listener-mcp` on macOS/Linux and
`.venv\Scripts\tone-listener-mcp.exe` on Windows. Contributors can instead
reproduce the pinned development environment with
[uv](https://docs.astral.sh/uv/): `uv sync --locked --extra dev` (see
[Development](#development)).

No ffmpeg or librosa is needed. WAV, FLAC, OGG and MP3 are read via SoundFile;
convert M4A/AAC to WAV first.

## Configure

Set variables in your MCP client's server environment (below) or your shell.
`.env.example` documents them but is **not loaded automatically**. Never commit a
real key.

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Unset | Enable Gemini uploads and listening; omit for local-only operation |
| `TONE_LISTENER_MODEL` | `gemini-3.8-flash` | Audio-capable Gemini model available to your account |
| `TONE_LISTENER_MAX_SECONDS` | `600` | Maximum selected duration; positive finite seconds |
| `TONE_LISTENER_REQUEST_DEADLINE_SECONDS` | `180` | Overall budget for queue, upload, processing and generation (>= 1) |
| `TONE_LISTENER_MAX_CONCURRENT_REQUESTS` | `2` | Gemini requests run at once per server process; others wait within the deadline |
| `TONE_LISTENER_CACHE_ENTRIES` | `64` | Successful listening results kept in memory; `0` disables the cache |

To run the server directly from a shell (it waits for an MCP client on stdio):

```sh
export GEMINI_API_KEY="your-key"           # PowerShell: $env:GEMINI_API_KEY = "your-key"
.venv/bin/tone-listener-mcp                # Windows: .venv\Scripts\tone-listener-mcp.exe
```

Invalid values make listening tools return an error naming the variable.
Model availability, quota and billing depend on your Google account; the tool
reports the model it used and never silently switches. The real YouTube check
used `gemini-3.6-flash` while 3.8 returned high-demand 503s.

## Register with an MCP client

Generic `mcpServers` configuration (use the absolute executable path from Install):

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

On Windows, escape backslashes in JSON:
`"command": "C:\\path\\to\\tone-listener-mcp\\.venv\\Scripts\\tone-listener-mcp.exe"`.

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

Omit `GEMINI_API_KEY` for offline use. The server speaks MCP over stdio; audio
paths are resolved on the machine running the server.

Ask your agent to call **`server_info`** to confirm the setup. It reports the
version, configured `model`, `listening_available` (with
`listening_unavailable` and `configuration_errors` explaining why not), the
duration cap and request settings. It never returns the key or calls Gemini.

## Example

Ask your agent to call `analyze_youtube`:

```json
{
  "url": "https://www.youtube.com/watch?v=401iS5McuoU",
  "start": 0,
  "end": 30,
  "question": "Describe the audible guitar gain, pick attack and effects without using video imagery."
}
```

Representative result (abridged; field values paraphrased from the
[real YouTube check](docs/validation.md#real-youtube-reference-check-2026-11-08),
not verbatim model output):

```json
{
  "model": "gemini-3.6-flash",
  "source": {
    "url": "https://www.youtube.com/watch?v=401iS5McuoU", "start_seconds": 0, "end_seconds": 30
  },
  "metrics": null,
  "listening": {
    "summary": "Clean to edge-of-breakup chords, then a brighter overdriven lead",
    "effects": {
      "reverb": {"presence": "present", "amount": "subtle", "notes": "short room/plate"},
      "delay": {"presence": "possible", "amount": "low", "notes": "may be doubling"},
      "modulation": {"presence": "absent", "amount": "none", "notes": ""}
    },
    "evidence": [{"timestamp": 19.2, "observation": "Overdriven lead enters"}],
    "uncertainties": ["Guitar highs may overlap cymbals and vocals in the mix"],
    "confidence": 0.6
  },
  "failure": null
}
```

Only audible evidence should support the answer; the model is told not to use
video imagery, titles or visible gear. Evidence timestamps are relative to the
selected segment. **Confidence is model self-assessment, not calibrated
accuracy.** Public `youtu.be` and watch/shorts/embed/live URLs are accepted;
playlists, channels, private videos and other sites are not. Omitting `end`
selects at most `TONE_LISTENER_MAX_SECONDS` after `start`.

## Tools

| Tool | Arguments | Result |
|---|---|---|
| `server_info` | None | Version, configured model, listening availability and settings; no API call |
| `analyze_youtube` | `url`, optional `start`, `end`, `question` | Primary public YouTube listening; `metrics: null`, no local audio extraction |
| `audio_metrics` | `path`, optional `start`, `end` | Local clip metadata, level, spectrum and stereo metrics; never uploads |
| `analyze_audio` | `path`, optional `start`, `end`, `question` | Local `metrics` plus Gemini `listening`, or `listening_unavailable` |
| `compare_audio` | `path_a`, `path_b`, optional `start_a`, `end_a`, `start_b`, `end_b`, `focus`, `matching` | Reference A vs candidate B, RMS matching, deltas, EQ directions and optional A/B listening |
| `compare_youtube_reference` | `url`, `path_b`, optional `start`, `end`, `start_b`, `end_b`, `focus` | YouTube reference A vs local candidate B: Gemini listening differences and prioritized adjustments only |

All tools return structured JSON with a published `outputSchema`. Listening
failures are typed (`failure.category`, `retryable`, `suggestion`) and every
listening result includes `diagnostics`. See the [tool reference](docs/reference.md)
for input rules, metric definitions, loudness matching, EQ hints, upload
preprocessing, failure categories, deadlines, caching and progress notifications.

## Benchmark status

| Check | Status | Details |
|---|---|---|
| DSP metrics, provider mocks, MCP stdio | Passing in CI (offline) | `pytest`; no key or network |
| Known gain/drive/EQ/reverb directions in local metrics | Verified offline | [validation](docs/validation.md) |
| Real YouTube reference listening | Spot-checked, 3 segments (2026-11-08) | Model judgments, unscored |
| Blinded synthetic A/B listening accuracy | **Not measured** | `scripts.eval_listening` |
| Real-recording listening accuracy | **Not measured** | `--manifest` harness ready |
| YouTube reference vs local candidate | **Not measured** live | Offline mocks only |

## Privacy and limitations

With a key configured, `analyze_youtube` sends the public video URL and selected
interval to Google, which processes the source remotely. It does not extract a
local waveform or persist a media file. Public video processing is subject to
Google's supported YouTube-input rules, quotas and terms.

`analyze_audio` uploads the selected clip to Google,
`compare_audio` uploads both selected clips, and `compare_youtube_reference`
sends the public video URL/interval plus the selected local clip. Only use recordings you have
permission to share. `audio_metrics` is always local. Temporary WAVs are deleted
on exit; uploaded files are deleted after the request, including most failure
paths. Cleanup failures are returned in `cleanup_warnings`. Remote handling and
retention are governed by Google's terms; deletion is not a zero-retention guarantee.
Local paths are included in local result metadata but not intentionally sent in
provider prompts. A user-supplied question/focus is sent to Google.

Full mixes can confuse cymbals with guitar fizz, doubling with delay and other
instruments with guitar ambience/compression. Isolate the guitar where possible.
The tool cannot identify exact amps, pedals, cabinets or knob settings. Spectral
differences may reflect notes, pickup position, playing or arrangement rather
than EQ. Gemini's descriptions may be wrong; use evidence and listening checks.

Local uploads are 16 kHz mono, so stereo and fine high-frequency detail come
from local metrics, not listening claims.

## Development

Use the **same DI and phrase** for A/B, avoid clipping, compare at matched
loudness and choose short, focused clips. Record dry and processed versions when
possible; start/end should cover equivalent musical material.

```sh
uv sync --locked --extra dev          # or: pip install -e '.[dev]'
uv run ruff check .
uv run pytest -q
uv run python scripts/make_validation_set.py --output /tmp/tone-validation
```

`uv.lock` pins the development environment; after changing dependencies in
`pyproject.toml`, run `uv lock` and commit both files. GitHub Actions runs
`uv lock --check`, lint and tests on Python 3.11 and 3.12 (plus Windows) for every
pull request (including forks), pushes to `main` and manual dispatch. It also
builds the wheel, installs it into a clean environment and smoke-tests the
installed executable over MCP stdio (`scripts/smoke_installed.py`). The
`ci-passed` job is the required status check for merging into `main`. Tests are synthetic/offline with mocked Gemini;
the stdio test spawns the real server and exercises all six tools.

See [controlled validation](docs/validation.md) for synthetic variants, the
opt-in blinded `scripts.eval_listening` harness and real-recording manifests
(`--manifest`). Live Gemini checks require a key and incur API usage; skipped
runs do not measure listening accuracy.

## License

MIT. No reference recordings or copyrighted song audio are included.
