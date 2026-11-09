"""Sandboxed standard-VST3 host for the installed Soundshed Guitar build."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.io import wavfile

from tone_listener_mcp.workbench.adapters.soundshed import (
    SUPPORTED_ENGINE_VERSION,
    SoundshedAdapter,
    new_project,
)
from tone_listener_mcp.workbench.models import (
    DI_SAMPLES,
    Interval,
    PreviewSnapshot,
    SourceRef,
    ToneProject,
)

SAMPLE_RATE = 44_100
MAX_RENDER_SECONDS = 30
MAX_TAIL_SECONDS = 10
_PLUGIN_PATH = Path("/Library/Audio/Plug-Ins/VST3/Soundshed Guitar.vst3")
_PROFILE_PATH = Path.home() / "Library" / "Application Support" / "Soundshed Guitar"
_LIBRARY_PROFILE_PATH = Path.home() / "Library" / "Soundshed Guitar"
_SANDBOX_PROFILE = (
    '(version 1)(allow default)'
    '(deny file-read* (subpath (param "PROFILE")))'
    '(deny file-read* (require-all (subpath (param "LIBRARY_PROFILE"))'
    ' (require-not (subpath (param "RESOURCES")))))'
    '(deny file-write* (subpath (param "HOME")))'
    '(deny network*)'
)


class RenderError(RuntimeError):
    """The actual installed Soundshed renderer could not produce a safe preview."""


@dataclass(frozen=True)
class RenderedAudio:
    samples: np.ndarray
    snapshot: PreviewSnapshot
    sample_name: str


class NativeRenderer:
    """Uses a fixed installed VST3, never a path or plug-in supplied by an API caller."""

    def __init__(
        self,
        adapter: SoundshedAdapter,
        binary_path: Path | None = None,
        plugin_path: Path = _PLUGIN_PATH,
    ) -> None:
        self.adapter = adapter
        self.binary_path = binary_path or (
            Path(__file__).resolve().parents[3]
            / "native"
            / "vst3-renderer"
            / "build"
            / "bin"
            / "tone-vst3-renderer"
        )
        self.plugin_path = plugin_path
        self._lock = threading.Lock()
        self._trailer: bytes | None = None
        self._engine_version: str | None = None
        self._canary: dict[str, float | str] | None = None
        self.initialization_error: str | None = None
        self._worker_sha256: str | None = None
        self._plugin_sha256: str | None = None
        self._validated_stamp: tuple[int, int, int, int] | None = None

    @property
    def verified(self) -> bool:
        if self._canary is None or self._trailer is None:
            return False
        try:
            unchanged = self._validated_stamp == self._engine_stamp()
        except OSError:
            unchanged = False
        if not unchanged:
            self.initialization_error = "Renderer or plugin files changed; restart and revalidate the engine."
        return unchanged

    def _engine_stamp(self) -> tuple[int, int, int, int]:
        worker = self.binary_path.stat()
        plugin = (self.plugin_path / "Contents" / "MacOS" / "Soundshed Guitar").stat()
        return worker.st_mtime_ns, worker.st_size, plugin.st_mtime_ns, plugin.st_size

    @property
    def canary_results(self) -> dict[str, float | str] | None:
        return dict(self._canary) if self._canary is not None else None

    def verifies_snapshot(self, snapshot: PreviewSnapshot) -> bool:
        return (
            self.verified
            and snapshot.dsp_conditions.get("state_canary_version") == 1
            and snapshot.dsp_conditions.get("worker_sha256") == self._worker_sha256
            and snapshot.dsp_conditions.get("plugin_sha256") == self._plugin_sha256
        )

    @property
    def available(self) -> bool:
        return (
            platform.system() == "Darwin"
            and self.plugin_path.is_dir()
            and self.binary_path.is_file()
            and os.access(self.binary_path, os.X_OK)
            and shutil_which("sandbox-exec") is not None
        )

    @property
    def engine_version(self) -> str:
        self.initialize()
        assert self._engine_version is not None
        return self._engine_version

    def initialize(self, cancel_event: threading.Event | None = None) -> None:
        _check_cancelled(cancel_event)
        if self.verified:
            return
        with self._lock:
            _check_cancelled(cancel_event)
            if self.verified:
                return
            if platform.system() != "Darwin":
                raise RenderError("The installed Soundshed renderer is available only on macOS.")
            if not self.plugin_path.is_dir():
                raise RenderError("Soundshed Guitar VST3 is not installed at the validated location.")
            if not self.binary_path.is_file() or not os.access(self.binary_path, os.X_OK):
                raise RenderError(
                    "The local VST3 renderer is not built. Follow the native renderer build steps in the README."
                )
            sandbox = shutil_which("sandbox-exec")
            if sandbox is None:
                raise RenderError("macOS sandbox-exec is required; rendering will not run without isolation.")
            with tempfile.TemporaryDirectory(
                prefix="tone-workbench-render-", dir="/private/tmp"
            ) as directory:
                scratch = Path(directory)
                os.chmod(scratch, 0o700)
                trailer_path = scratch / "trailer.bin"
                output = self._run_worker(
                    [
                        sandbox,
                        "-D",
                        f"HOME={Path.home()}",
                        "-D",
                        f"PROFILE={_PROFILE_PATH}",
                        "-D",
                        f"LIBRARY_PROFILE={_LIBRARY_PROFILE_PATH}",
                        "-D",
                        f"RESOURCES={self.adapter.resource_root}",
                        "-p",
                        _SANDBOX_PROFILE,
                        str(self.binary_path),
                        "--capture-trailer",
                        str(trailer_path),
                    ],
                    timeout=90,
                    cancel_event=cancel_event,
                )
                trailer = trailer_path.read_bytes()
                if not trailer.startswith(b"\0" * 8 + b"JUCEPrivateData") or not 32 <= len(trailer) <= 512:
                    raise RenderError("Soundshed returned an unsupported opaque VST3 state trailer.")
                match = re.search(r"\bengine=([A-Za-z0-9._-]+)\b", output)
                if not match:
                    raise RenderError("The installed Soundshed build did not report an engine version.")
                version = match.group(1)
                if version != SUPPORTED_ENGINE_VERSION:
                    raise RenderError(
                        f"Soundshed {version} is not validated; this renderer supports {SUPPORTED_ENGINE_VERSION}."
                    )
                self.adapter.set_engine_version(version)
                try:
                    canary = self._verify_state_application(trailer, version, cancel_event)
                except RenderError as exc:
                    self.initialization_error = str(exc)
                    raise
                self._worker_sha256 = _file_sha256(self.binary_path)
                self._plugin_sha256 = _file_sha256(
                    self.plugin_path / "Contents" / "MacOS" / "Soundshed Guitar"
                )
                self._validated_stamp = self._engine_stamp()
                self._trailer = trailer
                self._engine_version = version
                self._canary = canary
                self.initialization_error = None

    def _sandbox_arguments(self) -> list[str]:
        sandbox = shutil_which("sandbox-exec")
        if sandbox is None:
            raise RenderError("macOS sandbox-exec is required; rendering will not run without isolation.")
        return [
            sandbox, "-D", f"HOME={Path.home()}",
            "-D", f"PROFILE={_PROFILE_PATH}",
            "-D", f"LIBRARY_PROFILE={_LIBRARY_PROFILE_PATH}",
            "-D", f"RESOURCES={self.adapter.resource_root}",
            "-p", _SANDBOX_PROFILE, str(self.binary_path),
        ]

    def _render_raw(
        self, state: bytes, mono: np.ndarray, cancel_event: threading.Event | None = None
    ) -> np.ndarray:
        with tempfile.TemporaryDirectory(
            prefix="tone-workbench-render-", dir="/private/tmp"
        ) as directory:
            _check_cancelled(cancel_event)
            scratch = Path(directory)
            os.chmod(scratch, 0o700)
            state_path = scratch / "state.json"
            input_path = scratch / "input.f32"
            output_path = scratch / "output.f32"
            state_path.write_bytes(state)
            os.chmod(state_path, 0o600)
            np.ascontiguousarray(mono, dtype="<f4").tofile(input_path)
            os.chmod(input_path, 0o600)
            status = self._run_worker(
                [*self._sandbox_arguments(), str(state_path), str(input_path), str(output_path)],
                timeout=90, cancel_event=cancel_event,
            )
            if f"engine={SUPPORTED_ENGINE_VERSION} " not in status:
                raise RenderError("The native worker returned an unexpected engine version.")
            _check_cancelled(cancel_event)
            raw = np.fromfile(output_path, dtype="<f4")
        if raw.size != mono.size * 2:
            raise RenderError("Soundshed returned an unexpected audio length.")
        stereo = raw.reshape((-1, 2))
        if not np.isfinite(stereo).all() or float(np.max(np.abs(stereo))) > 64.0:
            raise RenderError("Soundshed returned invalid or unbounded audio.")
        return stereo

    def _verify_state_application(
        self, trailer: bytes, version: str, cancel_event: threading.Event | None
    ) -> dict[str, float | str]:
        """Measure DSP, not getState: JUCE may echo a pending unapplied restore."""
        path = Path(__file__).parent / "assets" / "di" / "riff-1.wav"
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE or samples.shape[1] != 1:
            raise RenderError("Renderer canary DI has an unsupported format.")
        mono = np.ascontiguousarray(samples[:3 * SAMPLE_RATE, 0], dtype="<f4")
        chain = self.adapter.self_test_chain()
        errors = self.adapter.validate_chain(chain)
        if errors or not all(any(node.kind == kind for node in chain.nodes) for kind in (
            "nam_fx", "nam_amp", "cab_ir"
        )):
            raise RenderError("Renderer self-test requires the validated local pedal NAM, amp NAM and IR.")
        project = new_project(
            source=SourceRef(
                kind="audio_file", display_name="Renderer self-test DI",
                duration_seconds=len(samples) / rate,
                selected_intervals=[Interval(start_seconds=0, end_seconds=3)],
            ),
            analysis=[{
                "interval": {"start_seconds": 0, "end_seconds": 3},
                "model": "none", "unavailable": "Local DSP self-test; no external analysis.",
            }],
            intent={"summary": "Renderer DSP self-test."}, chain=chain,
            engine_version=version, name="Renderer DSP self-test",
        )
        def run(candidate: ToneProject) -> np.ndarray:
            return self._render_raw(self.adapter.host_state(candidate, trailer), mono, cancel_event)

        bypass = project.model_copy(deep=True)
        for node in bypass.chain.nodes[1:-1]:
            node.enabled = False
        dry = run(bypass)
        # Soundshed's centred stereo mixer applies an equal-power pan law.
        expected = np.repeat(mono[:, None], 2, axis=1) / np.sqrt(2)
        dry_error = float(np.max(np.abs(dry - expected)))
        if dry_error > 1e-5:
            raise RenderError("Renderer cannot apply this chain: bypass self-test did not reproduce dry DI.")
        plus6 = bypass.model_copy(deep=True)
        plus6.chain.nodes[-1].params["gainDb"] = 6.0
        raised = run(plus6)
        gain_db = float(20 * np.log10(_rms(raised) / _rms(dry)))
        if abs(gain_db - 6.0) > 0.1:
            raise RenderError("Renderer cannot apply this chain: +6 dB state-sensitivity self-test failed.")
        wet = run(project)
        changes: dict[str, float | str] = {
            "utility_gain_delta_db": gain_db, "bypass_max_error": dry_error,
        }
        for kind in ("nam_fx", "nam_amp", "cab_ir"):
            candidate = project.model_copy(deep=True)
            next(node for node in candidate.chain.nodes if node.kind == kind).enabled = False
            relative = float(_rms(run(candidate) - wet) / _rms(wet))
            if relative < 0.01:
                raise RenderError(f"Renderer cannot apply this chain: {kind} resource self-test failed.")
            changes[f"{kind}_bypass_relative_difference"] = relative
        swapped = project.model_copy(deep=True)
        swapped.chain.nodes[1:3] = list(reversed(swapped.chain.nodes[1:3]))
        relative = float(_rms(run(swapped) - wet) / _rms(wet))
        if relative < 0.01:
            raise RenderError("Renderer cannot apply this chain: serial-order self-test failed.")
        changes["order_relative_difference"] = relative
        changes["sample_sha256"] = _file_sha256(path)
        return changes

    def render(
        self,
        project: ToneProject,
        sample_name: str,
        cache_root: Path,
        cancel_event: threading.Event | None = None,
    ) -> RenderedAudio:
        _check_cancelled(cancel_event)
        self.initialize(cancel_event)
        if sample_name not in DI_SAMPLES:
            raise RenderError("Choose one of the bundled dry DI samples.")
        if self._trailer is None or self._engine_version is None:
            raise RenderError("The Soundshed renderer has not initialized.")
        if self._engine_version != project.target.engine_version:
            raise RenderError("The project target version no longer matches the installed renderer.")
        errors = self.adapter.validate_chain(project.chain)
        if errors:
            raise RenderError("; ".join(errors))
        sample_path = Path(__file__).parent / "assets" / "di" / f"{sample_name}.wav"
        if not sample_path.is_file():
            raise RenderError("A bundled DI sample is missing.")
        source_samples, sample_rate = sf.read(
            sample_path, dtype="float32", always_2d=True
        )
        if sample_rate != SAMPLE_RATE or source_samples.shape[1] != 1:
            raise RenderError("Bundled DI sample has an unsupported format.")
        if not 0 < source_samples.shape[0] <= SAMPLE_RATE * MAX_RENDER_SECONDS:
            raise RenderError("Bundled DI sample exceeds the bounded render duration.")
        mono = np.ascontiguousarray(source_samples[:, 0], dtype="<f4")
        if not np.isfinite(mono).all() or float(np.max(np.abs(mono))) > 4.0:
            raise RenderError("Bundled DI sample contains invalid audio values.")
        source_sha256 = _file_sha256(sample_path)

        with self._lock:
            padded = np.pad(mono, (0, SAMPLE_RATE * MAX_TAIL_SECONDS))
            stereo = self._render_raw(self.adapter.host_state(project, self._trailer), padded, cancel_event)
        if float(np.max(np.abs(stereo[-SAMPLE_RATE // 4:]))) > 1e-5:
            raise RenderError(
                "The effect tail has not decayed within the 10-second render limit; "
                "reduce decay/feedback before previewing. No truncated preview was saved."
            )
        audible = np.flatnonzero(np.max(np.abs(stereo), axis=1) > 1e-6)
        retained = max(mono.size, int(audible[-1]) + 1 if audible.size else 0)
        stereo = stereo[:min(stereo.shape[0], retained + SAMPLE_RATE // 10)]
        rms = float(np.sqrt(np.mean(np.square(stereo, dtype=np.float64))))
        peak = float(np.max(np.abs(stereo)))
        if not np.isfinite(rms) or rms < 1e-8:
            raise RenderError("Soundshed returned a silent preview.")
        created_at = datetime.now(UTC).isoformat()
        snapshot = PreviewSnapshot(
            project_id=project.id,
            project_revision=project.revision,
            chain_revision=project.chain.revision,
            chain_sha256=self.adapter.chain_sha256(project.chain),
            sample_name=sample_name,
            source_sample_sha256=source_sha256,
            engine_version=self._engine_version,
            sample_rate=SAMPLE_RATE,
            dsp_conditions={
                "host": "Steinberg VST3 SDK",
                "nam_slimmable_size": 1.0,
                "nam_oversampling_index": 0,
                "nam_antialias_phase_index": 0,
                "calibration": "disabled per NAM node; shared profile untouched",
                "global_fx": "all explicitly bypassed",
                "global_input_gain_db": 0.0,
                "global_output_gain_db": 0.0,
                "host_master_gain": 1.0,
                "limiter_enabled": False,
                "sample_rate": SAMPLE_RATE,
                "state_canary_version": 1,
                "worker_sha256": self._worker_sha256,
                "plugin_sha256": self._plugin_sha256,
                "utility_canary_delta_db": self._canary["utility_gain_delta_db"],
                "tail_limit_seconds": MAX_TAIL_SECONDS,
                "tail_decay_peak_threshold": 1e-5,
                "input_duration_seconds": mono.size / SAMPLE_RATE,
            },
            audio_sha256=hashlib.sha256(stereo.tobytes()).hexdigest(),
            duration_seconds=stereo.shape[0] / SAMPLE_RATE,
            rms=rms,
            peak=peak,
            created_at=created_at,
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        wav_path = cache_root / f"{snapshot.id}.wav"
        try:
            wavfile.write(wav_path, SAMPLE_RATE, stereo.astype("<f4"))
            _check_cancelled(cancel_event)
        except (OSError, RuntimeError) as exc:
            raise RenderError("Could not save the bounded preview WAV.") from exc
        except RenderError:
            wav_path.unlink(missing_ok=True)
            raise
        os.chmod(wav_path, 0o600)
        snapshot = snapshot.model_copy(update={"audio_sha256": _file_sha256(wav_path)})
        return RenderedAudio(samples=stereo, snapshot=snapshot, sample_name=sample_name)

    @staticmethod
    def _run_worker(
        arguments: list[str],
        timeout: int,
        cancel_event: threading.Event | None = None,
    ) -> str:
        environment = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": "/private/tmp",
            "TMPDIR": "/private/tmp",
        }
        try:
            with tempfile.TemporaryFile() as output_file:
                process = subprocess.Popen(
                    arguments,
                    stdin=subprocess.DEVNULL,
                    stdout=output_file,
                    stderr=subprocess.DEVNULL,
                    env=environment,
                    start_new_session=True,
                )
                deadline = time.monotonic() + timeout
                while process.poll() is None:
                    if cancel_event is not None and cancel_event.is_set():
                        _kill_worker(process)
                        raise RenderError("Soundshed rendering was cancelled.")
                    if time.monotonic() >= deadline:
                        _kill_worker(process)
                        raise RenderError("Soundshed rendering exceeded the 90-second limit.")
                    try:
                        rss = subprocess.run(
                            ["/bin/ps", "-o", "rss=", "-p", str(process.pid)],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                            env=environment,
                            timeout=2,
                            check=False,
                        )
                    except (OSError, subprocess.TimeoutExpired) as exc:
                        _kill_worker(process)
                        raise RenderError("Could not enforce the Soundshed worker memory limit.") from exc
                    if process.poll() is None:
                        try:
                            resident_kib = int(rss.stdout.strip())
                        except ValueError as exc:
                            _kill_worker(process)
                            raise RenderError("Could not enforce the Soundshed worker memory limit.") from exc
                        if resident_kib > 4 * 1024 * 1024:
                            _kill_worker(process)
                            raise RenderError("Soundshed rendering exceeded the 4-GiB memory limit.")
                    time.sleep(0.25)
                process.wait()
                output_file.seek(0)
                stdout = output_file.read(4096)
        except OSError as exc:
            raise RenderError("Could not start the sandboxed Soundshed VST3 renderer.") from exc
        output = stdout[:4096].decode("utf-8", "replace")
        if process.returncode != 0:
            error = output.strip().splitlines()[:1]
            code = error[0].removeprefix("ERROR ") if error else "renderer-failed"
            safe_codes = {
                "invalid-arguments",
                "invalid-state-file",
                "invalid-single-preset-state",
                "invalid-audio-input",
                "soundshed-plugin-load-failed",
                "soundshed-effect-class-not-found",
                "soundshed-plugin-initialize-failed",
                "soundshed-state-restore-failed",
                "soundshed-controller-state-restore-failed",
                "soundshed-process-failed",
                "soundshed-activation-failed",
                "nonfinite-render-output",
                "silent-render-output",
                "unsupported-juce-state-trailer",
                "cannot-set-render-limits",
                "cannot-read-render-limits",
            }
            if code not in safe_codes:
                code = "renderer-failed"
            raise RenderError(f"Soundshed rendering failed ({code}).")
        return output


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _kill_worker(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _check_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RenderError("Soundshed rendering was cancelled.")


def shutil_which(name: str) -> str | None:
    """Resolve only a fixed system executable using a fixed PATH."""
    for directory in ("/usr/bin", "/bin", "/usr/sbin", "/sbin"):
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None
