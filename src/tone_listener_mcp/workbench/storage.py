"""Private app-owned project, upload and preview storage."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from tone_listener_mcp.workbench.models import DI_SAMPLES, PreviewSnapshot, ToneProject

UPLOAD_TTL_SECONDS = 30 * 60
PREVIEW_TTL_SECONDS = 24 * 60 * 60
PREVIEW_CACHE_LIMIT_BYTES = 512 * 1024 * 1024
_UUID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class StorageError(RuntimeError):
    """Local app storage could not be read or written safely."""


class ProjectConflict(StorageError):
    """A saved project advanced while its replacement was being prepared."""


@dataclass(frozen=True)
class UploadedSource:
    id: str
    path: Path
    display_name: str
    duration_seconds: float
    sample_rate: int
    channels: int
    file_size_bytes: int
    expires_at: float


@dataclass(frozen=True)
class CachedPreview:
    snapshot: PreviewSnapshot
    sample_name: str
    audio_path: Path


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise StorageError("App storage directory is not a regular directory.")
    if info.st_uid != os.getuid():
        raise StorageError("App storage directory is not owned by the current user.")
    os.chmod(path, 0o700)
    return path


def _data_root() -> Path:
    if os.name == "posix" and Path("/System/Library").exists():
        return Path.home() / "Library" / "Application Support" / "Tone Listener MCP" / "workbench"
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "tone-listener-mcp" / "workbench"


def _canonical_uuid(value: str) -> str:
    try:
        canonical = str(uuid.UUID(value))
    except (ValueError, AttributeError) as exc:
        raise ValueError("Invalid project or preview identifier") from exc
    if canonical != value.lower() or not _UUID_PATTERN.fullmatch(canonical):
        raise ValueError("Invalid project or preview identifier")
    return canonical


class LocalStorage:
    def __init__(self, root: Path | None = None) -> None:
        self.root = _private_directory((root or _data_root()).expanduser())
        self.projects = _private_directory(self.root / "projects")
        self.cache = _private_directory(self.root / "cache")
        self.uploads = _private_directory(self.root / "uploads")
        self.benchmarks = _private_directory(self.root / "benchmarks")
        self.analysis = _private_directory(self.root / "analysis")
        self._lock = threading.RLock()
        self._uploaded: dict[str, UploadedSource] = {}
        self._sweep_orphan_uploads()

    def save_project(self, project: ToneProject, expected_revision: int | None = None) -> None:
        project = ToneProject.model_validate(project.model_dump(mode="json"))
        identifier = _canonical_uuid(project.id)
        destination = self.projects / f"{identifier}.json"
        data = project.model_dump_json(indent=2).encode("utf-8") + b"\n"
        with self._lock:
            if expected_revision is not None:
                current = self.load_project(identifier)
                if current is None or current.revision != expected_revision:
                    raise ProjectConflict("Project changed; reload before saving.")
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=self.projects,
                    prefix=".project-",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    os.chmod(temporary, 0o600)
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
                os.chmod(destination, 0o600)
            except OSError as exc:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
                raise StorageError("Could not save the local project.") from exc

    def load_project(self, identifier: str) -> ToneProject | None:
        project_id = _canonical_uuid(identifier)
        path = self.projects / f"{project_id}.json"
        with self._lock:
            if not path.is_file() or path.is_symlink():
                return None
            try:
                return ToneProject.model_validate_json(path.read_bytes())
            except (OSError, ValidationError) as exc:
                raise StorageError("Saved local project is invalid or unreadable.") from exc

    def sanitize_saved_projects(self) -> dict[str, int]:
        """Migrate presentation metadata without changing revision or DSP controls."""
        checked = changed = 0
        with self._lock:
            for path in self.projects.glob("*.json"):
                if path.is_symlink() or not path.is_file() or not _UUID_PATTERN.fullmatch(path.stem):
                    continue
                try:
                    original = json.loads(path.read_bytes())
                    project = ToneProject.model_validate(original)
                except (OSError, ValueError) as exc:
                    raise StorageError("Could not sanitize an invalid saved project.") from exc
                checked += 1
                from .privacy import redact_paths
                if redact_paths(original) != original or any(
                    isinstance(segment.get("metrics"), dict)
                    and isinstance(segment["metrics"].get("clip"), dict)
                    and segment["metrics"]["clip"].get("source") != "Selected audio interval"
                    for segment in original.get("analysis", [])
                ):
                    self.save_project(project, expected_revision=project.revision)
                    changed += 1
        return {"checked": checked, "changed": changed}

    def list_projects(self, limit: int = 50) -> list[ToneProject]:
        projects = []
        with self._lock:
            for path in self.projects.glob("*.json"):
                if path.is_symlink() or not path.is_file() or not _UUID_PATTERN.fullmatch(path.stem):
                    continue
                try:
                    projects.append(ToneProject.model_validate_json(path.read_bytes()))
                except (OSError, ValidationError):
                    continue
        return sorted(projects, key=lambda project: project.updated_at, reverse=True)[:limit]

    def new_upload_path(self) -> tuple[str, Path]:
        identifier = str(uuid.uuid4())
        return identifier, self.uploads / f"upload-{identifier}.bin"

    def register_upload(
        self,
        identifier: str,
        path: Path,
        display_name: str,
        duration_seconds: float,
        sample_rate: int,
        channels: int,
    ) -> UploadedSource:
        identifier = _canonical_uuid(identifier)
        expected = self.uploads / f"upload-{identifier}.bin"
        if path != expected or path.is_symlink() or not path.is_file():
            raise StorageError("Uploaded source is not in app-owned storage.")
        info = path.stat()
        if info.st_uid != os.getuid():
            raise StorageError("Uploaded source is not owned by the current user.")
        os.chmod(path, 0o600)
        source = UploadedSource(
            id=identifier,
            path=path,
            display_name=display_name,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channels=channels,
            file_size_bytes=info.st_size,
            expires_at=time.time() + UPLOAD_TTL_SECONDS,
        )
        with self._lock:
            self._uploaded[identifier] = source
        return source

    def get_upload(self, identifier: str) -> UploadedSource | None:
        source_id = _canonical_uuid(identifier)
        with self._lock:
            source = self._uploaded.get(source_id)
        if source is None or source.expires_at <= time.time():
            if source:
                self.release_upload(source_id)
            return None
        if source.path.is_symlink() or not source.path.is_file():
            self.release_upload(source_id)
            return None
        return source

    def release_upload(self, identifier: str) -> None:
        source_id = _canonical_uuid(identifier)
        with self._lock:
            source = self._uploaded.pop(source_id, None)
        if source and source.path.parent == self.uploads:
            source.path.unlink(missing_ok=True)

    def _sweep_orphan_uploads(self) -> None:
        cutoff = time.time() - UPLOAD_TTL_SECONDS
        for path in self.uploads.iterdir():
            if (
                path.name.startswith("upload-")
                and path.name.endswith(".bin")
                and _UUID_PATTERN.fullmatch(path.name[7:-4])
                and not path.is_symlink()
                and path.is_file()
                and path.stat().st_mtime < cutoff
            ):
                path.unlink(missing_ok=True)


class PreviewCache:
    def __init__(
        self,
        directory: Path,
        max_bytes: int = PREVIEW_CACHE_LIMIT_BYTES,
        ttl_seconds: int = PREVIEW_TTL_SECONDS,
    ) -> None:
        self.directory = _private_directory(directory)
        self.max_bytes = max_bytes
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self.prune()

    def add(self, snapshot: PreviewSnapshot, sample_name: str) -> CachedPreview:
        identifier = _canonical_uuid(snapshot.id)
        audio = self.directory / f"{identifier}.wav"
        sidecar = self.directory / f"{identifier}.json"
        if audio.is_symlink() or not audio.is_file() or audio.parent != self.directory:
            raise StorageError("Preview audio is not in app-owned cache storage.")
        if _sha256(audio) != snapshot.audio_sha256:
            audio.unlink(missing_ok=True)
            raise StorageError("Preview audio changed before it could be cached.")
        record = {"snapshot": snapshot.model_dump(mode="json"), "sample_name": sample_name}
        temporary = self.directory / f".{identifier}.tmp"
        try:
            temporary.write_text(json.dumps(record, separators=(",", ":"), ensure_ascii=False))
            os.chmod(temporary, 0o600)
            os.replace(temporary, sidecar)
            os.chmod(sidecar, 0o600)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            audio.unlink(missing_ok=True)
            raise StorageError("Could not save preview metadata.") from exc
        self.prune()
        return CachedPreview(snapshot=snapshot, sample_name=sample_name, audio_path=audio)

    def get(self, identifier: str) -> CachedPreview | None:
        preview_id = _canonical_uuid(identifier)
        sidecar = self.directory / f"{preview_id}.json"
        audio = self.directory / f"{preview_id}.wav"
        with self._lock:
            if (
                sidecar.is_symlink()
                or audio.is_symlink()
                or not sidecar.is_file()
                or not audio.is_file()
            ):
                return None
            try:
                record = json.loads(sidecar.read_text())
                snapshot = PreviewSnapshot.model_validate(record["snapshot"])
                sample_name = record["sample_name"]
            except (OSError, KeyError, TypeError, json.JSONDecodeError, ValidationError):
                return None
            if (
                snapshot.id != preview_id
                or sample_name not in DI_SAMPLES
                or _sha256(audio) != snapshot.audio_sha256
            ):
                return None
            return CachedPreview(snapshot=snapshot, sample_name=sample_name, audio_path=audio)

    def for_project(self, project_id: str) -> list[CachedPreview]:
        project_id = _canonical_uuid(project_id)
        results = []
        for sidecar in self.directory.glob("*.json"):
            if sidecar.is_symlink() or not sidecar.is_file() or not _UUID_PATTERN.fullmatch(sidecar.stem):
                continue
            item = self.get(sidecar.stem)
            if item and item.snapshot.project_id == project_id:
                results.append(item)
        return sorted(results, key=lambda item: item.snapshot.created_at, reverse=True)

    def remove(self, identifier: str) -> None:
        preview_id = _canonical_uuid(identifier)
        for suffix in (".wav", ".json"):
            path = self.directory / f"{preview_id}{suffix}"
            if path.parent == self.directory and not path.is_symlink():
                path.unlink(missing_ok=True)

    def prune(self) -> None:
        with self._lock:
            entries = []
            now = time.time()
            for path in self.directory.iterdir():
                if path.is_symlink() or not path.is_file():
                    continue
                stem = path.stem
                if not _UUID_PATTERN.fullmatch(stem) or path.suffix not in (".wav", ".json"):
                    continue
                entries.append(path)
            groups: dict[str, list[Path]] = {}
            for path in entries:
                groups.setdefault(path.stem, []).append(path)
            for preview_id, paths in groups.items():
                age = max(now - path.stat().st_mtime for path in paths)
                if age > self.ttl_seconds:
                    self.remove(preview_id)
            active = sorted(
                (
                    path
                    for path in self.directory.iterdir()
                    if not path.is_symlink()
                    and path.is_file()
                    and _UUID_PATTERN.fullmatch(path.stem)
                    and path.suffix in (".wav", ".json")
                ),
                key=lambda path: path.stat().st_mtime,
            )
            total = sum(path.stat().st_size for path in active)
            while total > self.max_bytes and active:
                preview_id = active[0].stem
                related = [path for path in active if path.stem == preview_id]
                total -= sum(path.stat().st_size for path in related)
                self.remove(preview_id)
                active = [path for path in active if path.stem != preview_id]
            audio_ids = {
                path.stem for path in self.directory.glob("*.json")
                if not path.is_symlink() and path.is_file() and _UUID_PATTERN.fullmatch(path.stem)
            }
            for path in self.directory.glob("*.wav"):
                if (
                    not path.is_symlink()
                    and path.is_file()
                    and _UUID_PATTERN.fullmatch(path.stem)
                    and path.stem not in audio_ids
                ):
                    path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
