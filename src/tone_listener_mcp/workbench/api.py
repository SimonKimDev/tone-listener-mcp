"""Loopback-only local web API for the tone workbench."""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import math
import os
import secrets
import shutil
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote, urlsplit
from uuid import UUID, uuid4

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.responses import JSONResponse as BaseJSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from tone_listener_mcp.audio_io import AudioClip, load_audio, max_analysis_seconds
from tone_listener_mcp.listener import (
    DEFAULT_MODEL,
    listen_tone_profile,
    listen_youtube_tone_profile,
)
from tone_listener_mcp.metrics import measure
from tone_listener_mcp.workbench import refinement as refine
from tone_listener_mcp.workbench.adapters.soundshed import (
    AdapterError,
    SoundshedAdapter,
    new_project,
)
from tone_listener_mcp.workbench.analysis_cache import (
    AnalysisCache,
    CachedAnalysis,
    file_sha256,
    make_key,
)
from tone_listener_mcp.workbench.jobs import JobFailure, JobManager, JobQueueFull
from tone_listener_mcp.workbench.models import (
    DEFAULT_DI_SAMPLES,
    DI_SAMPLES,
    AdapterCatalog,
    AnalysisReuse,
    AnalysisSegment,
    ChainDraft,
    ChainNode,
    DiSample,
    Interval,
    JobStatus,
    PreviewSnapshot,
    SourceRef,
    TargetBinding,
    ToneIntent,
    ToneProject,
)
from tone_listener_mcp.workbench.privacy import redact_paths
from tone_listener_mcp.workbench.renderer import NativeRenderer, RenderError
from tone_listener_mcp.workbench.sources import automatic_intervals, safe_display_name
from tone_listener_mcp.workbench.storage import (
    LocalStorage,
    PreviewCache,
    ProjectConflict,
    StorageError,
)


class JSONResponse(BaseJSONResponse):
    def render(self, content) -> bytes:
        return super().render(redact_paths(content))


MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_UPLOAD_REQUEST_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024
MAX_JSON_REQUEST_BYTES = 2 * 1024 * 1024
MAX_SOURCE_DURATION = 86_400
_STATIC_ROOT = Path(__file__).resolve().parent / "static"


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnalysisSourceRequest(APIModel):
    kind: Literal["youtube", "audio_file"]
    youtube_url: str | None = Field(default=None, max_length=512)
    upload_id: str | None = None
    duration_seconds: float | None = Field(default=None, gt=0, le=MAX_SOURCE_DURATION)
    intervals: list[Interval] = Field(min_length=1, max_length=3)
    refresh: bool = False

    @model_validator(mode="after")
    def validate_source_fields(self) -> AnalysisSourceRequest:
        if self.kind == "youtube":
            if not self.youtube_url or self.upload_id is not None:
                raise ValueError("YouTube analysis requires only a YouTube URL.")
        elif not self.upload_id or self.youtube_url is not None:
            raise ValueError("Audio analysis requires only a valid uploaded-source token.")
        previous_end = -1.0
        for interval in sorted(self.intervals, key=lambda value: value.start_seconds):
            if interval.start_seconds < previous_end:
                raise ValueError("Selected analysis intervals must not overlap.")
            if self.duration_seconds is not None and interval.end_seconds > self.duration_seconds:
                raise ValueError("Selected analysis interval exceeds the known source duration.")
            previous_end = interval.end_seconds
        return self


class AnalysisRequest(AnalysisSourceRequest):
    external_consent: bool = False
    name: str | None = Field(default=None, min_length=1, max_length=120)
    # Upper bound on new provider generation requests the user confirmed from the preflight.
    max_new_requests: int | None = Field(default=None, ge=0, le=3)

    @model_validator(mode="after")
    def require_consent(self) -> AnalysisRequest:
        if not self.external_consent:
            raise ValueError("Explicit consent is required before sending a reference to Gemini.")
        return self


class IntervalPlan(APIModel):
    interval: Interval
    status: Literal["cache", "shared_in_flight", "new"]
    analyzed_at: str | None = None


class AnalysisPreflight(APIModel):
    model: str
    refresh: bool
    intervals: list[IntervalPlan]
    reused: int
    shared_in_flight: int
    max_new_requests: int
    provider_contacted: Literal[False] = False


class ChainUpdateRequest(APIModel):
    base_revision: int = Field(ge=1)
    name: str | None = Field(default=None, min_length=1, max_length=120)
    nodes: list[ChainNode] = Field(min_length=2, max_length=16)


class PreviewRequest(APIModel):
    project_revision: int = Field(ge=1)
    chain_revision: int = Field(ge=1)
    sample_name: DiSample


class ProfileSelectionRequest(APIModel):
    base_revision: int = Field(ge=1)
    analysis_index: int = Field(ge=0, le=2)


class RefinementPreflightRequest(APIModel):
    project_revision: int = Field(ge=1)
    chain_revision: int = Field(ge=1)
    model: Literal["gemini-3.6-flash", "gemini-4-argon"] = "gemini-3.6-flash"
    samples: list[DiSample] = Field(default_factory=lambda: list(DEFAULT_DI_SAMPLES), min_length=1, max_length=len(DI_SAMPLES))
    # "local": same gear/topology, bounded input-trim/EQ steps around the current (e.g. adopted) chain.
    search: Literal["explore", "local"] = "explore"
    # Uploaded-audio projects only: a fresh re-upload of the SAME analyzed file (verified by content SHA-256).
    upload_id: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def unique_samples(self) -> RefinementPreflightRequest:
        if len(set(self.samples)) != len(self.samples):
            raise ValueError("Each DI sample may be judged once per run.")
        return self


class RefinementRequest(RefinementPreflightRequest):
    plan_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    max_new_requests: int = Field(ge=1, le=len(DI_SAMPLES))
    external_consent: bool

    @model_validator(mode="after")
    def require_consent(self) -> RefinementRequest:
        if not self.external_consent:
            raise ValueError("Explicit consent is required before sending renders to Gemini.")
        return self


class AdoptRequest(APIModel):
    base_revision: int = Field(ge=1)
    variant_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")


class ExportRequest(APIModel):
    preview_id: str


class HealthResponse(APIModel):
    status: Literal["ready"]
    model_configured: bool
    model: str
    engine_version: str
    renderer_binary_available: bool
    renderer_available: bool
    renderer_error: str | None
    samples: list[str]


class UploadResponse(APIModel):
    upload_id: str
    display_name: str
    duration_seconds: float
    sample_rate: int
    channels: int
    file_size_bytes: int
    intervals: list[Interval]
    expires_in_seconds: int


class ProjectSummary(APIModel):
    id: str
    name: str
    updated_at: str
    source_kind: Literal["youtube", "audio_file"]
    source_name: str


class APIError(RuntimeError):
    """Known local API failure safe to display."""


def create_app(
    storage: LocalStorage | None = None,
    adapter: SoundshedAdapter | None = None,
    renderer: NativeRenderer | None = None,
    allowed_hosts: set[str] | None = None,
    job_manager: JobManager | None = None,
    allow_test_client: bool = False,
    refinement_judge=None,
) -> FastAPI:
    local_storage = storage or LocalStorage()
    soundshed = adapter or SoundshedAdapter()
    analysis_cache = AnalysisCache(local_storage.analysis)
    upload_digests: dict[str, str] = {}
    native_renderer = renderer or NativeRenderer(soundshed)
    previews = PreviewCache(local_storage.cache)
    jobs = job_manager or JobManager()
    judge = refinement_judge or refine.relative_judge.judge_relative
    refinements = _private_refinement_root(local_storage)
    # project_id -> exact reservation token; upload_id -> token of the refinement that leases it.
    active_refinements: dict[str, object] = {}
    upload_leases: dict[str, object] = {}
    refinement_lock = threading.Lock()
    authority_allowlist = {
        authority.lower()
        for authority in (allowed_hosts or {"127.0.0.1:8765", "localhost:8765"})
    }

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            await asyncio.to_thread(native_renderer.initialize)
        except RenderError as exc:
            native_renderer.initialization_error = str(exc)
        yield
        jobs.close()

    app = FastAPI(
        title="Tone Listener Workbench",
        version="1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
        default_response_class=JSONResponse,
    )
    app.state.storage = local_storage
    app.state.adapter = soundshed
    app.state.renderer = native_renderer
    app.state.previews = previews
    app.state.jobs = jobs
    app.state.request_token = secrets.token_urlsafe(32)
    app.state.allowed_hosts = authority_allowlist

    @app.exception_handler(StarletteHTTPException)
    async def public_http_error(_: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_: Request, exc: RequestValidationError):
        # Do not echo request bodies, filesystem-looking IDs or non-finite JSON.
        return JSONResponse(status_code=422, content={
            "detail": [
                {"type": error["type"], "msg": error["msg"]}
                for error in exc.errors()
            ]
        })

    @app.middleware("http")
    async def local_request_guard(request: Request, call_next):
        authority = request.headers.get("host", "").lower()
        if authority not in authority_allowlist:
            return Response("Local workbench accepts loopback Host headers only.", status_code=403)
        client_host = request.client.host if request.client else ""
        if client_host == "testclient" and not allow_test_client:
            return Response("Local workbench is available from this Mac only.", status_code=403)
        if client_host != "testclient":
            try:
                if not ipaddress.ip_address(client_host).is_loopback:
                    return Response("Local workbench is available from this Mac only.", status_code=403)
            except ValueError:
                return Response("Local workbench is available from this Mac only.", status_code=403)

        origin = request.headers.get("origin")
        if origin:
            parsed = urlsplit(origin)
            if (
                parsed.scheme != "http"
                or parsed.netloc.lower() not in authority_allowlist
                or parsed.path not in ("", "/")
                or parsed.query
                or parsed.fragment
                or parsed.username is not None
                or parsed.password is not None
            ):
                return Response("Cross-origin requests are not allowed.", status_code=403)

        if request.url.path.startswith("/api") and request.url.path != "/api/session":
            cookie = request.cookies.get("tone_session", "")
            supplied = request.headers.get("x-workbench-token", "")
            if not (
                cookie
                and supplied
                and hmac.compare_digest(cookie, app.state.request_token)
                and hmac.compare_digest(supplied, app.state.request_token)
            ):
                return Response("Local workbench request authentication failed.", status_code=403)

        if request.method in ("POST", "PUT", "PATCH"):
            maximum = (
                MAX_UPLOAD_REQUEST_BYTES
                if request.url.path == "/api/uploads"
                else MAX_JSON_REQUEST_BYTES
            )
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > maximum:
                        return Response("Request body exceeds the local size limit.", status_code=413)
                except ValueError:
                    return Response("Invalid request Content-Length.", status_code=400)

        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "frame-src 'self' https://www.youtube.com https://www.youtube-nocookie.com; "
            "media-src 'self' blob:; connect-src 'self' https://www.youtube.com "
            "https://www.youtube-nocookie.com; object-src 'none'; base-uri 'self'; "
            "frame-ancestors 'self'",
        )
        return response

    @app.get("/api/session")
    async def session(response: Response) -> dict[str, str]:
        response.set_cookie(
            "tone_session",
            app.state.request_token,
            httponly=True,
            secure=False,
            samesite="strict",
            max_age=8 * 60 * 60,
            path="/",
        )
        return {"token": app.state.request_token}

    @app.get("/api/health")
    async def health() -> HealthResponse:
        return {
            "status": "ready",
            "model_configured": bool(os.environ.get("GEMINI_API_KEY")),
            "model": os.environ.get("TONE_LISTENER_MODEL", DEFAULT_MODEL),
            "engine_version": soundshed.engine_version,
            "renderer_binary_available": native_renderer.binary_path.is_file(),
            "renderer_available": native_renderer.verified,
            "renderer_error": native_renderer.initialization_error,
            "samples": list(DI_SAMPLES),
        }

    @app.get("/api/catalog", response_model=AdapterCatalog)
    def catalog() -> AdapterCatalog:
        try:
            return soundshed.catalog()
        except AdapterError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/api/projects")
    def list_projects() -> list[ProjectSummary]:
        return [
            ProjectSummary(
                id=project.id,
                name=project.name,
                updated_at=project.updated_at,
                source_kind=project.source.kind,
                source_name=project.source.display_name,
            )
            for project in local_storage.list_projects()
        ]

    @app.get("/api/projects/{project_id}")
    def get_project(project_id: str) -> ToneProject:
        return _load_project(local_storage, project_id)

    @app.post("/api/uploads")
    async def upload_audio(file: Annotated[UploadFile, File()]) -> UploadResponse:
        identifier, destination = local_storage.new_upload_path()
        written = 0
        try:
            with destination.open("xb") as output:
                os.chmod(destination, 0o600)
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="Audio upload exceeds 100 MiB.")
                    output.write(chunk)
            if written == 0:
                raise HTTPException(status_code=422, detail="Uploaded audio is empty.")
            try:
                info = sf.info(destination)
            except (OSError, RuntimeError, sf.LibsndfileError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail="Could not read the uploaded audio; use a supported audio file.",
                ) from exc
            duration = info.frames / info.samplerate if info.samplerate else 0
            if (
                info.channels not in (1, 2)
                or info.samplerate <= 0
                or info.samplerate > 192_000
                or info.frames <= 0
                or not math.isfinite(duration)
                or duration <= 0
                or duration > MAX_SOURCE_DURATION
            ):
                raise HTTPException(
                    status_code=422,
                    detail="Audio must be mono or stereo, at most 192 kHz and 24 hours.",
                )
            uploaded = local_storage.register_upload(
                identifier,
                destination,
                safe_display_name(file.filename),
                duration,
                info.samplerate,
                info.channels,
            )
            return UploadResponse(
                upload_id=uploaded.id,
                display_name=uploaded.display_name,
                duration_seconds=uploaded.duration_seconds,
                sample_rate=uploaded.sample_rate,
                channels=uploaded.channels,
                file_size_bytes=uploaded.file_size_bytes,
                intervals=automatic_intervals(duration),
                expires_in_seconds=30 * 60,
            )
        except HTTPException:
            destination.unlink(missing_ok=True)
            raise
        except (OSError, StorageError) as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Could not safely store uploaded audio.") from exc
        finally:
            await file.close()

    @app.delete("/api/uploads/{upload_id}", status_code=204)
    def discard_upload(upload_id: str) -> Response:
        try:
            local_storage.release_upload(upload_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Uploaded audio was not found.") from exc
        upload_digests.pop(upload_id, None)
        return Response(status_code=204)

    def _resolve_analysis_source(body: AnalysisSourceRequest):
        """Validate the source without provider contact; return it with its cache identity."""
        try:
            maximum_duration = max_analysis_seconds()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if any(
            interval.end_seconds - interval.start_seconds > maximum_duration
            for interval in body.intervals
        ):
            raise HTTPException(
                status_code=422,
                detail=f"Selected clip exceeds {maximum_duration:g} seconds.",
            )
        uploaded = None
        source_url = None
        duration_seconds = body.duration_seconds
        if body.kind == "audio_file":
            try:
                uploaded = local_storage.get_upload(body.upload_id or "")
            except ValueError:
                uploaded = None
            if uploaded is None:
                upload_digests.pop(body.upload_id or "", None)
                raise HTTPException(status_code=410, detail="Uploaded audio expired; upload it again.")
            display_name = uploaded.display_name
            duration_seconds = uploaded.duration_seconds
            if any(interval.end_seconds > duration_seconds for interval in body.intervals):
                raise HTTPException(status_code=422, detail="Selected interval exceeds decoded audio duration.")
            # Content identity: the same bytes under another name or upload token reuse results.
            identity = upload_digests.get(uploaded.id)
            if identity is None:
                identity = upload_digests[uploaded.id] = file_sha256(uploaded.path)
        else:
            from tone_listener_mcp.listener import _youtube_input

            try:
                source_url, _, _, _ = _youtube_input(
                    body.youtube_url or "",
                    body.intervals[0].start_seconds,
                    body.intervals[0].end_seconds,
                )
            except (ValueError, TypeError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if duration_seconds is not None and any(
                interval.end_seconds > duration_seconds for interval in body.intervals
            ):
                raise HTTPException(status_code=422, detail="Selected interval exceeds the known video duration.")
            identity = source_url.rsplit("=", 1)[-1]
            display_name = f"YouTube video {identity}"
        return uploaded, source_url, display_name, duration_seconds, identity

    def _plan_analysis(body: AnalysisSourceRequest, identity: str, model: str) -> AnalysisPreflight:
        plans = []
        for interval in body.intervals:
            key = make_key(body.kind, identity, interval, model)
            cached = None if body.refresh else analysis_cache.get(key)
            if cached is not None:
                plans.append(IntervalPlan(interval=interval, status="cache", analyzed_at=cached.analyzed_at))
            elif analysis_cache.in_flight(key):
                plans.append(IntervalPlan(interval=interval, status="shared_in_flight"))
            else:
                plans.append(IntervalPlan(interval=interval, status="new"))
        return AnalysisPreflight(
            model=model,
            refresh=body.refresh,
            intervals=plans,
            reused=sum(plan.status == "cache" for plan in plans),
            shared_in_flight=sum(plan.status == "shared_in_flight" for plan in plans),
            max_new_requests=sum(plan.status == "new" for plan in plans),
        )

    @app.post("/api/analysis/preflight", response_model=AnalysisPreflight)
    def preflight_analysis(body: AnalysisSourceRequest) -> AnalysisPreflight:
        _, _, _, _, identity = _resolve_analysis_source(body)
        return _plan_analysis(body, identity, os.environ.get("TONE_LISTENER_MODEL", DEFAULT_MODEL))

    @app.post("/api/jobs/analysis", response_model=JobStatus, status_code=202)
    def start_analysis(body: AnalysisRequest) -> JobStatus:
        if not os.environ.get("GEMINI_API_KEY"):
            raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured locally.")
        uploaded, source_url, display_name, duration_seconds, identity = _resolve_analysis_source(body)
        model = os.environ.get("TONE_LISTENER_MODEL", DEFAULT_MODEL)
        request_budget = len(body.intervals) if body.max_new_requests is None else body.max_new_requests
        if _plan_analysis(body, identity, model).max_new_requests > request_budget:
            raise HTTPException(
                status_code=409,
                detail="Reusable results changed since you confirmed; review the new request count and confirm again.",
            )

        source = SourceRef(
            kind=body.kind,
            url=source_url,
            display_name=display_name,
            duration_seconds=duration_seconds,
            selected_intervals=body.intervals,
            content_sha256=identity if body.kind == "audio_file" else None,
        )
        project_name = body.name or display_name[:120]

        def release_uploaded_source() -> None:
            if uploaded is not None:
                local_storage.release_upload(uploaded.id)
                upload_digests.pop(uploaded.id, None)

        def work(progress, cancelled: threading.Event) -> dict[str, object]:
            engine_version = soundshed.engine_version
            segments = []
            unavailable_count = 0
            provider_pause = None
            paused_model = None
            new_requests = 0
            try:
                native_renderer.initialize(cancelled)
                for index, interval in enumerate(source.selected_intervals):
                    if cancelled.is_set():
                        raise JobFailure("Analysis cancelled.")
                    progress(0.1 + 0.7 * index / len(source.selected_intervals),
                             f"Analyzing interval {index + 1} of {len(source.selected_intervals)}.")
                    metrics = None
                    clip = None
                    if source.kind == "audio_file":
                        current_upload = local_storage.get_upload(uploaded.id if uploaded else "")
                        if current_upload is None:
                            raise JobFailure("Uploaded audio expired before analysis started.")
                        clip = _load_uploaded_interval(current_upload.path, interval)
                        clip = AudioClip(
                            samples=clip.samples, sample_rate=clip.sample_rate,
                            source=source.display_name, start_seconds=clip.start_seconds,
                        )
                        metrics = measure(clip)
                    key = make_key(source.kind, identity, interval, model)
                    cached = None if body.refresh else analysis_cache.get(key)
                    if cached is not None:
                        segments.append(AnalysisSegment(
                            interval=interval, model=cached.model, profile=cached.profile,
                            provider_attempted=False, metrics=metrics,
                            reuse=AnalysisReuse(kind="cache", analyzed_at=cached.analyzed_at),
                        ))
                        continue

                    def request(interval=interval, clip=clip, key=key) -> dict:
                        if clip is None:
                            outcome = listen_youtube_tone_profile(
                                source.url or "", interval.start_seconds, interval.end_seconds,
                            )
                        else:
                            outcome = listen_tone_profile(clip)
                        if isinstance(outcome.get("listening"), dict):
                            try:
                                analysis_cache.put(key, outcome["listening"], str(outcome.get("model", model)))
                            except (OSError, ValueError):
                                pass  # an unreusable result is still returned to this job
                        return outcome

                    mode, result = analysis_cache.run_shared(
                        key, request, cancelled,
                        may_start=provider_pause is None and new_requests < request_budget,
                        use_cache=not body.refresh,
                    )
                    if mode == "cancelled":
                        raise JobFailure("Analysis cancelled.")
                    if isinstance(result, CachedAnalysis):
                        segments.append(AnalysisSegment(
                            interval=interval, model=result.model, profile=result.profile,
                            provider_attempted=False, metrics=metrics,
                            reuse=AnalysisReuse(kind="cache", analyzed_at=result.analyzed_at),
                        ))
                        continue
                    if mode == "blocked" or result is None:
                        unavailable_count += 1
                        if provider_pause is not None:
                            retry_after = provider_pause.get("retry_after_seconds")
                            reason = "Not attempted: provider paused after an earlier HTTP 429." + (
                                f" Honor Retry-After: {retry_after:g} seconds."
                                if isinstance(retry_after, (int, float)) else ""
                            )
                        else:
                            reason = ("Not attempted: reusable results changed after confirmation and this "
                                      "would exceed the confirmed request count. Review and confirm again.")
                        segments.append(AnalysisSegment(
                            interval=interval, model=paused_model or model, provider_attempted=False,
                            metrics=metrics, unavailable=reason,
                        ))
                        continue
                    reuse = None
                    if mode == "fresh":
                        new_requests += 1
                    else:
                        reuse = AnalysisReuse(kind="shared_in_flight", analyzed_at=datetime.now(UTC).isoformat())
                    listening = result.get("listening")
                    unavailable = result.get("listening_unavailable")
                    counts = result.get("request_counts")
                    attempted = any(counts.get(name, 0) > 0 for name in ("generate", "upload")) if isinstance(counts, dict) else None
                    if reuse is not None:
                        attempted = False
                    failure_info = result.get("failure_info")
                    if isinstance(failure_info, dict) and failure_info.get("http_status") == 429:
                        provider_pause = failure_info
                        paused_model = str(result.get("model", model))
                    if not isinstance(listening, dict):
                        unavailable_count += 1
                        safe_unavailable = (
                            unavailable if isinstance(unavailable, str)
                            else "Listening result unavailable; no tone was inferred."
                        )
                        segments.append(
                            AnalysisSegment(
                                interval=interval,
                                model=str(result.get("model", model)),
                                unavailable=safe_unavailable[:500],
                                failure_info=failure_info,
                                provider_attempted=attempted,
                                reuse=reuse,
                                metrics=metrics,
                            )
                        )
                    else:
                        segments.append(
                            AnalysisSegment(
                                interval=interval,
                                model=str(result.get("model", model)),
                                profile=listening,
                                provider_attempted=attempted,
                                reuse=reuse,
                                metrics=metrics,
                            )
                        )
                if cancelled.is_set():
                    raise JobFailure("Analysis cancelled.")
                selected = next((
                    index for index, segment in enumerate(segments)
                    if segment.profile is not None and segment.profile.guitar_dominance == "clear"
                ), next((index for index, segment in enumerate(segments) if segment.profile is not None), 0))
                intent = _tone_intent(segments, selected)
                chain = soundshed.default_chain([segments[selected]])
                project = new_project(
                    source=source,
                    analysis=[segment.model_dump(mode="json") for segment in segments],
                    intent=intent.model_dump(mode="json"),
                    chain=chain,
                    engine_version=engine_version,
                    name=project_name,
                )
                project.selected_analysis_index = selected
                if cancelled.is_set():
                    raise JobFailure("Analysis cancelled.")
                local_storage.save_project(project)
                progress(1.0, "Analysis and initial local project saved.")
                return {
                    "_job_status": "partial" if unavailable_count else "completed",
                    "project": project.model_dump(mode="json"),
                }
            except (AdapterError, RenderError, StorageError) as exc:
                raise JobFailure(str(exc)) from exc
            finally:
                release_uploaded_source()

        try:
            return jobs.submit("analysis", work, on_abandon=release_uploaded_source)
        except JobQueueFull as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc

    @app.put("/api/projects/{project_id}/chain")
    def update_chain(project_id: str, body: ChainUpdateRequest) -> ToneProject:
        project = _load_project(local_storage, project_id)
        if body.base_revision != project.revision:
            raise HTTPException(status_code=409, detail="Project changed; reload before editing.")
        try:
            chain = ChainDraft(revision=project.chain.revision + 1, nodes=body.nodes)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Invalid single serial chain.") from exc
        errors = soundshed.validate_chain(chain, for_render=False)
        if errors:
            raise HTTPException(status_code=422, detail="; ".join(errors))
        target_assets = {
            node.id: node.asset for node in chain.nodes if node.asset is not None
        }
        updated = project.model_copy(
            update={
                "revision": project.revision + 1,
                "name": body.name or project.name,
                "chain": chain,
                "target": TargetBinding(
                    adapter_id=project.target.adapter_id,
                    engine_version=project.target.engine_version,
                    resources=target_assets,
                ),
                "updated_at": _now(),
            }
        )
        try:
            local_storage.save_project(updated, expected_revision=body.base_revision)
        except ProjectConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return updated

    @app.post("/api/projects/{project_id}/profile")
    def select_profile(project_id: str, body: ProfileSelectionRequest) -> ToneProject:
        project = _load_project(local_storage, project_id)
        if body.base_revision != project.revision:
            raise HTTPException(status_code=409, detail="Project changed; reload before selecting a profile.")
        if body.analysis_index >= len(project.analysis):
            raise HTTPException(status_code=422, detail="Selected interval does not exist.")
        chain = soundshed.default_chain([project.analysis[body.analysis_index]])
        chain.revision = project.chain.revision + 1
        updated = project.model_copy(update={
            "selected_analysis_index": body.analysis_index,
            "intent": _tone_intent(project.analysis, body.analysis_index),
            "chain": chain, "revision": project.revision + 1, "updated_at": _now(),
            "target": TargetBinding(
                adapter_id=project.target.adapter_id, engine_version=project.target.engine_version,
                resources={node.id: node.asset for node in chain.nodes if node.asset is not None},
            ),
        })
        try:
            local_storage.save_project(updated, expected_revision=body.base_revision)
        except ProjectConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return updated

    @app.post("/api/projects/{project_id}/jobs/preview", response_model=JobStatus, status_code=202)
    def start_preview(project_id: str, body: PreviewRequest) -> JobStatus:
        if not native_renderer.verified:
            raise HTTPException(
                status_code=503,
                detail=native_renderer.initialization_error or "Renderer self-test has not passed.",
            )
        project = _load_project(local_storage, project_id)
        _assert_revision(project, body.project_revision, body.chain_revision)

        def work(progress, cancelled: threading.Event) -> dict[str, object]:
            current = _load_project_for_job(local_storage, project_id)
            _check_job_revision(current, body.project_revision, body.chain_revision)
            progress(0.1, "Validating the current chain and installed Soundshed resources.")
            try:
                rendered = native_renderer.render(current, body.sample_name, local_storage.cache, cancelled)
                if cancelled.is_set():
                    previews.remove(rendered.snapshot.id)
                    raise JobFailure("Preview cancelled.")
                latest = _load_project_for_job(local_storage, project_id)
                if (
                    latest.revision != body.project_revision
                    or latest.chain.revision != body.chain_revision
                    or soundshed.chain_sha256(latest.chain) != rendered.snapshot.chain_sha256
                ):
                    previews.remove(rendered.snapshot.id)
                    return {"_job_status": "stale", "message": "Chain changed during render."}
                cached = previews.add(rendered.snapshot, rendered.sample_name)
                if cancelled.is_set():
                    previews.remove(rendered.snapshot.id)
                    raise JobFailure("Preview cancelled.")
            except (AdapterError, RenderError, StorageError, APIError) as exc:
                raise JobFailure(str(exc)) from exc
            progress(1.0, "Current chain preview rendered and cached.")
            return {
                "preview": cached.snapshot.model_dump(mode="json"),
                "_cancel_cleanup": lambda: previews.remove(cached.snapshot.id),
            }

        try:
            return jobs.submit("preview", work)
        except JobQueueFull as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc

    def _release_refinement_upload(upload_id: str | None, lease: object | None = None) -> None:
        """Delete a reselected upload unless another refinement's reservation holds it."""
        if upload_id:
            with refinement_lock:
                owner = upload_leases.get(upload_id)
                if owner is not None and owner is not lease:
                    return
                upload_leases.pop(upload_id, None)
            try:
                local_storage.release_upload(upload_id)
            except ValueError:
                pass
            upload_digests.pop(upload_id, None)

    def _refinement_upload(project: ToneProject, body: RefinementPreflightRequest, interval: Interval):
        """Verify a re-upload is byte-identical to the analyzed file; release it on any refusal."""
        if project.source.content_sha256 is None:
            _release_refinement_upload(body.upload_id)
            raise HTTPException(status_code=409, detail=(
                "This project was analyzed before source fingerprints were recorded, so a re-upload cannot be "
                "verified as the same file. Analyze the file again (matching cached intervals are reused without "
                "new requests), then refine the new project."))
        if body.upload_id is None:
            raise HTTPException(status_code=428, detail=(
                "Uploaded audio is not retained after analysis. Reselect the same file to compare against it; "
                "it is verified by content hash and deleted when the refinement ends."))
        try:
            uploaded = local_storage.get_upload(body.upload_id)
        except ValueError:
            uploaded = None
        if uploaded is None:
            raise HTTPException(status_code=410, detail="Reselected audio expired; select the file again.")
        digest = upload_digests.get(uploaded.id)
        if digest is None:
            digest = upload_digests[uploaded.id] = file_sha256(uploaded.path)
        if digest != project.source.content_sha256:
            _release_refinement_upload(uploaded.id)
            raise HTTPException(status_code=409, detail=(
                "The selected file is not the analyzed source (content differs); it was discarded. "
                "Select the exact same file."))
        if interval.end_seconds > uploaded.duration_seconds:
            _release_refinement_upload(uploaded.id)
            raise HTTPException(status_code=409, detail="The selected interval exceeds the reselected audio.")
        return uploaded

    def _refinement_plan(project: ToneProject, body: RefinementPreflightRequest):
        """Return the plan plus the verified reselected upload; any refusal discards that upload."""
        try:
            plan = _refinement_plan_inner(project, body)
        except HTTPException:
            _release_refinement_upload(body.upload_id)
            raise
        uploaded = None
        if project.source.kind == "audio_file":
            uploaded = local_storage.get_upload(body.upload_id or "")
        elif body.upload_id is not None:
            _release_refinement_upload(body.upload_id)
        return plan, uploaded

    def _refinement_plan_inner(project: ToneProject, body: RefinementPreflightRequest) -> refine.RefinementPlan:
        if project.revision != body.project_revision or project.chain.revision != body.chain_revision:
            raise HTTPException(status_code=409, detail="Project changed; reload before refining.")
        if project.synthetic_fixture:
            raise HTTPException(status_code=409, detail="Synthetic fixture projects have no real reference to compare.")
        if project.source.kind == "youtube" and not project.source.url:
            raise HTTPException(status_code=409, detail="The YouTube reference URL is missing.")
        segment = project.analysis[project.selected_analysis_index]
        if segment.profile is None:
            raise HTTPException(status_code=409, detail="The selected interval has no successful tone profile.")
        if segment.profile.guitar_dominance != "clear":
            raise HTTPException(status_code=409, detail=(
                "The selected interval's guitar is not clearly separable; comparisons would be unusable."))
        errors = soundshed.validate_chain(project.chain)
        if errors:
            raise HTTPException(status_code=409, detail="Resolve the chain before refining: " + "; ".join(errors))
        variants = refine.build_variants(soundshed, project.chain, segment.profile, body.search)
        if not any(item.role == "control" for item in variants) or not any(item.role == "variant" for item in variants):
            raise HTTPException(status_code=409, detail=(
                "No distinct local variant and wrong-tone control could be built from installed resources."))
        bounds = {"start_seconds": segment.interval.start_seconds, "end_seconds": segment.interval.end_seconds,
                  "analysis_index": project.selected_analysis_index}
        if project.source.kind == "audio_file":
            uploaded = _refinement_upload(project, body, segment.interval)
            reference = {"kind": "audio_file", "content_sha256": project.source.content_sha256,
                         "upload_id": uploaded.id, **bounds}
            uploads = len(variants) + 1
            sent = "the reselected reference audio interval"
            uploaded_note = True
        else:
            uploaded_note = False
            reference = {"kind": "youtube", "url": project.source.url, **bounds}
            uploads = len(variants)
            sent = "the public YouTube interval"
        base_sha = soundshed.chain_sha256(project.chain)
        identity = {
            "project_id": project.id, "project_revision": project.revision, "chain_sha256": base_sha,
            "model": body.model, "samples": body.samples, "reference": reference, "search": body.search,
            "variants": [[item.id, item.chain_sha256] for item in variants],
            "engine_version": soundshed.engine_version, "judge": refine.judge_identity(),
        }
        return refine.RefinementPlan(
            project_id=project.id, project_revision=project.revision, base_chain_sha256=base_sha,
            model=body.model, samples=body.samples, reference=reference, variants=variants, search=body.search,
            max_new_requests=len(body.samples), uploads_per_request=uploads,
            plan_sha256=refine.plan_sha256(identity),
            disclosure=(f"{refine.SEARCH_DESCRIPTIONS[body.search]} "
                        f"Sends {len(variants)} level-matched renders per DI sample plus {sent} "
                        f"({uploads} uploads per request) to {body.model}: at most {len(body.samples)} new "
                        "generation request(s), never retried or reused. Rendering, adopting, editing and export "
                        "make no requests."
                        + (" The reselected file is deleted when the refinement ends." if uploaded_note else "")),
        )

    @app.post("/api/projects/{project_id}/refinement/preflight", response_model=refine.RefinementPlan)
    def refinement_preflight(project_id: str, body: RefinementPreflightRequest) -> refine.RefinementPlan:
        try:
            project = _load_project(local_storage, project_id)
        except HTTPException:
            _release_refinement_upload(body.upload_id)
            raise
        # A verified reselected upload stays registered (TTL-bounded) for the confirmed run.
        return _refinement_plan(project, body)[0]

    @app.post("/api/projects/{project_id}/jobs/refinement", response_model=JobStatus, status_code=202)
    def start_refinement(project_id: str, body: RefinementRequest) -> JobStatus:
        try:
            return _start_refinement(project_id, body)
        except HTTPException:
            _release_refinement_upload(body.upload_id)
            raise

    def _start_refinement(project_id: str, body: RefinementRequest) -> JobStatus:
        if not native_renderer.verified:
            raise HTTPException(status_code=503,
                                detail=native_renderer.initialization_error or "Renderer self-test has not passed.")
        if not os.environ.get("GEMINI_API_KEY") and refinement_judge is None:
            raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured locally.")
        project = _load_project(local_storage, project_id)
        plan, uploaded = _refinement_plan(project, body)
        if plan.plan_sha256 != body.plan_sha256 or plan.max_new_requests > body.max_new_requests:
            raise HTTPException(status_code=409, detail=(
                "The refinement plan changed since you confirmed; review the new plan and confirm again."))
        if project.source.kind == "audio_file" and uploaded is None:
            raise HTTPException(status_code=410, detail="Reselected audio expired; select the file again.")
        reservation = object()
        with refinement_lock:
            if project_id in active_refinements:
                raise HTTPException(status_code=409, detail="A refinement for this project is already running.")
            if uploaded is not None and uploaded.id in upload_leases:
                raise HTTPException(status_code=409, detail="That reselected file is already in use.")
            active_refinements[project_id] = reservation
            if uploaded is not None:
                upload_leases[uploaded.id] = reservation

        def release_reservation() -> None:
            """Idempotent; only ever releases THIS reservation, never a newer job's."""
            if uploaded is not None:
                _release_refinement_upload(uploaded.id, reservation)
            with refinement_lock:
                if active_refinements.get(project_id) is reservation:
                    del active_refinements[project_id]

        record = refine.RefinementRecord(
            id=str(uuid4()), status="running", project_id=project.id, project_revision=project.revision,
            chain_revision=project.chain.revision, base_chain_sha256=plan.base_chain_sha256,
            plan_sha256=plan.plan_sha256, model=plan.model, reference=plan.reference, samples=plan.samples,
            search=plan.search,
            confirmed_max_new_requests=body.max_new_requests, created_at=_now(),
            variants=[refine.VariantSummary(id=item.id, role=item.role, label=item.label,
                                            description=item.description, chain_sha256=item.chain_sha256)
                      for item in plan.variants],
        )
        directory = refinements / record.id
        hot = refine.control_is_hotter(project.analysis[project.selected_analysis_index].profile)

        def persist() -> None:
            _write_private_json(directory / "record.json", record.model_dump(mode="json"))

        def work(progress, cancelled: threading.Event) -> dict[str, object]:
            try:
                directory.mkdir(mode=0o700)
                _write_private_json(directory / "chains.json",
                                    {item.id: item.chain.model_dump(mode="json") for item in plan.variants})
                persist()
                clips: dict[str, dict[str, AudioClip]] = {sample: {} for sample in plan.samples}
                total = len(plan.samples) * len(plan.variants)
                for sample in plan.samples:
                    for variant in plan.variants:
                        if cancelled.is_set():
                            raise JobFailure("Refinement cancelled before any request was sent.")
                        progress(0.05 + 0.45 * sum(map(len, clips.values())) / total,
                                 f"Rendering {variant.label} on {sample} locally.")
                        candidate = project.model_copy(update={"chain": variant.chain})
                        rendered = native_renderer.render(candidate, sample, directory, cancelled)
                        wav = directory / f"{variant.id}-{sample}.wav"
                        os.replace(directory / f"{rendered.snapshot.id}.wav", wav)
                        clips[sample][variant.id] = AudioClip(
                            samples=rendered.samples.T, sample_rate=rendered.snapshot.sample_rate,
                            source=f"{variant.id}-{sample}")
                        next(item for item in record.variants if item.id == variant.id).renders.append(
                            refine.VariantRender(sample=sample, audio_sha256=rendered.snapshot.audio_sha256,
                                                 duration_seconds=rendered.snapshot.duration_seconds,
                                                 rms=rendered.snapshot.rms, peak=rendered.snapshot.peak))
                persist()
                if uploaded is not None:
                    anchor = _load_uploaded_interval(uploaded.path, Interval(
                        start_seconds=plan.reference["start_seconds"], end_seconds=plan.reference["end_seconds"]))
                    reference_input = {"anchor_clip": anchor}
                else:
                    reference_input = {"youtube": (plan.reference["url"], plan.reference["start_seconds"],
                                                   plan.reference["end_seconds"])}
                paused = False
                for index, sample in enumerate(plan.samples):
                    if cancelled.is_set() or paused or record.new_generate_requests >= body.max_new_requests:
                        record.judgments.append(refine.SampleJudgment(
                            sample=sample, status="not_attempted",
                            reason="Provider paused after HTTP 429." if paused else "Cancelled before this request."))
                        continue
                    progress(0.5 + 0.45 * index / len(plan.samples),
                             f"Requesting one blinded {plan.model} comparison for {sample}.")
                    result = judge(model=plan.model, candidates=clips[sample], seed=secrets.randbelow(2**31),
                                   **reference_input)
                    summary = refine.summarize_judgment(sample, result, hot)
                    record.new_generate_requests += summary.generate_requests
                    record.new_upload_requests += summary.upload_requests
                    record.judgments.append(summary)
                    paused = summary.http_status == 429
                    persist()
                record.suggestion = refine.suggest_from(record.judgments)
                judged = sum(item.status == "judged" for item in record.judgments)
                record.status = ("cancelled" if cancelled.is_set() else
                                 "completed" if judged == len(plan.samples) else "partial")
                record.completed_at = _now()
                persist()
                _prune_refinements(refinements, protect={record.id})
                progress(1.0, "Refinement judgments saved.")
                return {"_job_status": "completed" if record.status == "completed" else "partial",
                        "refinement": record.model_dump(mode="json")}
            except (AdapterError, RenderError, StorageError, OSError, APIError) as exc:
                record.status = "failed"
                record.completed_at = _now()
                if directory.is_dir():
                    persist()
                raise JobFailure(str(exc)) from exc
            except Exception:
                record.status = "cancelled" if cancelled.is_set() else "failed"
                record.completed_at = _now()
                if directory.is_dir():
                    persist()
                raise
            finally:
                release_reservation()

        try:
            return jobs.submit("refinement", work, on_abandon=release_reservation)
        except JobQueueFull as exc:
            release_reservation()
            raise HTTPException(status_code=429, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}/refinements", response_model=list[refine.RefinementRecord])
    def list_refinements(project_id: str) -> list[refine.RefinementRecord]:
        _load_project(local_storage, project_id)
        records = []
        for path in refinements.glob("*/record.json"):
            try:
                item = refine.RefinementRecord.model_validate_json(path.read_text())
            except (OSError, ValueError):
                continue
            if item.project_id == project_id:
                records.append(item)
        return sorted(records, key=lambda item: item.created_at, reverse=True)[:5]

    def _refinement_record(project_id: str, refinement_id: str) -> refine.RefinementRecord:
        try:
            identifier = str(UUID(refinement_id))
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Refinement was not found.") from exc
        path = refinements / identifier / "record.json"
        try:
            item = refine.RefinementRecord.model_validate_json(path.read_text())
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Refinement was not found.") from exc
        if item.project_id != project_id:
            raise HTTPException(status_code=404, detail="Refinement was not found.")
        return item

    @app.get("/api/projects/{project_id}/refinements/{refinement_id}/audio/{variant_id}/{sample_name}")
    def refinement_audio(project_id: str, refinement_id: str, variant_id: str, sample_name: str):
        item = _refinement_record(project_id, refinement_id)
        variant = next((entry for entry in item.variants if entry.id == variant_id), None)
        render = variant and next((entry for entry in variant.renders if entry.sample == sample_name), None)
        if render is None:
            raise HTTPException(status_code=404, detail="Variant render was not found.")
        directory = refinements / item.id
        path = directory / f"{variant.id}-{render.sample}.wav"
        if not path.is_file() or path.is_symlink():
            raise HTTPException(status_code=404, detail="Variant render was not found.")
        # Audition is level-matched like the judge's input (full band, stereo): every variant of this sample
        # gets the same RMS with one shared headroom attenuation, so louder variants do not sound "better".
        siblings = {}
        for entry in item.variants:
            sibling = directory / f"{entry.id}-{render.sample}.wav"
            if any(r.sample == render.sample for r in entry.renders) and sibling.is_file() \
                    and not sibling.is_symlink():
                siblings[entry.id] = sibling
        try:
            audio = _audition_level_matched(siblings, variant.id)
        except (OSError, ValueError, sf.LibsndfileError) as exc:
            raise HTTPException(status_code=404, detail="Variant render could not be read.") from exc
        return Response(audio, media_type="audio/wav", headers={
            "Cache-Control": "no-store", "Content-Disposition": 'attachment; filename="refinement-variant.wav"',
            "X-Audition-Level": "rms-matched"})

    @app.post("/api/projects/{project_id}/refinements/{refinement_id}/adopt")
    def adopt_variant(project_id: str, refinement_id: str, body: AdoptRequest) -> ToneProject:
        project = _load_project(local_storage, project_id)
        item = _refinement_record(project_id, refinement_id)
        if body.base_revision != project.revision or project.revision != item.project_revision \
                or soundshed.chain_sha256(project.chain) != item.base_chain_sha256:
            raise HTTPException(status_code=409, detail=(
                "The chain changed after this refinement; its variants no longer apply. Refine again."))
        try:
            chains = json.loads((refinements / item.id / "chains.json").read_text())
            draft = ChainDraft.model_validate(chains[body.variant_id])
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=404, detail="Variant was not found.") from exc
        variant = next((entry for entry in item.variants if entry.id == body.variant_id), None)
        if variant is None or variant.role == "control" or soundshed.chain_sha256(draft) != variant.chain_sha256:
            raise HTTPException(status_code=409, detail="Only a recorded, non-control variant can be adopted.")
        chain = ChainDraft(revision=project.chain.revision + 1, nodes=draft.nodes)
        errors = soundshed.validate_chain(chain, for_render=False)
        if errors:
            raise HTTPException(status_code=422, detail="; ".join(errors))
        updated = project.model_copy(update={
            "revision": project.revision + 1, "chain": chain, "updated_at": _now(),
            "target": TargetBinding(
                adapter_id=project.target.adapter_id, engine_version=project.target.engine_version,
                resources={node.id: node.asset for node in chain.nodes if node.asset is not None}),
        })
        try:
            local_storage.save_project(updated, expected_revision=body.base_revision)
        except ProjectConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return updated

    @app.get("/api/jobs/{job_id}", response_model=JobStatus)
    def get_job(job_id: str) -> JobStatus:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Local job was not found.")
        return job

    @app.post("/api/jobs/{job_id}/cancel", response_model=JobStatus)
    def cancel_job(job_id: str) -> JobStatus:
        job = jobs.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Local job was not found.")
        return job

    @app.get("/api/projects/{project_id}/previews", response_model=list[PreviewSnapshot])
    def project_previews(project_id: str) -> list[PreviewSnapshot]:
        _load_project(local_storage, project_id)
        return [
            item.snapshot for item in previews.for_project(project_id)
            if native_renderer.verifies_snapshot(item.snapshot)
        ]

    @app.get("/api/projects/{project_id}/previews/{preview_id}/audio")
    def preview_audio(project_id: str, preview_id: str):
        project = _load_project(local_storage, project_id)
        cached = _get_preview(previews, preview_id)
        if cached.snapshot.project_id != project.id:
            raise HTTPException(status_code=404, detail="Preview does not belong to this project.")
        if not native_renderer.verifies_snapshot(cached.snapshot):
            raise HTTPException(status_code=409, detail="Renderer validation is missing; render again.")
        return FileResponse(
            cached.audio_path,
            media_type="audio/wav",
            filename="tone-preview.wav",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/samples/{sample_name}")
    def sample_audio(sample_name: str):
        if sample_name not in DI_SAMPLES:
            raise HTTPException(status_code=404, detail="Dry DI sample not found.")
        path = Path(__file__).parent / "assets" / "di" / f"{sample_name}.wav"
        if not path.is_file():
            raise HTTPException(status_code=503, detail="Bundled dry DI sample is missing.")
        return FileResponse(path, media_type="audio/wav", filename=f"{sample_name}.wav")

    @app.post("/api/projects/{project_id}/export")
    def export_preset(project_id: str, body: ExportRequest):
        project = _load_project(local_storage, project_id)
        cached = _get_preview(previews, body.preview_id)
        if not native_renderer.verifies_snapshot(cached.snapshot):
            raise HTTPException(status_code=409, detail="Renderer validation is missing; render again.")
        try:
            archive = soundshed.export_archive(project, cached.snapshot)
        except AdapterError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        filename = quote(f"{project.name}.soundshed.preset", safe="")
        return Response(
            archive,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{filename}",
                "Cache-Control": "no-store",
            },
        )

    if (_STATIC_ROOT / "assets").is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=_STATIC_ROOT / "assets"),
            name="assets",
        )

    @app.get("/", response_class=HTMLResponse)
    def frontend():
        index = _STATIC_ROOT / "index.html"
        if index.is_file():
            return FileResponse(index, media_type="text/html")
        return HTMLResponse(
            "<!doctype html><title>Tone Listener Workbench</title>"
            "<main><h1>Tone Listener Workbench</h1>"
            "<p>The frontend has not been built. Run <code>npm --prefix web run build</code>.</p>"
            "</main>",
            status_code=503,
        )

    @app.get("/youtube-player.html")
    def isolated_youtube_player():
        player = _STATIC_ROOT / "youtube-player.html"
        if not player.is_file():
            raise HTTPException(status_code=503, detail="Build the local frontend first.")
        return FileResponse(player, media_type="text/html", headers={
            "Content-Security-Policy":
                "default-src 'none'; script-src 'unsafe-inline' https://www.youtube.com https://s.ytimg.com; "
                "style-src 'unsafe-inline'; frame-src https://www.youtube.com https://www.youtube-nocookie.com; "
                "connect-src https://www.youtube.com; img-src data:; base-uri 'none'; "
                "frame-ancestors 'self'; sandbox allow-scripts",
        })

    return app


def _load_project(storage: LocalStorage, project_id: str) -> ToneProject:
    try:
        project = storage.load_project(project_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Local project was not found.") from exc
    except StorageError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if project is None:
        raise HTTPException(status_code=404, detail="Local project was not found.")
    return project


def _load_project_for_job(storage: LocalStorage, project_id: str) -> ToneProject:
    try:
        project = storage.load_project(project_id)
    except (ValueError, StorageError) as exc:
        raise APIError("Local project is no longer available.") from exc
    if project is None:
        raise APIError("Local project is no longer available.")
    return project


def _assert_revision(project: ToneProject, project_revision: int, chain_revision: int) -> None:
    if project.revision != project_revision or project.chain.revision != chain_revision:
        raise HTTPException(status_code=409, detail="Project chain changed; reload before previewing.")


def _check_job_revision(project: ToneProject, project_revision: int, chain_revision: int) -> None:
    if project.revision != project_revision or project.chain.revision != chain_revision:
        raise APIError("Project chain changed; this preview was not started.")


def _get_preview(cache: PreviewCache, preview_id: str):
    try:
        preview = cache.get(preview_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Preview was not found.") from exc
    if preview is None:
        raise HTTPException(status_code=404, detail="Preview was not found or has expired.")
    return preview


def _load_uploaded_interval(path: Path, interval: Interval) -> AudioClip:
    try:
        return load_audio(
            str(path),
            start=interval.start_seconds,
            end=interval.end_seconds,
        )
    except (OSError, ValueError, sf.LibsndfileError) as exc:
        reason = str(exc).replace(str(path), "uploaded audio")
        raise JobFailure(f"Could not decode selected audio interval: {reason[:300]}") from exc


def _tone_intent(segments: list[AnalysisSegment], selected: int = 0) -> ToneIntent:
    segment = segments[selected]
    if segment.profile is not None:
        profile = segment.profile
        return ToneIntent(
            summary=f"Selected interval {selected + 1}: {profile.gain} saturation, "
                    f"{profile.brightness} brightness and {profile.mids} mids. "
                    "Equipment choices are starting candidates, not original-rig identification.",
            observations=[
                f"{label}: {getattr(profile, field)}" for label, field in (
                    ("gain", "gain"), ("brightness", "brightness"), ("bass", "bass_level"),
                    ("bass tightness", "bass_tightness"), ("mids", "mids"), ("pick attack", "attack"),
                    ("compression", "compression"), ("sustain", "sustain"),
                )
            ],
            uncertainties=profile.uncertainties[:32],
            gear_notes=[
                "This recipe uses only the selected interval; other clips are not averaged or OR-merged.",
                "Capture choices use curated local-library metadata, not validated perceptual identity.",
                "Attack, sustain and compression depend on the DI performance and capture; no physical knobs were invented.",
            ],
        )
    successful = [segment.listening] if segment.listening is not None else []
    if not successful:
        return ToneIntent(
            summary="No tone intent was inferred because the selected analysis intervals were unavailable.",
            uncertainties=["No successful listening analysis was available."],
            gear_notes=["Audible observations do not identify the original rig or exact settings."],
        )
    summaries = list(dict.fromkeys(item.summary.strip() for item in successful if item.summary.strip()))
    observations = []
    for label, attribute in (
        ("gain", "gain_level"),
        ("brightness", "brightness"),
        ("low end", "low_end"),
        ("mids", "mids_character"),
        ("pick attack", "pick_attack"),
        ("compression", "compression_feel"),
    ):
        values = list(dict.fromkeys(getattr(item, attribute).strip() for item in successful))
        observations.append(f"{label}: {'; '.join(values)}")
    uncertainties = list(
        dict.fromkeys(note for item in successful for note in item.uncertainties)
    )
    return ToneIntent(
        summary=" ".join(summaries)[:2000] or "Tone observations recorded from the selected intervals.",
        observations=observations,
        uncertainties=uncertainties[:32],
        gear_notes=["Audible observations do not identify the original rig or exact settings."],
    )


def _private_refinement_root(storage: LocalStorage) -> Path:
    root = storage.root / "refinements"
    root.mkdir(mode=0o700, exist_ok=True)
    os.chmod(root, 0o700)
    return root


def _write_private_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


AUDITION_RMS_DBFS = -20.0
AUDITION_PEAK = 10 ** (-1 / 20)


def _audition_level_matched(paths: dict[str, Path], wanted: str) -> bytes:
    """Equal-RMS audition of ``wanted`` among its siblings with shared peak headroom (-1 dBFS)."""
    import io

    import numpy as np

    loaded = {key: sf.read(str(path), dtype="float32", always_2d=True) for key, path in paths.items()}
    gains, peaks = {}, []
    for key, (samples, _) in loaded.items():
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        if not np.isfinite(rms) or rms <= 1e-6:
            raise ValueError("silent render")
        gains[key] = 10 ** (AUDITION_RMS_DBFS / 20) / rms
        peaks.append(float(np.max(np.abs(samples))) * gains[key])
    shared = min(1.0, AUDITION_PEAK / max(peaks))
    samples, rate = loaded[wanted]
    buffer = io.BytesIO()
    sf.write(buffer, samples * gains[wanted] * shared, rate, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


TERMINAL_REFINEMENTS = frozenset({"completed", "partial", "cancelled", "failed"})


def _prune_refinements(root: Path, keep: int = 20, protect: frozenset[str] | set[str] = frozenset()) -> None:
    """Keep the newest ``keep`` terminal records; never touch running, protected or unrecognized directories."""
    terminal = []
    for path in root.iterdir():
        if not path.is_dir() or path.is_symlink() or path.name in protect:
            continue
        try:
            status = json.loads((path / "record.json").read_text()).get("status")
        except (OSError, ValueError, AttributeError):
            continue
        if status in TERMINAL_REFINEMENTS:
            terminal.append(path)
    terminal.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for directory in terminal[keep:]:
        shutil.rmtree(directory, ignore_errors=True)


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
