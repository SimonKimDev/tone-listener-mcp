"""Loopback-only local web API for the tone workbench."""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import math
import os
import secrets
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote, urlsplit

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
) -> FastAPI:
    local_storage = storage or LocalStorage()
    soundshed = adapter or SoundshedAdapter()
    analysis_cache = AnalysisCache(local_storage.analysis)
    upload_digests: dict[str, str] = {}
    native_renderer = renderer or NativeRenderer(soundshed)
    previews = PreviewCache(local_storage.cache)
    jobs = job_manager or JobManager()
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


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
