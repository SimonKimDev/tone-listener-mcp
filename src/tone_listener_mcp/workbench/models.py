"""Versioned, target-neutral workbench project and chain models."""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tone_listener_mcp.listener import ListeningResult, ToneProfile

# Bundled dry DI recordings. The riffs remain the default; the leads are opt-in.
DiSample = Literal["riff-1", "riff-2", "lead-1", "lead-2"]
DI_SAMPLES: tuple[str, ...] = ("riff-1", "riff-2", "lead-1", "lead-2")
DEFAULT_DI_SAMPLES: tuple[str, ...] = ("riff-1", "riff-2")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Interval(StrictModel):
    start_seconds: float = Field(ge=0, le=86_400)
    end_seconds: float = Field(gt=0, le=86_430)

    @model_validator(mode="after")
    def validate_span(self) -> Interval:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("interval end must be after its start")
        if self.end_seconds - self.start_seconds > 30:
            raise ValueError("each analysis interval is limited to 30 seconds")
        return self


class AssetRef(StrictModel):
    resource_type: Literal["nam", "ir"]
    resource_id: str = Field(min_length=1, max_length=256)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    display_name: str = Field(min_length=1, max_length=200)
    capture_scope: Literal["pedal", "amp", "unknown"] = "unknown"


NodeKind = Literal[
    "input",
    "nam_fx",
    "nam_amp",
    "cab_ir",
    "parametric_eq",
    "room_reverb",
    "digital_delay",
    "chorus",
    "gain",
    "gear_note",
    "output",
]


class ChainNode(StrictModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    kind: NodeKind
    title: str = Field(min_length=1, max_length=120)
    enabled: bool = True
    params: dict[str, float] = Field(default_factory=dict)
    asset: AssetRef | None = None
    note: str | None = Field(default=None, max_length=1000)
    # blocking: required/unresolved equipment, the chain cannot be previewed or exported.
    # omitted_optional: an uncertain optional effect deliberately left OUT of the rendered
    # and exported chain; the note stays visible as a limitation, nothing is substituted.
    note_role: Literal["blocking", "omitted_optional"] = "blocking"

    @model_validator(mode="after")
    def validate_resource_shape(self) -> ChainNode:
        requires_asset = self.kind in ("nam_fx", "nam_amp", "cab_ir")
        if requires_asset != (self.asset is not None):
            raise ValueError("NAM and cabinet nodes require exactly one local library asset")
        if self.kind == "gear_note" and not self.note:
            raise ValueError("gear notes must explain the unsupported or missing equipment")
        if self.kind != "gear_note" and self.note_role != "blocking":
            raise ValueError("only gear notes can record an omitted optional effect")
        return self


class ChainDraft(StrictModel):
    revision: int = Field(default=1, ge=1)
    nodes: list[ChainNode] = Field(min_length=2, max_length=16)

    @model_validator(mode="after")
    def validate_serial_chain(self) -> ChainDraft:
        if self.nodes[0].kind != "input" or self.nodes[-1].kind != "output":
            raise ValueError("the serial chain must start with Input and end with Output")
        if sum(node.kind == "input" for node in self.nodes) != 1:
            raise ValueError("Input is fixed and must appear exactly once")
        if sum(node.kind == "output" for node in self.nodes) != 1:
            raise ValueError("Output is fixed and must appear exactly once")
        ids = [node.id for node in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("chain node ids must be unique")
        return self


class ToneIntent(StrictModel):
    summary: str = Field(min_length=1, max_length=2000)
    observations: list[str] = Field(default_factory=list, max_length=32)
    uncertainties: list[str] = Field(default_factory=list, max_length=32)
    gear_notes: list[str] = Field(default_factory=list, max_length=32)


class SourceRef(StrictModel):
    kind: Literal["youtube", "audio_file"]
    url: str | None = None
    display_name: str = Field(min_length=1, max_length=200)
    duration_seconds: float | None = Field(default=None, gt=0, le=86_400)
    selected_intervals: list[Interval] = Field(min_length=1, max_length=3)
    # SHA-256 of the analyzed audio bytes; lets a later re-upload be verified as the same source.
    # Absent on projects analyzed before fingerprints were recorded.
    content_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_source(self) -> SourceRef:
        if (self.kind == "youtube") != (self.url is not None):
            raise ValueError("only YouTube sources may include a URL")
        if self.kind != "audio_file" and self.content_sha256 is not None:
            raise ValueError("only uploaded audio sources carry a content fingerprint")
        if self.kind == "audio_file" and self.duration_seconds is None:
            raise ValueError("uploaded audio must include its decoded duration")
        if self.url is not None and len(self.url) > 512:
            raise ValueError("YouTube URL is too long")
        previous_end = -1.0
        for interval in sorted(self.selected_intervals, key=lambda item: item.start_seconds):
            if self.duration_seconds is not None and interval.end_seconds > self.duration_seconds:
                raise ValueError("selected interval exceeds source duration")
            if interval.start_seconds < previous_end:
                raise ValueError("selected intervals must not overlap")
            previous_end = interval.end_seconds
        return self


class AnalysisFailure(StrictModel):
    http_status: int | None = Field(default=None, ge=100, le=599)
    category: Literal[
        "authentication", "permission", "model_unavailable", "quota_unspecified",
        "daily_quota", "rate_window", "capacity", "request_failure",
    ]
    retry_after_seconds: float | None = Field(default=None, ge=0)
    retry_after_invalid: bool = False
    daily_request_limit: int | None = Field(default=None, ge=0)
    daily_token_limit: int | None = Field(default=None, ge=0)


class AnalysisReuse(StrictModel):
    """This job did not send this interval to the provider; the result came from elsewhere."""

    kind: Literal["cache", "shared_in_flight"]
    analyzed_at: str = Field(max_length=64)


class AnalysisSegment(StrictModel):
    interval: Interval
    model: str
    listening: ListeningResult | None = None
    profile: ToneProfile | None = None
    unavailable: str | None = None
    failure_info: AnalysisFailure | None = None
    provider_attempted: bool | None = None
    reuse: AnalysisReuse | None = None
    metrics: dict[str, object] | None = None

    @model_validator(mode="after")
    def require_outcome(self) -> AnalysisSegment:
        if ((self.listening is not None or self.profile is not None)
                == (self.unavailable is not None)):
            raise ValueError("analysis segment must contain either a result or an explicit failure")
        if self.reuse is not None and self.provider_attempted:
            raise ValueError("reused analysis cannot also be a provider attempt by this job")
        if self.reuse is not None and self.reuse.kind == "cache" and self.profile is None:
            raise ValueError("only successful profiles can be reused from the cache")
        if self.metrics is not None and isinstance(self.metrics.get("clip"), dict):
            self.metrics = {
                **self.metrics,
                "clip": {**self.metrics["clip"], "source": "Selected audio interval"},
            }
        return self


class TargetBinding(StrictModel):
    adapter_id: str
    engine_version: str
    resources: dict[str, AssetRef] = Field(default_factory=dict)


class ToneProject(StrictModel):
    schema_version: Literal[1] = 1
    id: str = Field(default_factory=lambda: str(uuid4()))
    revision: int = Field(default=1, ge=1)
    name: str = Field(min_length=1, max_length=120)
    source: SourceRef
    analysis: list[AnalysisSegment] = Field(min_length=1, max_length=3)
    selected_analysis_index: int = Field(default=0, ge=0, le=2)
    intent: ToneIntent
    chain: ChainDraft
    target: TargetBinding
    created_at: str
    updated_at: str
    synthetic_fixture: bool = False

    @model_validator(mode="before")
    @classmethod
    def sanitize_private_paths(cls, value):
        from .privacy import redact_paths
        return redact_paths(value)

    @model_validator(mode="after")
    def validate_selected_analysis(self) -> ToneProject:
        if self.selected_analysis_index >= len(self.analysis):
            raise ValueError("selected analysis must refer to a stored source interval")
        return self


class ParameterSpec(StrictModel):
    id: str
    label: str
    minimum: float
    maximum: float
    default: float
    unit: str
    description: str


class BlockDefinition(StrictModel):
    kind: NodeKind
    label: str
    description: str
    parameters: list[ParameterSpec] = Field(default_factory=list)


class ResourceDefinition(StrictModel):
    asset: AssetRef
    provider: str
    file_size_bytes: int = Field(gt=0)


class AdapterCatalog(StrictModel):
    adapter_id: str
    adapter_label: str
    engine_version: str
    blocks: list[BlockDefinition]
    resources: list[ResourceDefinition]


class PreviewSnapshot(StrictModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    project_id: str
    project_revision: int = Field(ge=1)
    chain_revision: int = Field(ge=1)
    chain_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sample_name: DiSample
    source_sample_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    engine_version: str
    sample_rate: int = Field(gt=0)
    dsp_conditions: dict[str, str | int | float | bool | None]
    audio_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    duration_seconds: float = Field(gt=0, le=40)
    rms: float = Field(ge=0)
    peak: float = Field(ge=0)
    created_at: str


class JobStatus(StrictModel):
    id: str
    kind: Literal["analysis", "preview", "refinement"]
    status: Literal["queued", "running", "completed", "partial", "failed", "cancelled", "stale"]
    progress: float = Field(ge=0, le=1)
    message: str
    result: dict[str, object] | None = None
    error: str | None = None
