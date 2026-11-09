"""Pydantic response models that give MCP clients meaningful tool output schemas.

Responses allow extra keys so a newer server field never fails a call; tests assert that
every produced key is declared here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tone_listener_mcp.listener import ComparisonResult, ListeningResult

BandId = Literal[
    "below_80", "80_250", "250_800", "800_2000", "2000_5000", "5000_8000", "above_8000"
]
FailureStage = Literal[
    "prepare_upload", "queue", "create_client", "upload", "upload_processing", "generate",
    "validate",
]
FailureCategory = Literal[
    "not_configured", "audio_preparation", "authentication", "permission", "model_unavailable",
    "invalid_request", "quota", "capacity", "provider_error", "deadline", "busy", "network",
    "upload_failed", "invalid_response", "local_io", "daily_quota", "rate_window",
]


class Response(BaseModel):
    model_config = ConfigDict(extra="allow")


class ClipMetadata(Response):
    source: str
    sample_rate: int
    channels: int
    duration_seconds: float
    start_seconds: float


class ShortTermRms(Response):
    p10: float | None
    p50: float | None
    p90: float | None


class LevelMetrics(Response):
    peak_dbfs: float | None
    rms_dbfs: float | None
    crest_factor_db: float | None
    gated_rms_dbfs: float | None = Field(description="RMS of gated, non-silent windows")
    gated_crest_factor_db: float | None
    gated_duration_seconds: float
    gating_method: str
    short_term_window_seconds: float
    short_term_rms_dbfs: ShortTermRms
    short_term_rms_range_db: float | None
    short_term_range_method: str
    active_window_fraction: float
    silent_window_fraction: float


class Band(Response):
    available: bool
    effective_range_hz: list[float] | None = Field(
        description="[low, min(high, Nyquist)] Hz; null when the band starts at or above Nyquist"
    )
    relative_db: float | None = Field(description="Band power relative to total power, dB")


class SpectrumMetrics(Response):
    method: str
    nyquist_hz: float
    resolution_hz: float
    centroid_hz: float | None
    rolloff_85_hz: float | None
    flatness: float | None
    bands: dict[BandId, Band]


class StereoMetrics(Response):
    lr_correlation: float | None
    side_mid_energy_ratio_db: float | None
    mid_rms_dbfs: float | None
    side_rms_dbfs: float | None


class AudioMetrics(Response):
    clip: ClipMetadata
    level: LevelMetrics
    spectrum: SpectrumMetrics
    stereo: StereoMetrics | None = Field(description="null for mono clips")


class UploadLoudnessMatch(Response):
    applied: bool
    strategy: Literal["active", "whole"]
    additional_gain_db_applied_to_b: float | None
    reason: str | None


class Failure(Response):
    category: FailureCategory
    http_status: int | None = Field(description="Provider HTTP status when one was returned")
    retryable: bool
    suggestion: str
    retry_after_seconds: float | None = Field(
        description="Longest valid Retry-After or RetryInfo delay, in seconds"
    )
    retry_after_invalid: bool
    daily_request_limit: int | None
    daily_token_limit: int | None


class Usage(Response):
    prompt_token_count: int | None
    candidates_token_count: int | None
    thoughts_token_count: int | None
    cached_content_token_count: int | None
    total_token_count: int | None


class Preprocessing(Response):
    label: Literal["A", "B", "C"]
    original_sample_rate: int
    original_channels: int
    duration_seconds: float
    upload_sample_rate: int
    upload_channels: int


class Diagnostics(Response):
    elapsed_seconds: float
    deadline_seconds: float
    max_concurrent_requests: int
    prompt_version: str
    schema_version: str
    cache: Literal["hit", "miss", "disabled", "not_used"] = Field(
        description="not_used when no provider request was attempted"
    )
    usage: Usage | None = Field(description="Provider token usage; null on cache-less failures")
    preprocessing: list[Preprocessing]


class RequestCounts(Response):
    """SDK operation invocations in this call, including failed attempts and cleanup."""

    generate: int
    upload: int
    poll_upload: int
    delete_upload: int


class ListeningFields(Response):
    model: str
    listening_unavailable: str | None = Field(
        description="Why listening is absent; null when listening succeeded"
    )
    failure: Failure | None = Field(
        description="Machine-readable failure category and recovery hint; never provider text"
    )
    diagnostics: Diagnostics
    request_counts: RequestCounts
    cleanup_warnings: list[str]
    upload_gain_db: float
    upload_loudness_match: UploadLoudnessMatch | None
    listening_failure_stage: FailureStage | None
    input_transport: Literal["files", "youtube_url", "youtube_url+files"]


class RequestSettings(Response):
    deadline_seconds: float
    max_concurrent_requests: int
    cache_entries: int


class ServerInfo(Response):
    name: str
    version: str
    primary_tool: str
    youtube_local_extraction: bool
    model: str = Field(description="Gemini model listening requests will use")
    listening_available: bool = Field(
        description="True when a key is configured and settings are valid; no request is made"
    )
    listening_unavailable: str | None = Field(description="Why listening is unavailable")
    max_seconds: float | None = Field(description="TONE_LISTENER_MAX_SECONDS; null if invalid")
    request_settings: RequestSettings | None = Field(description="Null if any value is invalid")
    configuration_errors: list[str]


class AnalyzeAudioResponse(ListeningFields):
    metrics: AudioMetrics
    listening: ListeningResult | None


class YouTubeSource(Response):
    url: str
    start_seconds: float
    end_seconds: float


class AnalyzeYouTubeResponse(ListeningFields):
    source: YouTubeSource
    metrics: None
    metrics_unavailable: str
    listening: ListeningResult | None


class Activity(Response):
    duration_seconds_a: float
    duration_seconds_b: float
    active_seconds_a: float
    active_seconds_b: float
    duration_mismatch: bool
    activity_mismatch: bool


class LoudnessMatch(Response):
    applied: bool
    strategy: Literal["active", "whole"]
    method: str
    gain_db_applied_to_b: float | None
    reason: str | None
    activity: Activity


class StereoDeltas(Response):
    lr_correlation: float | None
    side_mid_energy_ratio_db: float | None


class Deltas(Response):
    relative_band_db: dict[BandId, float | None]
    band_evidence: dict[BandId, Literal["sufficient", "insufficient_energy", "unavailable"]]
    band_evidence_floor_db: float
    centroid_hz: float | None
    rolloff_85_hz: float | None
    crest_factor_db: float | None
    crest_factor_basis: Literal["gated_crest_factor_db", "crest_factor_db"]
    stereo: StereoDeltas | None


class EqHint(Response):
    family: Literal["eq"]
    band: BandId
    band_label: str
    direction: Literal["cut", "boost"] = Field(description="Relative change to B, only if EQ")
    delta_db: float = Field(description="B minus A relative band energy")
    priority: int = Field(ge=1, description="1 = largest difference")
    observation: str
    alternative_explanation: str
    listen_for: str
    text: str = Field(description="Human-readable summary of the other fields")


class AvailableAnalyses(Response):
    """Which analyses this source combination supports; listening also needs a Gemini key."""

    semantic_listening: bool
    rms_matching: bool
    spectral_deltas: bool


class CompareResponse(ListeningFields):
    metrics_a: AudioMetrics
    metrics_b: AudioMetrics
    matched_metrics_b: AudioMetrics
    loudness_match: LoudnessMatch
    deltas_b_minus_a: Deltas
    eq_direction_hints: list[EqHint]
    cautions: list[str]
    available_analyses: AvailableAnalyses
    listening: ComparisonResult | None


class UnavailableAnalyses(Response):
    rms_matching: str
    spectral_deltas: str


class CompareYouTubeResponse(ListeningFields):
    reference: YouTubeSource
    available_analyses: AvailableAnalyses
    unavailable_analyses: UnavailableAnalyses
    metrics_b: AudioMetrics
    cautions: list[str]
    listening: ComparisonResult | None
