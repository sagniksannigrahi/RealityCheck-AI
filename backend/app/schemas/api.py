"""Request/response models for the public REST API."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

from app.schemas.claims import AtomicClaim
from app.schemas.evidence import EvidenceItem

T = TypeVar("T")


class VerificationMode(StrEnum):
    VERIFY_DESCRIPTION = "verify_description"
    GENERATE_AND_VERIFY = "generate_and_verify"


class JobSettings(BaseModel):
    """User-adjustable verification settings (bounded; server limits always apply)."""

    model_config = {"extra": "forbid"}

    abstention: bool = True
    disagreement_analysis: bool = True
    use_detector: bool = True
    use_grounding: bool = True
    use_vlm: bool = True
    sampling_strategy: Literal["uniform", "interval", "scene_aware"] | None = None
    max_frames: int | None = Field(default=None, ge=2, le=32)


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: Any | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int


class MediaOut(BaseModel):
    media_type: str
    mime_type: str
    url: str | None
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None
    fps: float | None = None
    frame_count: int | None = None
    size_bytes: int
    original_filename: str | None = None


class JobError(BaseModel):
    code: str | None
    message: str | None


class JobCreated(BaseModel):
    job_id: str
    status: str
    mode: VerificationMode
    media_type: str
    demo_mode: bool
    links: dict[str, str]


class JobSummary(BaseModel):
    job_id: str
    status: str
    mode: str
    media_type: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    progress_stage: str | None
    demo_mode: bool
    attempts: int
    description_original: str | None
    description_generated: str | None
    description_used: str | None
    summary: dict[str, Any] | None
    warnings: list[str] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)
    error: JobError | None = None
    media: MediaOut | None = None
    pipeline_version: str | None = None
    policy: str | None = None
    timings_ms: dict[str, Any] | None = None


class HistoryItem(BaseModel):
    job_id: str
    status: str
    mode: str
    media_type: str
    created_at: datetime
    finished_at: datetime | None
    demo_mode: bool
    description_preview: str | None
    verdict_counts: dict[str, int] | None
    headline: str | None
    thumbnail_url: str | None


class ClaimResultOut(BaseModel):
    verdict: str
    evidence_case: str
    verification_score: float | None
    calibrated_probability: float | None
    evidence_quality: str
    explanation: str
    uncertainty_reasons: list[str]
    review_recommended: bool
    supporting_evidence: list[str]
    contradicting_evidence: list[str]
    considered_evidence: list[str]
    decision_trace: list[str]
    models: list[dict[str, Any]]
    duration_ms: float


class ClaimOut(BaseModel):
    claim: AtomicClaim
    result: ClaimResultOut | None


class EvidenceOut(EvidenceItem):
    crop_url: str | None = None


class FrameOut(BaseModel):
    frame_index: int
    timestamp_s: float
    url: str | None
    reason: str
    scene_change_score: float | None
    width: int
    height: int


class ModelExecutionOut(BaseModel):
    component: str
    model_name: str
    model_version: str | None
    device: str | None
    calls: int
    total_ms: float
    demo: bool
    external: bool


class Report(BaseModel):
    report_version: str = "1.0"
    generated_at: datetime
    disclaimer: str
    job: JobSummary
    claims: list[ClaimOut]
    evidence: list[EvidenceOut]
    frames: list[FrameOut]
    model_executions: list[ModelExecutionOut]
    processing_errors: list[dict[str, Any]]
