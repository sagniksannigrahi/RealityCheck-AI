"""ORM models. See docs/DATABASE.md for the schema description."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.persistence.db import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ALL = (QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED)
    TERMINAL = (COMPLETED, FAILED, CANCELLED)


class MediaAsset(Base):
    __tablename__ = "media_assets"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[str] = mapped_column(String(64), index=True)
    storage_key: Mapped[str] = mapped_column(String(255), unique=True)
    media_type: Mapped[str] = mapped_column(String(16))  # image | video
    mime_type: Mapped[str] = mapped_column(String(64))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    duration_s: Mapped[float | None] = mapped_column(Float)
    fps: Mapped[float | None] = mapped_column(Float)
    frame_count: Mapped[int | None] = mapped_column(Integer)
    original_filename: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class VerificationJob(Base):
    __tablename__ = "verification_jobs"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED, index=True)
    mode: Mapped[str] = mapped_column(String(32))  # verify_description | generate_and_verify
    media_type: Mapped[str] = mapped_column(String(16), index=True)
    media_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("media_assets.id", ondelete="RESTRICT"), index=True)
    description_original: Mapped[str | None] = mapped_column(Text)
    description_generated: Mapped[str | None] = mapped_column(Text)
    description_used: Mapped[str | None] = mapped_column(Text)
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    warnings: Mapped[list[str] | None] = mapped_column(JSON)
    demo_mode: Mapped[bool] = mapped_column(Boolean, default=False)
    pipeline_version: Mapped[str | None] = mapped_column(String(32))
    policy_name: Mapped[str | None] = mapped_column(String(64))
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    progress_stage: Mapped[str | None] = mapped_column(String(64))
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_by: Mapped[str | None] = mapped_column(String(128))
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    request_id: Mapped[str | None] = mapped_column(String(64))
    timings_ms: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    media: Mapped[MediaAsset] = relationship(lazy="joined")
    claims: Mapped[list[Claim]] = relationship(back_populates="job", cascade="all, delete-orphan",
                                               order_by="Claim.position")
    __table_args__ = (Index("ix_jobs_owner_created", "owner_id", "created_at"),
                      Index("ix_jobs_status_created", "status", "created_at"))


class Claim(Base):
    __tablename__ = "claims"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("verification_jobs.id", ondelete="CASCADE"), index=True)
    claim_key: Mapped[str] = mapped_column(String(16))
    position: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    claim_type: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    job: Mapped[VerificationJob] = relationship(back_populates="claims")
    result: Mapped[ClaimResult | None] = relationship(back_populates="claim", uselist=False,
                                                      cascade="all, delete-orphan")
    __table_args__ = (UniqueConstraint("job_id", "claim_key", name="uq_claim_job_key"),)


class ClaimResult(Base):
    __tablename__ = "claim_results"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    claim_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"), unique=True)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("verification_jobs.id", ondelete="CASCADE"), index=True)
    verdict: Mapped[str] = mapped_column(String(32), index=True)
    evidence_case: Mapped[str] = mapped_column(String(32))
    verification_score: Mapped[float | None] = mapped_column(Float)
    calibrated_probability: Mapped[float | None] = mapped_column(Float)
    evidence_quality: Mapped[str] = mapped_column(String(16))
    explanation: Mapped[str] = mapped_column(Text)
    uncertainty_reasons: Mapped[list[str]] = mapped_column(JSON, default=list)
    review_recommended: Mapped[bool] = mapped_column(Boolean, default=False)
    supporting_evidence: Mapped[list[str]] = mapped_column(JSON, default=list)
    contradicting_evidence: Mapped[list[str]] = mapped_column(JSON, default=list)
    considered_evidence: Mapped[list[str]] = mapped_column(JSON, default=list)
    decision_trace: Mapped[list[str]] = mapped_column(JSON, default=list)
    model_metadata: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    claim: Mapped[Claim] = relationship(back_populates="result")


class EvidenceRecord(Base):
    __tablename__ = "evidence_records"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    evidence_key: Mapped[str] = mapped_column(String(32))
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("verification_jobs.id", ondelete="CASCADE"), index=True)
    claim_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"), index=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    model_name: Mapped[str | None] = mapped_column(String(128))
    model_version: Mapped[str | None] = mapped_column(String(128))
    stance: Mapped[str] = mapped_column(String(16))
    strength: Mapped[str] = mapped_column(String(8))
    raw_score: Mapped[float | None] = mapped_column(Float)
    box: Mapped[dict[str, float] | None] = mapped_column(JSON)
    label: Mapped[str | None] = mapped_column(String(128))
    frame_index: Mapped[int | None] = mapped_column(Integer)
    timestamp_s: Mapped[float | None] = mapped_column(Float)
    crop_key: Mapped[str | None] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    __table_args__ = (UniqueConstraint("job_id", "evidence_key", name="uq_evidence_job_key"),)


class VideoFrame(Base):
    __tablename__ = "video_frames"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("verification_jobs.id", ondelete="CASCADE"), index=True)
    frame_index: Mapped[int] = mapped_column(Integer)
    timestamp_s: Mapped[float] = mapped_column(Float)
    storage_key: Mapped[str | None] = mapped_column(String(255))
    reason: Mapped[str] = mapped_column(String(32))
    scene_change_score: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    __table_args__ = (UniqueConstraint("job_id", "frame_index", name="uq_frame_job_index"),)


class ModelExecution(Base):
    __tablename__ = "model_executions"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("verification_jobs.id", ondelete="CASCADE"), index=True)
    component: Mapped[str] = mapped_column(String(32))
    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str | None] = mapped_column(String(128))
    device: Mapped[str | None] = mapped_column(String(16))
    calls: Mapped[int] = mapped_column(Integer, default=0)
    total_ms: Mapped[float] = mapped_column(Float, default=0.0)
    demo: Mapped[bool] = mapped_column(Boolean, default=False)
    external: Mapped[bool] = mapped_column(Boolean, default=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ProcessingError(Base):
    __tablename__ = "processing_errors"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("verification_jobs.id", ondelete="CASCADE"), index=True)
    stage: Mapped[str] = mapped_column(String(64))
    error_type: Mapped[str] = mapped_column(String(128))
    message: Mapped[str] = mapped_column(Text)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[str] = mapped_column(String(128), unique=True)
    experiment: Mapped[str] = mapped_column(String(128), index=True)
    system: Mapped[str] = mapped_column(String(64), index=True)
    dataset: Mapped[str] = mapped_column(String(128), index=True)
    dataset_version: Mapped[str | None] = mapped_column(String(64))
    split: Mapped[str] = mapped_column(String(32))
    seed: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="completed")
    simulated: Mapped[bool] = mapped_column(Boolean, default=False)
    n_examples: Mapped[int] = mapped_column(Integer, default=0)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    model_versions: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    output_path: Mapped[str | None] = mapped_column(String(255))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("evaluation_runs.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(64))
    category: Mapped[str] = mapped_column(String(64), default="all")
    value: Mapped[float | None] = mapped_column(Float)
    ci_low: Mapped[float | None] = mapped_column(Float)
    ci_high: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (UniqueConstraint("run_id", "name", "category", name="uq_metric_run_name_cat"),)
