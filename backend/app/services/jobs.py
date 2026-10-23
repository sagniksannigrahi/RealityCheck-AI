"""Job lifecycle: creation, result persistence, serialization and retention cleanup."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.observability.logging import redact_all
from app.persistence.orm import (
    Claim,
    ClaimResult,
    EvidenceRecord,
    JobStatus,
    MediaAsset,
    ModelExecution,
    ProcessingError,
    VerificationJob,
    VideoFrame,
)
from app.schemas.api import (
    ClaimOut,
    ClaimResultOut,
    EvidenceOut,
    FrameOut,
    HistoryItem,
    JobError,
    JobSettings,
    JobSummary,
    MediaOut,
    ModelExecutionOut,
    Report,
    VerificationMode,
)
from app.schemas.claims import AtomicClaim
from app.services.media_validation import (
    MediaValidationError,
    decode_validated_image,
    sanitize_filename,
    validate_upload,
)
from app.services.storage import StorageBackend, new_key, signed_query
from app.verification.pipeline import PIPELINE_VERSION, PipelineResult

DISCLAIMER = (
    "Automated verification report. Verdicts combine independent model outputs with rule-based checks; "
    "scores are evidence-agreement scores, not guarantees of truth. 'Insufficient evidence' means the system "
    "abstained, not that the claim is false. Model explanations are hypotheses, not ground truth."
)


class Signer:
    def __init__(self, settings: Settings, owner: str, base: str = "/api/v1/files") -> None:
        self._secret = settings.secret_key.get_secret_value()
        self._ttl = settings.signed_url_ttl_seconds
        self._owner = owner
        self._base = base

    def url(self, key: str | None) -> str | None:
        if not key:
            return None
        return f"{self._base}?{signed_query(key, self._owner, self._secret, self._ttl)}"


# --------------------------------------------------------------------------------- creation


def create_job(db: Session, *, owner: str, data: bytes, filename: str | None, description: str | None,
               mode: VerificationMode, job_settings: JobSettings, storage: StorageBackend, settings: Settings,
               demo_mode: bool, request_id: str | None) -> VerificationJob:
    kind = validate_upload(data, settings)
    width = height = frame_count = None
    duration = fps = None
    if mode == VerificationMode.VERIFY_DESCRIPTION:
        if not description or not description.strip():
            raise MediaValidationError("description_required", "A description is required in verify_description mode.")
        if len(description) > settings.max_description_chars:
            raise MediaValidationError("description_too_long",
                                       f"Description exceeds {settings.max_description_chars} characters.")
    key = new_key("media", kind.ext)
    if kind.media_type == "image":
        img = decode_validated_image(data, settings)
        width, height = img.size
        storage.put(key, data)
    else:
        from app.video.sampling import VideoDecodeError, probe_video, validate_video

        storage.put(key, data)
        path = storage.local_path(key)
        tmp = None
        try:
            if path is None:  # remote storage: probe from a temp copy
                import tempfile

                tmp = tempfile.NamedTemporaryFile(suffix=f".{kind.ext}", delete=False)  # noqa: SIM115
                tmp.write(data)
                tmp.close()
                path = tmp.name
            info = probe_video(path)
            validate_video(info, max_seconds=settings.max_video_seconds, max_side=settings.max_video_side)
        except VideoDecodeError as exc:
            storage.delete(key)
            raise MediaValidationError("invalid_video", str(exc)) from exc
        finally:
            if tmp is not None:
                import os

                os.unlink(tmp.name)
        width, height, duration, fps, frame_count = info.width, info.height, info.duration_s, info.fps, info.frame_count

    media = MediaAsset(
        owner_id=owner, storage_key=key, media_type=kind.media_type, mime_type=kind.mime,
        sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data), width=width, height=height,
        duration_s=duration, fps=fps, frame_count=frame_count, original_filename=sanitize_filename(filename),
        expires_at=datetime.now(UTC) + timedelta(days=settings.media_retention_days),
    )
    db.add(media)
    db.flush()
    job = VerificationJob(
        owner_id=owner, status=JobStatus.QUEUED, mode=mode.value, media_type=kind.media_type, media_id=media.id,
        description_original=description.strip() if description else None,
        settings=job_settings.model_dump(), demo_mode=demo_mode, pipeline_version=PIPELINE_VERSION,
        request_id=request_id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


# --------------------------------------------------------------------------------- results


def persist_result(db: Session, job: VerificationJob, res: PipelineResult) -> None:
    # idempotent on retry: clear partial results from an earlier attempt
    for model in (EvidenceRecord, ClaimResult, Claim, VideoFrame, ModelExecution):
        db.execute(delete(model).where(model.job_id == job.id))
    claim_ids: dict[str, uuid.UUID] = {}
    for pos, c in enumerate(res.extraction.claims):
        row = Claim(job_id=job.id, claim_key=c.claim_id, position=pos, text=c.text, claim_type=c.claim_type.value,
                    payload=c.model_dump(mode="json"))
        db.add(row)
        db.flush()
        claim_ids[c.claim_id] = row.id
    for r in res.results:
        db.add(ClaimResult(
            claim_id=claim_ids[r.claim_id], job_id=job.id, verdict=r.verdict.value, evidence_case=r.evidence_case.value,
            verification_score=r.verification_score, calibrated_probability=r.calibrated_probability,
            evidence_quality=r.evidence_quality.value, explanation=r.explanation,
            uncertainty_reasons=r.uncertainty_reasons, review_recommended=r.review_recommended,
            supporting_evidence=r.supporting_evidence, contradicting_evidence=r.contradicting_evidence,
            considered_evidence=r.considered_evidence, decision_trace=r.decision_trace,
            model_metadata=[m.model_dump() for m in r.models], duration_ms=r.duration_ms,
        ))
    for e in res.evidence:
        db.add(EvidenceRecord(
            evidence_key=e.evidence_id, job_id=job.id, claim_id=claim_ids.get(e.claim_id or ""), source=e.source.value,
            model_name=e.model_name, model_version=e.model_version, stance=e.stance.value, strength=e.strength.value,
            raw_score=e.raw_score, box=e.box.model_dump() if e.box else None, label=e.label,
            frame_index=e.frame_index, timestamp_s=e.timestamp_s, crop_key=e.crop_key, description=e.description,
            details=e.details,
        ))
    for f in res.frames:
        db.add(VideoFrame(job_id=job.id, frame_index=f.frame_index, timestamp_s=f.timestamp_s,
                          storage_key=f.storage_key, reason=f.reason, scene_change_score=f.scene_change_score,
                          width=f.width, height=f.height))
    for m in res.model_executions:
        db.add(ModelExecution(job_id=job.id, component=m["component"], model_name=m["model_name"] or m["component"],
                              model_version=m.get("model_version"), device=m.get("device"), calls=m.get("calls", 0),
                              total_ms=m.get("total_ms", 0.0), demo=bool(m.get("demo")), external=bool(m.get("external")),
                              details={k: v for k, v in m.items() if k not in {
                                  "component", "model_name", "model_version", "device", "calls", "total_ms", "demo",
                                  "external"}}))
    job.description_generated = res.generated_description
    job.description_used = res.description_used
    job.summary = res.summary
    job.warnings = res.warnings
    job.timings_ms = res.timings_ms
    job.demo_mode = res.demo
    job.policy_name = res.summary.get("policy")


def record_error(db: Session, job: VerificationJob, stage: str, exc: BaseException, attempt: int) -> None:
    db.add(ProcessingError(job_id=job.id, stage=stage[:64], error_type=type(exc).__name__[:128],
                           message=redact_all(str(exc))[:2000], attempt=attempt))


# --------------------------------------------------------------------------------- queries


def get_owned_job(db: Session, job_id: str, owner: str) -> VerificationJob | None:
    try:
        jid = uuid.UUID(job_id)
    except ValueError:
        return None
    job = db.get(VerificationJob, jid)
    if job is None or job.owner_id != owner:
        return None  # do not reveal existence of other owners' jobs
    return job


def media_out(job: VerificationJob, signer: Signer) -> MediaOut | None:
    m = job.media
    if m is None:
        return None
    return MediaOut(media_type=m.media_type, mime_type=m.mime_type,
                    url=None if m.deleted_at else signer.url(m.storage_key), width=m.width, height=m.height,
                    duration_s=m.duration_s, fps=m.fps, frame_count=m.frame_count, size_bytes=m.size_bytes,
                    original_filename=m.original_filename)


def job_summary(job: VerificationJob, signer: Signer) -> JobSummary:
    err = JobError(code=job.error_code, message=job.error_message) if job.error_code else None
    return JobSummary(
        job_id=str(job.id), status=job.status, mode=job.mode, media_type=job.media_type, created_at=job.created_at,
        started_at=job.started_at, finished_at=job.finished_at, progress_stage=job.progress_stage,
        demo_mode=job.demo_mode, attempts=job.attempts, description_original=job.description_original,
        description_generated=job.description_generated, description_used=job.description_used, summary=job.summary,
        warnings=job.warnings or [], settings=job.settings or {}, error=err, media=media_out(job, signer),
        pipeline_version=job.pipeline_version, policy=job.policy_name, timings_ms=job.timings_ms,
    )


def claims_out(db: Session, job: VerificationJob) -> list[ClaimOut]:
    rows = db.execute(select(Claim).where(Claim.job_id == job.id).order_by(Claim.position)).scalars().all()
    out = []
    for c in rows:
        r = c.result
        res = None
        if r is not None:
            res = ClaimResultOut(
                verdict=r.verdict, evidence_case=r.evidence_case, verification_score=r.verification_score,
                calibrated_probability=r.calibrated_probability, evidence_quality=r.evidence_quality,
                explanation=r.explanation, uncertainty_reasons=r.uncertainty_reasons or [],
                review_recommended=r.review_recommended, supporting_evidence=r.supporting_evidence or [],
                contradicting_evidence=r.contradicting_evidence or [], considered_evidence=r.considered_evidence or [],
                decision_trace=r.decision_trace or [], models=r.model_metadata or [], duration_ms=r.duration_ms,
            )
        out.append(ClaimOut(claim=AtomicClaim.model_validate(c.payload), result=res))
    return out


def evidence_out(db: Session, job: VerificationJob, signer: Signer, claim_key: str | None = None) -> list[EvidenceOut]:
    q = select(EvidenceRecord, Claim.claim_key).outerjoin(Claim, Claim.id == EvidenceRecord.claim_id).where(
        EvidenceRecord.job_id == job.id)
    if claim_key:
        q = q.where(Claim.claim_key == claim_key)
    out = []
    for e, ck in db.execute(q.order_by(EvidenceRecord.created_at)).all():
        out.append(EvidenceOut(
            evidence_id=e.evidence_key, claim_id=ck, source=e.source, model_name=e.model_name,
            model_version=e.model_version, stance=e.stance, strength=e.strength, raw_score=e.raw_score,
            box=e.box, label=e.label, frame_index=e.frame_index, timestamp_s=e.timestamp_s, crop_key=None,
            description=e.description, details=e.details or {}, crop_url=signer.url(e.crop_key),
        ))
    return out


def frames_out(db: Session, job: VerificationJob, signer: Signer) -> list[FrameOut]:
    rows = db.execute(select(VideoFrame).where(VideoFrame.job_id == job.id).order_by(VideoFrame.frame_index)).scalars()
    return [FrameOut(frame_index=f.frame_index, timestamp_s=f.timestamp_s, url=signer.url(f.storage_key),
                     reason=f.reason, scene_change_score=f.scene_change_score, width=f.width, height=f.height)
            for f in rows]


def build_report(db: Session, job: VerificationJob, signer: Signer) -> Report:
    execs = db.execute(select(ModelExecution).where(ModelExecution.job_id == job.id)).scalars().all()
    errs = db.execute(select(ProcessingError).where(ProcessingError.job_id == job.id)
                      .order_by(ProcessingError.created_at)).scalars().all()
    return Report(
        generated_at=datetime.now(UTC), disclaimer=DISCLAIMER + (
            " DEMO MODE: results come from deterministic fake models and are NOT real model outputs."
            if job.demo_mode else ""),
        job=job_summary(job, signer), claims=claims_out(db, job), evidence=evidence_out(db, job, signer),
        frames=frames_out(db, job, signer),
        model_executions=[ModelExecutionOut(component=m.component, model_name=m.model_name,
                                            model_version=m.model_version, device=m.device, calls=m.calls,
                                            total_ms=m.total_ms, demo=m.demo, external=m.external) for m in execs],
        processing_errors=[{"stage": e.stage, "error_type": e.error_type, "message": e.message, "attempt": e.attempt,
                            "created_at": e.created_at.isoformat()} for e in errs],
    )


def history(db: Session, owner: str, signer: Signer, *, limit: int, offset: int, status: str | None,
            media_type: str | None, q: str | None, demo: bool | None) -> tuple[list[HistoryItem], int]:
    base = select(VerificationJob).where(VerificationJob.owner_id == owner)
    if status:
        base = base.where(VerificationJob.status == status)
    if media_type:
        base = base.where(VerificationJob.media_type == media_type)
    if demo is not None:
        base = base.where(VerificationJob.demo_mode == demo)
    if q:
        like = f"%{q.lower()[:100]}%"
        base = base.where(func.lower(func.coalesce(VerificationJob.description_used,
                                                    VerificationJob.description_original, "")).like(like))
    total = db.execute(select(func.count()).select_from(base.subquery())).scalar_one()
    rows = db.execute(base.order_by(VerificationJob.created_at.desc()).limit(limit).offset(offset)).scalars().all()
    items = []
    for j in rows:
        desc = j.description_used or j.description_original
        thumb = None
        if j.media is not None and j.media.media_type == "image" and not j.media.deleted_at:
            thumb = signer.url(j.media.storage_key)
        items.append(HistoryItem(
            job_id=str(j.id), status=j.status, mode=j.mode, media_type=j.media_type, created_at=j.created_at,
            finished_at=j.finished_at, demo_mode=j.demo_mode, description_preview=(desc[:160] if desc else None),
            verdict_counts=(j.summary or {}).get("verdict_counts"), headline=(j.summary or {}).get("headline"),
            thumbnail_url=thumb))
    return items, total


def history_stats(db: Session, owner: str) -> dict[str, Any]:
    rows = db.execute(select(VerificationJob.status, func.count()).where(VerificationJob.owner_id == owner)
                      .group_by(VerificationJob.status)).all()
    by_status = {s: n for s, n in rows}
    verdicts = db.execute(
        select(ClaimResult.verdict, func.count()).join(VerificationJob, VerificationJob.id == ClaimResult.job_id)
        .where(VerificationJob.owner_id == owner).group_by(ClaimResult.verdict)).all()
    return {"jobs_by_status": by_status, "claims_by_verdict": {v: n for v, n in verdicts},
            "total_jobs": sum(by_status.values())}


# --------------------------------------------------------------------------------- retention


def cleanup_expired_media(db: Session, storage: StorageBackend, now: datetime | None = None) -> int:
    """Delete stored media (and derived crops/frames) past their retention date. Job records are kept."""
    now = now or datetime.now(UTC)
    rows = db.execute(select(MediaAsset).where(MediaAsset.expires_at < now, MediaAsset.deleted_at.is_(None))).scalars().all()
    n = 0
    for m in rows:
        jobs = db.execute(select(VerificationJob.id).where(VerificationJob.media_id == m.id)).scalars().all()
        try:
            storage.delete(m.storage_key)
        except Exception:  # noqa: BLE001, S110 - best effort; record stays marked
            pass
        for jid in jobs:
            for prefix in ("crops", "frames"):
                try:
                    storage.delete_prefix(f"{prefix}/{jid.hex}")  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001, S110
                    pass
        m.deleted_at = now
        n += 1
    db.commit()
    return n
