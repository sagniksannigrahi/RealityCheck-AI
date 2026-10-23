"""Verification job endpoints."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.api.deps import APIError, Principal, get_principal, rate_limiter
from app.core.config import get_settings
from app.observability.logging import request_id_var
from app.persistence.db import get_db
from app.persistence.orm import JobStatus, VerificationJob
from app.schemas.api import (
    ClaimOut,
    EvidenceOut,
    FrameOut,
    JobCreated,
    JobSettings,
    JobSummary,
    Page,
    Report,
    VerificationMode,
)
from app.services import jobs as svc
from app.services.media_validation import MediaValidationError
from app.services.model_manager import get_registry
from app.services.report_markdown import report_to_markdown
from app.services.storage import get_storage

router = APIRouter(prefix="/api/v1/verifications", tags=["verifications"])


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    chunks, total = [], 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise APIError(413, "file_too_large", f"File exceeds the {limit // (1024 * 1024)} MB upload limit.")
        chunks.append(chunk)
    return b"".join(chunks)


def _job_or_404(db: Session, job_id: str, p: Principal) -> VerificationJob:
    job = svc.get_owned_job(db, job_id, p.owner_id)
    if job is None:
        raise APIError(404, "job_not_found", "Verification job not found.")
    return job


@router.post("", status_code=202, response_model=JobCreated, summary="Submit an image or video for verification")
async def create_verification(
    request: Request,
    file: UploadFile = File(..., description="Image (JPEG/PNG/WebP/GIF) or video (MP4/MOV/WebM/AVI)."),
    description: str | None = Form(None, description="Description to verify (required in verify_description mode)."),
    mode: VerificationMode = Form(VerificationMode.VERIFY_DESCRIPTION),
    settings: str | None = Form(None, description="Optional JSON object of JobSettings."),
    p: Principal = Depends(get_principal),
    db: Session = Depends(get_db),
) -> JSONResponse:
    s = get_settings()
    rate_limiter.check(f"create:{p.owner_id}", s.rate_limit_per_minute)
    try:
        js = JobSettings.model_validate(json.loads(settings)) if settings else JobSettings()
    except (ValueError, ValidationError) as exc:
        raise APIError(422, "invalid_settings", "The settings field must be a valid JobSettings JSON object.",
                       {"reason": str(exc)[:300]}) from exc
    if not (js.use_detector or js.use_grounding or js.use_vlm):
        raise APIError(422, "invalid_settings", "At least one evidence component must be enabled.")
    data = await _read_limited(file, max(s.max_image_bytes, s.max_video_bytes))
    try:
        job = svc.create_job(db, owner=p.owner_id, data=data, filename=file.filename, description=description,
                             mode=mode, job_settings=js, storage=get_storage(), settings=s,
                             demo_mode=get_registry().demo, request_id=request_id_var.get())
    except MediaValidationError as exc:
        status = 413 if exc.code == "file_too_large" else (415 if exc.code == "unsupported_media_type" else 422)
        raise APIError(status, exc.code, exc.message) from exc
    jid = str(job.id)
    body = JobCreated(job_id=jid, status=job.status, mode=mode, media_type=job.media_type, demo_mode=job.demo_mode,
                      links={"self": f"/api/v1/verifications/{jid}", "claims": f"/api/v1/verifications/{jid}/claims",
                             "report": f"/api/v1/verifications/{jid}/report"})
    return JSONResponse(status_code=202, content=body.model_dump(mode="json"),
                        headers={"Location": f"/api/v1/verifications/{jid}"})


@router.get("/{job_id}", response_model=JobSummary, summary="Job status and summary")
def get_verification(job_id: str, p: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> JobSummary:
    job = _job_or_404(db, job_id, p)
    return svc.job_summary(job, svc.Signer(get_settings(), p.owner_id))


@router.get("/{job_id}/claims", response_model=Page[ClaimOut], summary="Atomic claims and verdicts")
def get_claims(job_id: str, limit: int = Query(100, ge=1, le=200), offset: int = Query(0, ge=0),
               verdict: str | None = Query(None, max_length=32), p: Principal = Depends(get_principal),
               db: Session = Depends(get_db)) -> Page[ClaimOut]:
    job = _job_or_404(db, job_id, p)
    items = svc.claims_out(db, job)
    if verdict:
        items = [c for c in items if c.result and c.result.verdict == verdict.upper()]
    return Page[ClaimOut](items=items[offset:offset + limit], total=len(items), limit=limit, offset=offset)


@router.get("/{job_id}/evidence", response_model=Page[EvidenceOut], summary="Evidence references and metadata")
def get_evidence(job_id: str, claim_id: str | None = Query(None, pattern=r"^c\d{1,4}$"),
                 limit: int = Query(200, ge=1, le=1000), offset: int = Query(0, ge=0),
                 p: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> Page[EvidenceOut]:
    job = _job_or_404(db, job_id, p)
    items = svc.evidence_out(db, job, svc.Signer(get_settings(), p.owner_id), claim_id)
    return Page[EvidenceOut](items=items[offset:offset + limit], total=len(items), limit=limit, offset=offset)


@router.get("/{job_id}/frames", response_model=list[FrameOut], summary="Sampled video frames with timestamps")
def get_frames(job_id: str, p: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> list[FrameOut]:
    job = _job_or_404(db, job_id, p)
    if job.media_type != "video":
        raise APIError(409, "not_a_video", "Frames are only available for video jobs.")
    return svc.frames_out(db, job, svc.Signer(get_settings(), p.owner_id))


@router.get("/{job_id}/report", summary="Structured verification report (JSON or Markdown)",
            response_model=Report, responses={200: {"content": {"text/markdown": {}}}})
def get_report(job_id: str, format: str = Query("json", pattern="^(json|markdown)$"),
               p: Principal = Depends(get_principal), db: Session = Depends(get_db)):
    job = _job_or_404(db, job_id, p)
    if job.status != JobStatus.COMPLETED:
        raise APIError(409, "job_not_completed", f"Report is available once the job completes (status: {job.status}).")
    rep = svc.build_report(db, job, svc.Signer(get_settings(), p.owner_id))
    if format == "markdown":
        return PlainTextResponse(report_to_markdown(rep), media_type="text/markdown",
                                 headers={"Content-Disposition": f'attachment; filename="realitycheck-{job.id}.md"'})
    return rep


@router.post("/{job_id}/cancel", response_model=JobSummary, summary="Request cancellation")
def cancel(job_id: str, p: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> JobSummary:
    job = _job_or_404(db, job_id, p)
    if job.status in JobStatus.TERMINAL:
        raise APIError(409, "job_finished", f"Job already finished with status {job.status}.")
    job.cancel_requested = True
    if job.status == JobStatus.QUEUED:
        job.status, job.finished_at, job.progress_stage = JobStatus.CANCELLED, datetime.now(UTC), "cancelled"
    db.commit()
    return svc.job_summary(job, svc.Signer(get_settings(), p.owner_id))


@router.delete("/{job_id}", status_code=204, summary="Delete a job, its results and its stored media")
def delete_job(job_id: str, p: Principal = Depends(get_principal), db: Session = Depends(get_db)):
    job = _job_or_404(db, job_id, p)
    if job.status == JobStatus.RUNNING:
        raise APIError(409, "job_running", "Cancel the job before deleting it.")
    storage = get_storage()
    media = job.media
    for prefix in ("crops", "frames"):
        try:
            storage.delete_prefix(f"{prefix}/{job.id.hex}")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001, S110 - best effort
            pass
    db.delete(job)
    db.flush()
    from sqlalchemy import func, select

    remaining = db.execute(select(func.count()).select_from(VerificationJob).where(
        VerificationJob.media_id == media.id)).scalar_one()
    if remaining == 0:
        try:
            storage.delete(media.storage_key)
        except Exception:  # noqa: BLE001, S110
            pass
        db.delete(media)
    db.commit()
    return PlainTextResponse(status_code=204, content="")
