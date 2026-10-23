"""Processes one verification job end to end, with cancellation, timeouts and failure classification."""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from datetime import UTC, datetime

from app.core.config import Settings, get_settings
from app.core.imaging import ImageDecodeError
from app.observability.logging import get_logger, job_id_var, redact_all
from app.persistence.db import session_scope
from app.persistence.orm import JobStatus, VerificationJob
from app.schemas.api import JobSettings
from app.services import jobs as job_service
from app.services.media_validation import MediaValidationError, decode_validated_image
from app.services.model_manager import ModelUnavailableError, get_registry
from app.services.storage import StorageBackend, StorageError, get_storage
from app.verification.pipeline import VerificationPipeline
from app.verification.policy import VerificationPolicy

log = get_logger(__name__)


class JobCancelled(Exception):
    pass


class JobTimeout(Exception):
    pass


# Failure classification: (error code, retryable)
def classify(exc: BaseException) -> tuple[str, bool]:
    if isinstance(exc, JobCancelled):
        return "cancelled", False
    if isinstance(exc, JobTimeout):
        return "timeout", False
    if isinstance(exc, MediaValidationError):
        return exc.code, False
    if isinstance(exc, ImageDecodeError):
        return "invalid_image", False
    if type(exc).__name__ == "VideoDecodeError":
        return "invalid_video", False
    if isinstance(exc, ModelUnavailableError):
        return "model_unavailable", False
    if isinstance(exc, StorageError):
        return "storage_error", True
    if isinstance(exc, MemoryError):
        return "out_of_memory", False
    return "internal_error", True


def policy_for(job_settings: JobSettings, base: VerificationPolicy) -> VerificationPolicy:
    p = base.model_copy(deep=True)
    p.use_abstention = job_settings.abstention
    p.use_disagreement = job_settings.disagreement_analysis
    p.components.detector = job_settings.use_detector
    p.components.grounding = job_settings.use_grounding
    p.components.vlm = job_settings.use_vlm
    return p


def settings_for(job_settings: JobSettings, base: Settings) -> Settings:
    upd = {}
    if job_settings.sampling_strategy:
        upd["video_sampling_strategy"] = job_settings.sampling_strategy
    if job_settings.max_frames:
        upd["video_max_sampled_frames"] = min(job_settings.max_frames, base.video_max_sampled_frames * 2, 32)
    return base.model_copy(update=upd) if upd else base


def _set_stage(job_id: uuid.UUID, stage: str) -> bool:
    """Persist progress; returns True if cancellation was requested."""
    with session_scope() as db:
        j = db.get(VerificationJob, job_id)
        if j is None:
            return True
        j.progress_stage = stage[:64]
        j.locked_at = datetime.now(UTC)  # heartbeat
        return bool(j.cancel_requested)


def process_job(job_id: uuid.UUID, *, storage: StorageBackend | None = None, settings: Settings | None = None,
                registry=None) -> str:
    settings = settings or get_settings()
    storage = storage or get_storage()
    registry = registry or get_registry()
    token = job_id_var.set(str(job_id))
    started = time.monotonic()
    try:
        with session_scope() as db:
            job = db.get(VerificationJob, job_id)
            if job is None:
                return "missing"
            if job.cancel_requested:
                job.status, job.finished_at, job.progress_stage = JobStatus.CANCELLED, datetime.now(UTC), "cancelled"
                return JobStatus.CANCELLED
            js = JobSettings.model_validate(job.settings or {})
            media_key, media_type = job.media.storage_key, job.media_type
            description, mode, attempt = job.description_original, job.mode, job.attempts

        eff_settings = settings_for(js, settings)
        base_policy = VerificationPolicy.load(settings.policy_path)
        pipeline = VerificationPipeline(registry, policy=policy_for(js, base_policy), storage=storage,
                                        settings=eff_settings)
        stage_holder = {"stage": "starting"}

        def check(stage: str) -> None:
            stage_holder["stage"] = stage
            if time.monotonic() - started > settings.job_timeout_seconds:
                raise JobTimeout(f"Job exceeded the {settings.job_timeout_seconds}s processing limit.")
            if _set_stage(job_id, stage):
                raise JobCancelled("Cancelled by user.")

        generate = mode == "generate_and_verify"
        log.info("job started", extra={"media_type": media_type, "mode": mode, "attempt": attempt})
        try:
            check("loading_media")
            data = storage.get(media_key)
            if media_type == "image":
                img = decode_validated_image(data, settings)
                result = pipeline.run_image(img, description, generate=generate, job_id=job_id.hex, check=check)
            else:
                path = storage.local_path(media_key)
                tmp = None
                if path is None:
                    tmp = tempfile.NamedTemporaryFile(suffix=os.path.splitext(media_key)[1], delete=False)  # noqa: SIM115
                    tmp.write(data)
                    tmp.close()
                    path = tmp.name
                try:
                    result = pipeline.run_video(path, description, generate=generate, job_id=job_id.hex, check=check)
                finally:
                    if tmp is not None:
                        os.unlink(tmp.name)
            check("persisting")
        except BaseException as exc:  # noqa: BLE001 - classify everything, re-raise nothing to the worker loop
            if isinstance(exc, KeyboardInterrupt | SystemExit):
                raise
            return _handle_failure(job_id, exc, stage_holder["stage"], settings)

        with session_scope() as db:
            job = db.get(VerificationJob, job_id)
            job_service.persist_result(db, job, result)
            job.status, job.progress_stage = JobStatus.COMPLETED, "completed"
            job.finished_at = datetime.now(UTC)
            job.locked_by = job.locked_at = None
            job.error_code = job.error_message = None
        log.info("job completed", extra={"elapsed_s": round(time.monotonic() - started, 2)})
        return JobStatus.COMPLETED
    finally:
        job_id_var.reset(token)


def _handle_failure(job_id: uuid.UUID, exc: BaseException, stage: str, settings: Settings) -> str:
    code, retryable = classify(exc)
    with session_scope() as db:
        job: VerificationJob | None = db.get(VerificationJob, job_id)
        if job is None:
            return "missing"
        job_service.record_error(db, job, stage, exc, job.attempts)
        if code == "cancelled":
            job.status, job.progress_stage = JobStatus.CANCELLED, "cancelled"
        elif retryable and job.attempts < settings.job_max_attempts:
            job.status, job.progress_stage = JobStatus.QUEUED, f"retry_after_{code}"
            job.locked_by = job.locked_at = None
            log.warning("job failed; requeued", extra={"code": code, "stage": stage})
            return JobStatus.QUEUED
        else:
            job.status, job.progress_stage = JobStatus.FAILED, "failed"
            job.error_code = code
            # user-facing message: safe, redacted, no internals for unexpected errors
            job.error_message = (redact_all(str(exc))[:500] if code != "internal_error"
                                 else "An internal error occurred while processing this job.")
        job.finished_at = datetime.now(UTC)
        job.locked_by = job.locked_at = None
        log.warning("job ended", extra={"code": code, "stage": stage, "status": job.status})
        return job.status


def run_one(worker_id: str, db_factory=session_scope) -> str | None:
    from app.workers.queue import claim_next

    with db_factory() as db:  # type: Session
        jid = claim_next(db, worker_id)
    if jid is None:
        return None
    return process_job(jid)
