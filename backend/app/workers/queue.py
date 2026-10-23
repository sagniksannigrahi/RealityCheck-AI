"""Database-backed job queue.

Design decision: the queue lives in the same database as the job records, so enqueueing is transactional
with job creation and no extra service is required. PostgreSQL uses ``FOR UPDATE SKIP LOCKED`` so several
worker processes can claim jobs safely; SQLite (single-node dev) uses an atomic compare-and-set update.
Redis was evaluated but not adopted: it would duplicate state already held transactionally in the DB.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.persistence.orm import JobStatus, VerificationJob


def claim_next(db: Session, worker_id: str) -> uuid.UUID | None:
    now = datetime.now(UTC)
    dialect = db.get_bind().dialect.name
    if dialect == "postgresql":
        row = db.execute(
            select(VerificationJob.id).where(VerificationJob.status == JobStatus.QUEUED)
            .order_by(VerificationJob.created_at).limit(1).with_for_update(skip_locked=True)
        ).scalar_one_or_none()
        if row is None:
            db.rollback()
            return None
        db.execute(update(VerificationJob).where(VerificationJob.id == row).values(
            status=JobStatus.RUNNING, locked_by=worker_id, locked_at=now, started_at=now,
            attempts=VerificationJob.attempts + 1, progress_stage="starting"))
        db.commit()
        return row
    # generic compare-and-set (SQLite)
    for _ in range(5):
        cand = db.execute(select(VerificationJob.id).where(VerificationJob.status == JobStatus.QUEUED)
                          .order_by(VerificationJob.created_at).limit(1)).scalar_one_or_none()
        if cand is None:
            return None
        res = db.execute(update(VerificationJob).where(VerificationJob.id == cand,
                                                       VerificationJob.status == JobStatus.QUEUED).values(
            status=JobStatus.RUNNING, locked_by=worker_id, locked_at=now, started_at=now,
            attempts=VerificationJob.attempts + 1, progress_stage="starting"))
        db.commit()
        if res.rowcount == 1:
            return cand
    return None


def requeue_stale(db: Session, stale_after_s: int, max_attempts: int) -> int:
    """Recover jobs whose worker died: requeue if attempts remain, else fail them."""
    cutoff = datetime.now(UTC) - timedelta(seconds=stale_after_s)
    stale = db.execute(select(VerificationJob).where(VerificationJob.status == JobStatus.RUNNING,
                                                     VerificationJob.locked_at < cutoff)).scalars().all()
    for j in stale:
        if j.attempts < max_attempts:
            j.status, j.locked_by, j.locked_at, j.progress_stage = JobStatus.QUEUED, None, None, "requeued_after_stale_lock"
        else:
            j.status, j.error_code = JobStatus.FAILED, "worker_lost"
            j.error_message = "The worker processing this job stopped responding."
            j.finished_at = datetime.now(UTC)
    db.commit()
    return len(stale)
