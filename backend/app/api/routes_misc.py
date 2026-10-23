"""History, signed file access, health, system info and evaluation endpoints."""

from __future__ import annotations

import mimetypes
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.api.deps import APIError, Principal, get_principal
from app.core.config import get_settings
from app.persistence.db import get_db
from app.persistence.orm import EvaluationMetric, EvaluationRun, JobStatus
from app.schemas.api import HistoryItem, Page
from app.services import jobs as svc
from app.services.model_manager import get_registry, ml_dependencies_available
from app.services.storage import StorageError, get_storage, validate_key, verify_signature

router = APIRouter()


# ------------------------------------------------------------------------------------- history
@router.get("/api/v1/history", response_model=Page[HistoryItem], tags=["history"], summary="Previous jobs")
def list_history(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
                 status: str | None = Query(None), media_type: str | None = Query(None, pattern="^(image|video)$"),
                 q: str | None = Query(None, max_length=100), demo: bool | None = Query(None),
                 p: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> Page[HistoryItem]:
    if status and status not in JobStatus.ALL:
        raise APIError(422, "invalid_status", f"status must be one of {', '.join(JobStatus.ALL)}.")
    items, total = svc.history(db, p.owner_id, svc.Signer(get_settings(), p.owner_id), limit=limit, offset=offset,
                               status=status, media_type=media_type, q=q, demo=demo)
    return Page[HistoryItem](items=items, total=total, limit=limit, offset=offset)


@router.get("/api/v1/history/stats", tags=["history"], summary="Aggregate statistics over the caller's jobs")
def history_stats(p: Principal = Depends(get_principal), db: Session = Depends(get_db)) -> dict[str, Any]:
    return svc.history_stats(db, p.owner_id)


# ------------------------------------------------------------------------------------- files
@router.get("/api/v1/files", tags=["files"], summary="Signed, expiring access to stored media, frames and crops")
def get_file(key: str = Query(..., max_length=255), owner: str = Query(..., max_length=64),
             exp: int = Query(...), sig: str = Query(..., max_length=64)) -> Response:
    try:
        validate_key(key)
    except StorageError as exc:
        raise APIError(400, "invalid_key", "Invalid file reference.") from exc
    if not verify_signature(key, owner, exp, sig, get_settings().secret_key.get_secret_value()):
        raise APIError(403, "invalid_signature", "The file link is invalid or has expired.")
    try:
        data = get_storage().get(key)
    except StorageError as exc:
        raise APIError(404, "file_not_found", "File not found (it may have been deleted by the retention policy).") from exc
    ctype = mimetypes.guess_type(key)[0] or "application/octet-stream"
    return Response(content=data, media_type=ctype, headers={
        "Cache-Control": "private, max-age=600", "X-Content-Type-Options": "nosniff",
        "Content-Disposition": "inline", "Content-Security-Policy": "default-src 'none'; sandbox"})


# ------------------------------------------------------------------------------------- health
@router.get("/api/v1/health/live", tags=["health"], summary="Liveness probe")
def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/api/v1/health/ready", tags=["health"], summary="Readiness probe (database, storage, model backend)")
def ready(db: Session = Depends(get_db)) -> Response:
    import json

    checks: dict[str, Any] = {}
    ok = True
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["database"] = f"error: {type(exc).__name__}"
        ok = False
    try:
        st = get_storage()
        checks["storage"] = f"ok ({st.name})"
    except Exception as exc:  # noqa: BLE001
        checks["storage"] = f"error: {type(exc).__name__}"
        ok = False
    reg = get_registry()
    s = get_settings()
    if not reg.demo and not ml_dependencies_available():
        checks["models"] = "error: ML dependencies missing (install the 'ml' extra or set RC_MODEL_BACKEND=demo)"
        ok = False
    else:
        checks["models"] = "demo" if reg.demo else "real (lazy-loaded on first job)"
    checks["worker_mode"] = s.worker_mode.value
    return Response(content=json.dumps({"status": "ready" if ok else "not_ready", "checks": checks}),
                    status_code=200 if ok else 503, media_type="application/json")


@router.get("/api/v1/system/info", tags=["system"], summary="Model backend, limits and privacy information")
def system_info(p: Principal = Depends(get_principal)) -> dict[str, Any]:
    s = get_settings()
    reg = get_registry()
    st = reg.status()
    return {
        "demo_mode": reg.demo,
        "models": st,
        "configured_models": {"detector": s.detector_weights, "grounding": s.grounding_model_id,
                              "vlm_verifier": s.vlm_model_id if s.vlm_provider.value == "smolvlm" else s.anthropic_model,
                              "vlm_generator": s.generator_model_id, "vlm_provider": s.vlm_provider.value},
        "limits": {"max_image_mb": s.max_image_bytes // (1024 * 1024), "max_video_mb": s.max_video_bytes // (1024 * 1024),
                   "max_video_seconds": s.max_video_seconds, "max_description_chars": s.max_description_chars,
                   "max_claims_per_job": s.max_claims_per_job, "max_sampled_frames": s.video_max_sampled_frames},
        "privacy": {
            "processing": "external" if st["external_provider_enabled"] else "local",
            "statement": ("Images/crops are sent to the Anthropic API for VLM verification."
                          if st["external_provider_enabled"] else
                          "All processing runs on this server; no media is sent to external services."),
            "media_retention_days": s.media_retention_days,
        },
        "auth_required": s.auth_required,
        "authenticated": p.authenticated,
    }


# ------------------------------------------------------------------------------------- evaluation
def _run_dict(r: EvaluationRun, include_metrics: bool = True) -> dict[str, Any]:
    d = {"run_id": r.run_id, "experiment": r.experiment, "system": r.system, "dataset": r.dataset,
         "dataset_version": r.dataset_version, "split": r.split, "seed": r.seed, "status": r.status,
         "simulated": r.simulated, "n_examples": r.n_examples, "model_versions": r.model_versions,
         "created_at": r.created_at.isoformat(), "finished_at": r.finished_at.isoformat() if r.finished_at else None,
         "config": r.config}
    if include_metrics:
        d["metrics"] = r.metrics
    return d


@router.get("/api/v1/evaluations", tags=["evaluation"], summary="Evaluation runs (filter by experiment/dataset)")
def list_evaluations(experiment: str | None = Query(None, max_length=128), dataset: str | None = Query(None, max_length=128),
                     limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0),
                     db: Session = Depends(get_db), p: Principal = Depends(get_principal)) -> dict[str, Any]:
    q = select(EvaluationRun)
    if experiment:
        q = q.where(EvaluationRun.experiment == experiment)
    if dataset:
        q = q.where(EvaluationRun.dataset == dataset)
    rows = db.execute(q.order_by(EvaluationRun.created_at.desc())).scalars().all()
    exps = sorted({(r.experiment, r.dataset, r.split) for r in rows})
    return {"items": [_run_dict(r) for r in rows[offset:offset + limit]], "total": len(rows), "limit": limit,
            "offset": offset,
            "experiments": [{"experiment": e, "dataset": d, "split": sp} for e, d, sp in exps]}


@router.get("/api/v1/evaluations/{run_id}", tags=["evaluation"], summary="One evaluation run with all metrics")
def get_evaluation(run_id: str, db: Session = Depends(get_db), p: Principal = Depends(get_principal)) -> dict[str, Any]:
    r = db.execute(select(EvaluationRun).where(EvaluationRun.run_id == run_id)).scalar_one_or_none()
    if r is None:
        try:
            r = db.get(EvaluationRun, uuid.UUID(run_id))
        except ValueError:
            r = None
    if r is None:
        raise APIError(404, "run_not_found", "Evaluation run not found.")
    d = _run_dict(r)
    d["metric_rows"] = [{"name": m.name, "category": m.category, "value": m.value, "ci_low": m.ci_low,
                         "ci_high": m.ci_high} for m in db.execute(select(EvaluationMetric).where(
                             EvaluationMetric.run_id == r.id)).scalars()]
    return d
