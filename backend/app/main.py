"""FastAPI application entrypoint: ``uvicorn app.main:app``."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.deps import APIError
from app.api.routes_misc import router as misc_router
from app.api.routes_verifications import router as verif_router
from app.core.config import WorkerMode, get_settings
from app.observability.logging import (
    configure_logging,
    get_logger,
    redact_all,
    register_secret,
    request_id_var,
)

log = get_logger("app")


def run_migrations(url: str) -> None:
    from alembic import command
    from alembic.config import Config

    root = Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "backend" / "migrations"))
    cfg.attributes["database_url"] = url
    command.upgrade(cfg, "head")


def _error(status: int, code: str, message: str, details=None) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "request_id": request_id_var.get()}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


def create_app(*, start_worker: bool | None = None, migrate: bool = True) -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    for k in settings.api_key_list:
        register_secret(k)
    register_secret(settings.secret_key.get_secret_value())
    if settings.environment == "production" and settings.secret_key.get_secret_value() == "dev-insecure-change-me":
        raise RuntimeError("RC_SECRET_KEY must be set in production.")

    worker_holder: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from app.persistence.db import init_engine

        init_engine()
        if migrate:
            run_migrations(settings.database_url)
        run_worker = start_worker if start_worker is not None else settings.worker_mode == WorkerMode.INLINE
        if run_worker:
            from app.workers.worker import Worker

            w = Worker(name="inline-worker")
            w.start_thread()
            worker_holder["w"] = w
            log.info("inline worker started")
        yield
        if "w" in worker_holder:
            worker_holder["w"].stop()
            worker_holder["w"].join(timeout=10)
        from app.services.model_manager import get_registry

        get_registry().unload()

    app = FastAPI(
        title="RealityCheck AI",
        version="0.1.0",
        description="Evidence-grounded verification of claims in image and video descriptions.",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=False,
                       allow_methods=["GET", "POST", "DELETE"], allow_headers=["X-API-Key", "Authorization",
                                                                               "Content-Type", "X-Request-ID"],
                       expose_headers=["X-Request-ID", "Location"])

    max_body = max(settings.max_image_bytes, settings.max_video_bytes) + 1024 * 1024

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id", "")
        if not rid or len(rid) > 64 or not all(c.isalnum() or c in "-_" for c in rid):
            rid = uuid.uuid4().hex
        token = request_id_var.set(rid)
        try:
            cl = request.headers.get("content-length")
            if cl and cl.isdigit() and int(cl) > max_body:
                resp = _error(413, "request_too_large", "Request body exceeds the upload limit.")
            else:
                resp = await call_next(request)
        finally:
            request_id_var.reset(token)
        resp.headers["X-Request-ID"] = rid
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        if request.url.path.startswith("/api/") and request.url.path not in ("/api/docs", "/api/redoc"):
            resp.headers.setdefault("Cache-Control", "no-store")
        return resp

    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError):
        resp = _error(exc.status, exc.code, exc.message, exc.details)
        if exc.status == 429 and isinstance(exc.details, dict):
            resp.headers["Retry-After"] = str(exc.details.get("retry_after_s", 60))
        return resp

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        details = [{"loc": [str(x) for x in e.get("loc", [])], "msg": e.get("msg"), "type": e.get("type")}
                   for e in exc.errors()][:20]
        return _error(422, "validation_error", "Request validation failed.", details)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error(exc.status_code, code, str(exc.detail) if exc.status_code != 404 else "Not found.")

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.error("unhandled error: %s", redact_all(f"{type(exc).__name__}: {exc}"))
        return _error(500, "internal_error", "An internal error occurred.")

    app.include_router(verif_router)
    app.include_router(misc_router)

    dist = settings.frontend_dist
    if dist.is_dir() and (dist / "index.html").exists():
        dist_root = dist.resolve()

        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa(full_path: str):
            if full_path.startswith("api/"):
                return _error(404, "not_found", "Not found.")
            candidate = (dist_root / full_path).resolve()
            if full_path and dist_root in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist_root / "index.html")

    return app


app = create_app()
