"""Test configuration: isolated SQLite DB + storage per test session, demo (fake) models, no network."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="rc-tests-"))
os.environ.update({
    # CI sets RC_TEST_DATABASE_URL to run the same tests against PostgreSQL
    "RC_DATABASE_URL": os.environ.get("RC_TEST_DATABASE_URL") or f"sqlite:///{_TMP / 'test.db'}",
    "RC_STORAGE_ROOT": str(_TMP / "storage"),
    "RC_MODEL_BACKEND": "demo",
    "RC_WORKER_MODE": "external",
    "RC_LOG_JSON": "false",
    "RC_LOG_LEVEL": "WARNING",
    "RC_RATE_LIMIT_PER_MINUTE": "1000",
    "RC_SECRET_KEY": "test-secret-key-1234567890",
    "RC_DEMO_MANIFEST_DIR": str(_TMP / "no-demo"),
    "RC_FRONTEND_DIST": str(_TMP / "no-dist"),
    "RC_AUTH_REQUIRED": "false",
    "RC_API_KEYS": "",
})

from app.core.config import reset_settings_cache  # noqa: E402

reset_settings_cache()

from app.api.deps import rate_limiter  # noqa: E402
from app.services import fake_models  # noqa: E402
from app.services.model_manager import ModelRegistry, set_registry  # noqa: E402
from app.services.storage import LocalStorage, set_storage  # noqa: E402


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP


def _drain_queue() -> None:
    """Cancel jobs left queued/running by earlier tests so each test sees an empty queue."""
    from sqlalchemy import update
    from sqlalchemy.exc import OperationalError

    from app.persistence.db import session_scope
    from app.persistence.orm import JobStatus, VerificationJob

    try:
        with session_scope() as db:
            db.execute(update(VerificationJob).where(VerificationJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]))
                       .values(status=JobStatus.CANCELLED))
    except OperationalError:
        pass  # tables not created yet


@pytest.fixture(autouse=True)
def _isolation():
    fake_models.SCENES.clear()
    rate_limiter.reset()
    set_registry(ModelRegistry(force_demo=True))
    set_storage(LocalStorage(_TMP / "storage"))
    _drain_queue()
    yield
    fake_models.SCENES.clear()


@pytest.fixture(scope="session")
def app():
    from app.main import create_app

    return create_app(start_worker=False)


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def run_jobs():
    """Process all queued jobs synchronously (what the worker process does)."""
    from app.workers.runner import run_one

    def _run(max_jobs: int = 20) -> list[str]:
        out = []
        for _ in range(max_jobs):
            r = run_one("test-worker")
            if r is None:
                break
            out.append(r)
        return out

    return _run


@pytest.fixture()
def scenes():
    return fake_models.SCENES
