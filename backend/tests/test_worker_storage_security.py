"""Queue/worker reliability, storage isolation, signed URLs, secret redaction and persistence."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from app.observability.logging import JsonFormatter, redact, redact_all, register_secret
from app.persistence.db import session_scope
from app.persistence.orm import Claim, EvidenceRecord, JobStatus, ProcessingError, VerificationJob
from app.services.fake_models import FakeScene
from app.services.storage import LocalStorage, StorageError, new_key, sign, validate_key, verify_signature
from app.workers import runner
from app.workers.queue import claim_next, requeue_stale
from tests.doubles import make_image, obj, register_scene

DOG = (0.1, 0.3, 0.4, 0.8)


def _submit(client, scenes, desc="A red dog."):
    png = make_image([("red", DOG)])
    register_scene(scenes, png, FakeScene(objects=[obj("dog", DOG)]))
    r = client.post("/api/v1/verifications", files={"file": ("a.png", png, "image/png")},
                    data={"description": desc, "mode": "verify_description"})
    assert r.status_code == 202, r.text
    return uuid.UUID(r.json()["job_id"])


# ------------------------------------------------------------------------------------- queue / worker
def test_claim_next_is_exclusive(client, scenes):
    jid = _submit(client, scenes)
    with session_scope() as db:
        assert claim_next(db, "w1") == jid
    with session_scope() as db:
        assert claim_next(db, "w2") is None
        j = db.get(VerificationJob, jid)
        assert j.status == JobStatus.RUNNING and j.locked_by == "w1" and j.attempts == 1


def test_stale_lock_is_recovered_then_failed_after_max_attempts(client, scenes):
    jid = _submit(client, scenes)
    with session_scope() as db:
        claim_next(db, "dead-worker")
        db.get(VerificationJob, jid).locked_at = datetime.now(UTC) - timedelta(hours=2)
    with session_scope() as db:
        assert requeue_stale(db, stale_after_s=60, max_attempts=2) == 1
        assert db.get(VerificationJob, jid).status == JobStatus.QUEUED
    with session_scope() as db:
        claim_next(db, "dead-again")
        db.get(VerificationJob, jid).locked_at = datetime.now(UTC) - timedelta(hours=2)
    with session_scope() as db:
        requeue_stale(db, stale_after_s=60, max_attempts=2)
        j = db.get(VerificationJob, jid)
        assert j.status == JobStatus.FAILED and j.error_code == "worker_lost"


def test_transient_failure_is_retried_then_succeeds(client, scenes, run_jobs, monkeypatch):
    jid = _submit(client, scenes)
    calls = {"n": 0}
    from app.verification.pipeline import VerificationPipeline

    orig = VerificationPipeline.run_image

    def flaky(self, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient failure api_key=sk-ant-SECRETSECRETSECRET")
        return orig(self, *a, **kw)

    monkeypatch.setattr(VerificationPipeline, "run_image", flaky)
    assert run_jobs() == ["queued", "completed"]
    with session_scope() as db:
        j = db.get(VerificationJob, jid)
        assert j.status == JobStatus.COMPLETED and j.attempts == 2
        errs = db.execute(select(ProcessingError).where(ProcessingError.job_id == jid)).scalars().all()
        assert len(errs) == 1 and "SECRET" not in errs[0].message and "[REDACTED]" in errs[0].message


def test_permanent_failure_is_not_retried_and_message_is_safe(client, scenes, run_jobs, monkeypatch):
    jid = _submit(client, scenes)
    from app.verification.pipeline import VerificationPipeline

    def boom(self, *a, **kw):
        raise RuntimeError("internal detail /Users/someone/secret/path")

    monkeypatch.setattr(VerificationPipeline, "run_image", boom)
    monkeypatch.setattr(runner.get_settings(), "job_max_attempts", 1)
    assert run_jobs() == ["failed"]
    job = client.get(f"/api/v1/verifications/{jid}").json()
    assert job["error"]["code"] == "internal_error"
    assert "/Users/" not in job["error"]["message"]


def test_model_unavailable_is_classified():
    from app.services.model_manager import ModelUnavailableError

    assert runner.classify(ModelUnavailableError("x")) == ("model_unavailable", False)
    assert runner.classify(runner.JobTimeout("x")) == ("timeout", False)
    assert runner.classify(ValueError("x")) == ("internal_error", True)


def test_timeout_is_enforced(client, scenes, run_jobs, monkeypatch):
    jid = _submit(client, scenes)
    monkeypatch.setattr(runner.get_settings(), "job_timeout_seconds", -1)
    assert run_jobs() == ["failed"]
    assert client.get(f"/api/v1/verifications/{jid}").json()["error"]["code"] == "timeout"


def test_cancellation_while_running(client, scenes, run_jobs, monkeypatch):
    jid = _submit(client, scenes, "A red dog. A blue ball. A green car.")
    orig = runner._set_stage
    seen = {"n": 0}

    def stage_then_cancel(job_id, stage):
        seen["n"] += 1
        if seen["n"] == 3:
            with session_scope() as db:
                db.get(VerificationJob, job_id).cancel_requested = True
        return orig(job_id, stage)

    monkeypatch.setattr(runner, "_set_stage", stage_then_cancel)
    assert run_jobs() == ["cancelled"]
    with session_scope() as db:
        assert db.get(VerificationJob, jid).status == JobStatus.CANCELLED


def test_results_persist_and_cascade_on_delete(client, scenes, run_jobs):
    jid = _submit(client, scenes)
    run_jobs()
    with session_scope() as db:
        assert db.execute(select(Claim).where(Claim.job_id == jid)).scalars().first() is not None
        assert db.execute(select(EvidenceRecord).where(EvidenceRecord.job_id == jid)).scalars().first() is not None
    assert client.delete(f"/api/v1/verifications/{jid}").status_code == 204
    with session_scope() as db:
        assert db.execute(select(Claim).where(Claim.job_id == jid)).scalars().first() is None
        assert db.execute(select(EvidenceRecord).where(EvidenceRecord.job_id == jid)).scalars().first() is None


def test_retention_cleanup_deletes_media_but_keeps_records(client, scenes, run_jobs):
    from app.services.jobs import cleanup_expired_media
    from app.services.storage import get_storage

    jid = _submit(client, scenes)
    run_jobs()
    with session_scope() as db:
        job = db.get(VerificationJob, jid)
        key = job.media.storage_key
        n = cleanup_expired_media(db, get_storage(), now=datetime.now(UTC) + timedelta(days=365))
        assert n >= 1
    assert not get_storage().exists(key)
    job = client.get(f"/api/v1/verifications/{jid}").json()
    assert job["status"] == "completed" and job["media"]["url"] is None


# ------------------------------------------------------------------------------------- storage
@pytest.mark.parametrize("key", ["../etc/passwd", "/etc/passwd", "media/../../x.jpg", "media\\x.jpg", "other/x.jpg",
                                 "media/x.exe", "media/a b.jpg", "", "media/" + "a" * 100 + ".jpg"])
def test_invalid_keys_rejected(key):
    with pytest.raises(StorageError):
        validate_key(key)


def test_local_storage_confined_to_root(tmp_path):
    st = LocalStorage(tmp_path / "root")
    k = new_key("media", "jpg")
    st.put(k, b"data")
    assert st.get(k) == b"data" and st.exists(k)
    assert (tmp_path / "root") in st.local_path(k).parents
    with pytest.raises(StorageError):
        st.get("../outside.jpg")
    assert not st.exists("../../x.jpg")
    st.delete(k)
    assert not st.exists(k)


def test_new_key_sanitizes_parts():
    k = new_key("crops", "jpg", "../../evil")
    assert ".." not in k and validate_key(k)


def test_signature_verification():
    s = sign("media/a.jpg", "o", 9999999999, "k")
    assert verify_signature("media/a.jpg", "o", 9999999999, s, "k")
    assert not verify_signature("media/b.jpg", "o", 9999999999, s, "k")
    assert not verify_signature("media/a.jpg", "o", 9999999999, s, "other-secret")
    assert not verify_signature("media/a.jpg", "o", 1, sign("media/a.jpg", "o", 1, "k"), "k")


# ------------------------------------------------------------------------------------- redaction
@pytest.mark.parametrize("text", [
    "key sk-ant-api03-abcdefghijklmnop", "Authorization: Bearer abcdefghijklmnop123",
    "api_key=supersecretvalue", "postgresql+psycopg://user:hunter2@db:5432/rc", "AKIAABCDEFGHIJKLMNOP",
])
def test_secret_redaction(text):
    out = redact(text)
    for s in ["abcdefghijklmnop", "supersecretvalue", "hunter2", "ABCDEFGHIJKLMNOP"]:
        assert s not in out


def test_registered_secret_redacted_in_logs():
    import logging

    register_secret("my-literal-api-key-42")
    rec = logging.LogRecord("x", logging.INFO, "f", 1, "using my-literal-api-key-42 now", (), None)
    assert "my-literal-api-key-42" not in JsonFormatter().format(rec)
    assert "[REDACTED]" in redact_all("token my-literal-api-key-42")
