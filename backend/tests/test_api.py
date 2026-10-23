"""API integration tests: upload -> queued job -> worker -> persisted results -> API/report."""

import json

import pytest

from app.core.config import get_settings
from app.services.fake_models import FakeScene
from tests.doubles import make_image, make_video, obj, register_scene

DOG = (0.1, 0.3, 0.4, 0.8)


def upload(client, png, description="A red dog is next to a blue ball.", mode="verify_description", settings=None,
           headers=None, filename="scene.png", ctype="image/png"):
    data = {"description": description, "mode": mode}
    if settings is not None:
        data["settings"] = json.dumps(settings)
    return client.post("/api/v1/verifications", files={"file": (filename, png, ctype)}, data=data, headers=headers or {})


def dog_png(scenes):
    png = make_image([("red", DOG)])
    register_scene(scenes, png, FakeScene(objects=[obj("dog", DOG)], vlm_answers={"is the dog red": 0.95}))
    return png


def test_image_upload_to_final_report(client, scenes, run_jobs):
    r = upload(client, dog_png(scenes))
    assert r.status_code == 202, r.text
    body = r.json()
    jid = body["job_id"]
    assert body["status"] == "queued" and body["demo_mode"] is True and r.headers["Location"].endswith(jid)
    assert client.get(f"/api/v1/verifications/{jid}").json()["status"] == "queued"
    assert run_jobs() == ["completed"]

    job = client.get(f"/api/v1/verifications/{jid}").json()
    assert job["status"] == "completed" and job["summary"]["total_claims"] > 0
    assert job["media"]["url"].startswith("/api/v1/files?")
    assert "/" not in job["media"]["url"].split("key=")[1].split("&")[0].replace("media/", "", 1)  # opaque key

    claims = client.get(f"/api/v1/verifications/{jid}/claims").json()
    by_text = {c["claim"]["text"]: c for c in claims["items"]}
    assert by_text["A dog is visible."]["result"]["verdict"] == "SUPPORTED"
    assert by_text["The dog is red."]["result"]["verdict"] == "SUPPORTED"
    assert by_text["A ball is visible."]["result"]["verdict"] in ("CONTRADICTED", "INSUFFICIENT_EVIDENCE")

    ev = client.get(f"/api/v1/verifications/{jid}/evidence", params={"claim_id": "c1"}).json()
    assert ev["total"] >= 2 and all(e["claim_id"] == "c1" for e in ev["items"])
    det = [e for e in ev["items"] if e["source"] == "detector"][0]
    assert det["box"] and det["crop_url"]
    crop = client.get(det["crop_url"])
    assert crop.status_code == 200 and crop.headers["content-type"] == "image/jpeg"

    rep = client.get(f"/api/v1/verifications/{jid}/report")
    assert rep.status_code == 200
    rj = rep.json()
    assert "DEMO MODE" in rj["disclaimer"] and rj["claims"] and rj["model_executions"]
    md = client.get(f"/api/v1/verifications/{jid}/report", params={"format": "markdown"})
    assert md.status_code == 200 and md.text.startswith("# RealityCheck AI")

    assert client.get(f"/api/v1/verifications/{jid}/frames").status_code == 409


def test_video_upload_returns_timestamped_frames(client, run_jobs, tmp_path):
    p = make_video(tmp_path / "v.mp4", [[]] * 3 + [[("red", (0.4, 0.3, 0.6, 0.9))]] * 3, fps=10, repeat=5)
    r = upload(client, p.read_bytes(), "A person appears.", filename="v.mp4", ctype="video/mp4")
    assert r.status_code == 202, r.text
    jid = r.json()["job_id"]
    assert r.json()["media_type"] == "video"
    assert run_jobs() == ["completed"]
    frames = client.get(f"/api/v1/verifications/{jid}/frames").json()
    assert len(frames) >= 2
    assert frames[0]["timestamp_s"] == 0.0 and all(f["url"] for f in frames)
    assert [f["timestamp_s"] for f in frames] == sorted(f["timestamp_s"] for f in frames)
    job = client.get(f"/api/v1/verifications/{jid}").json()
    assert job["media"]["duration_s"] == pytest.approx(3.0, abs=0.05)
    assert "sampling" in job["summary"]


def test_generate_and_verify_mode(client, scenes, run_jobs):
    png = make_image([("red", DOG)])
    register_scene(scenes, png, FakeScene(objects=[obj("dog", DOG)], caption="A dog sits next to a cat."))
    r = upload(client, png, description="", mode="generate_and_verify")
    assert r.status_code == 202, r.text
    run_jobs()
    job = client.get(f"/api/v1/verifications/{r.json()['job_id']}").json()
    assert job["description_generated"] == "A dog sits next to a cat."
    assert job["description_original"] is None


@pytest.mark.parametrize("payload,status,code", [
    (b"", 422, "empty_file"),
    (b"%PDF-1.4 not an image", 415, "unsupported_media_type"),
    (b"\x89PNG\r\n\x1a\n" + b"garbage" * 10, 422, "invalid_image"),
])
def test_invalid_uploads(client, payload, status, code):
    r = upload(client, payload)
    assert r.status_code == status, r.text
    err = r.json()["error"]
    assert err["code"] == code and err["request_id"] and r.headers["X-Request-ID"] == err["request_id"]


def test_description_required_and_bounded(client, scenes):
    png = dog_png(scenes)
    assert upload(client, png, description="").json()["error"]["code"] == "description_required"
    long = "a dog " * 2000
    assert upload(client, png, description=long).json()["error"]["code"] == "description_too_long"


def test_invalid_settings_rejected(client, scenes):
    png = dog_png(scenes)
    r = upload(client, png, settings={"abstention": True, "unknown_field": 1})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_settings"
    r = upload(client, png, settings={"use_detector": False, "use_grounding": False, "use_vlm": False})
    assert r.status_code == 422


def test_invalid_mode_is_validation_error(client, scenes):
    r = upload(client, dog_png(scenes), mode="hack")
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_oversized_image_rejected(client, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "max_image_bytes", 1000)
    r = upload(client, make_image([("red", DOG)]))
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


def test_oversized_video_rejected(client, monkeypatch, tmp_path):
    s = get_settings()
    monkeypatch.setattr(s, "max_video_seconds", 1.0)
    p = make_video(tmp_path / "long.mp4", [[]] * 6, fps=10, repeat=5)
    r = upload(client, p.read_bytes(), filename="long.mp4", ctype="video/mp4")
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_video"


def test_decompression_bomb_rejected(client, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "max_image_pixels", 10_000)
    r = upload(client, make_image([("red", DOG)], size=(200, 200)))
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_image"


def test_unknown_job_and_malformed_id(client):
    assert client.get("/api/v1/verifications/00000000-0000-0000-0000-000000000000").status_code == 404
    r = client.get("/api/v1/verifications/../../etc/passwd")
    assert r.status_code == 404
    assert client.get("/api/v1/verifications/not-a-uuid").json()["error"]["code"] == "job_not_found"


def test_access_control_between_owners(client, scenes, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "api_keys", type(s.api_keys)("key-alice-123456,key-bob-7890123"))
    monkeypatch.setattr(s, "auth_required", True)
    assert upload(client, dog_png(scenes)).status_code == 401
    r = upload(client, dog_png(scenes), headers={"X-API-Key": "key-alice-123456"})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    assert client.get(f"/api/v1/verifications/{jid}", headers={"X-API-Key": "key-alice-123456"}).status_code == 200
    assert client.get(f"/api/v1/verifications/{jid}", headers={"X-API-Key": "key-bob-7890123"}).status_code == 404
    assert client.get(f"/api/v1/verifications/{jid}", headers={"Authorization": "Bearer key-bob-7890123"}).status_code == 404
    assert client.get(f"/api/v1/verifications/{jid}", headers={"X-API-Key": "wrong"}).status_code == 401
    hist = client.get("/api/v1/history", headers={"X-API-Key": "key-bob-7890123"}).json()
    assert all(i["job_id"] != jid for i in hist["items"])


def test_signed_file_urls_cannot_be_tampered(client, scenes, run_jobs):
    jid = upload(client, dog_png(scenes)).json()["job_id"]
    url = client.get(f"/api/v1/verifications/{jid}").json()["media"]["url"]
    assert client.get(url).status_code == 200
    assert client.get(url.replace("sig=", "sig=0")).status_code == 403
    assert client.get(url.replace("owner=local", "owner=other")).status_code == 403
    bad = client.get("/api/v1/files", params={"key": "../../etc/passwd", "owner": "local", "exp": 9999999999, "sig": "x"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "invalid_key"
    expired = client.get("/api/v1/files", params={"key": "media/abc.jpg", "owner": "local", "exp": 1, "sig": "x"})
    assert expired.status_code == 403


def test_responses_do_not_leak_filesystem_paths(client, scenes, run_jobs, tmp_root):
    jid = upload(client, dog_png(scenes)).json()["job_id"]
    run_jobs()
    blob = client.get(f"/api/v1/verifications/{jid}/report").text
    assert str(tmp_root) not in blob and "/Users/" not in blob and "storage/" not in blob


def test_cancel_queued_job_and_delete(client, scenes, run_jobs):
    jid = upload(client, dog_png(scenes)).json()["job_id"]
    r = client.post(f"/api/v1/verifications/{jid}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "cancelled"
    assert run_jobs() == []
    assert client.post(f"/api/v1/verifications/{jid}/cancel").status_code == 409
    assert client.get(f"/api/v1/verifications/{jid}/report").status_code == 409
    assert client.delete(f"/api/v1/verifications/{jid}").status_code == 204
    assert client.get(f"/api/v1/verifications/{jid}").status_code == 404


def test_history_pagination_filter_and_search(client, scenes, run_jobs):
    png = dog_png(scenes)
    ids = [upload(client, png, description=f"A dog number {i} is red.").json()["job_id"] for i in range(5)]
    run_jobs()
    page = client.get("/api/v1/history", params={"limit": 2, "offset": 0}).json()
    assert page["limit"] == 2 and len(page["items"]) == 2 and page["total"] >= 5
    page2 = client.get("/api/v1/history", params={"limit": 2, "offset": 2}).json()
    assert {i["job_id"] for i in page["items"]}.isdisjoint({i["job_id"] for i in page2["items"]})
    found = client.get("/api/v1/history", params={"q": "number 3"}).json()
    assert [i["job_id"] for i in found["items"]] == [ids[3]]
    assert client.get("/api/v1/history", params={"status": "bogus"}).status_code == 422
    done = client.get("/api/v1/history", params={"status": "completed", "media_type": "image"}).json()
    assert all(i["status"] == "completed" for i in done["items"])
    assert done["items"][0]["verdict_counts"] is not None
    stats = client.get("/api/v1/history/stats").json()
    assert stats["total_jobs"] >= 5
    assert client.get("/api/v1/history", params={"limit": 1000}).status_code == 422


def test_health_and_system_info(client):
    assert client.get("/api/v1/health/live").json() == {"status": "ok"}
    ready = client.get("/api/v1/health/ready")
    assert ready.status_code == 200 and ready.json()["checks"]["database"] == "ok"
    info = client.get("/api/v1/system/info").json()
    assert info["demo_mode"] is True and info["privacy"]["processing"] == "local"


def test_rate_limit(client, scenes, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "rate_limit_per_minute", 2)
    png = dog_png(scenes)
    assert upload(client, png).status_code == 202
    assert upload(client, png).status_code == 202
    r = upload(client, png)
    assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limited" and "Retry-After" in r.headers


def test_unknown_route_uses_error_format(client):
    r = client.get("/api/v1/does-not-exist")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_security_headers(client):
    r = client.get("/api/v1/health/live")
    assert r.headers["X-Content-Type-Options"] == "nosniff" and r.headers["X-Frame-Options"] == "DENY"


def test_evaluation_endpoints_empty_and_404(client):
    assert client.get("/api/v1/evaluations").status_code == 200
    assert client.get("/api/v1/evaluations/nope").status_code == 404
