# Security and privacy

## Threat model (summary)

| Threat | Mitigation | Tested in |
|---|---|---|
| Malicious / malformed uploads | Type decided from magic bytes (client `Content-Type` ignored); size limits enforced while streaming; PIL `verify()` + full decode with pixel and side limits (decompression-bomb guard); OpenCV probing with duration/resolution limits; HEIC/AVIF rejected | `test_api.py` (invalid uploads, oversize image/video, decompression bomb), `test_video.py` (malformed video) |
| Path traversal / filesystem disclosure | Server-generated opaque keys only; strict key regex; resolution confined to storage root; API never returns paths | `test_worker_storage_security.py`, `test_api.py::test_responses_do_not_leak_filesystem_paths` |
| Unauthorized media access | Media/crops/frames served only via HMAC-signed, expiring URLs bound to the owner; `nosniff`, `Content-Disposition: inline`, sandbox CSP | `test_api.py::test_signed_file_urls_cannot_be_tampered` |
| Cross-tenant access | API keys map to owner IDs; every job query filters by owner; foreign jobs return 404 (no existence leak) | `test_api.py::test_access_control_between_owners` |
| Prompt injection in descriptions / OCR | Description is parsed as data; instruction-like sentences become `UNVERIFIABLE` claims and are never sent to the VLM; VLM questions are built from structured fields; the external adapter wraps statements in tags and states they are untrusted | `test_claim_extraction.py`, `test_pipeline_adversarial.py::test_prompt_injection_in_description_is_not_executed` |
| Instructions embedded in images | The VLM only answers yes/no to narrow questions; its free text is never executed or used as configuration; VLM output is labelled a hypothesis | design |
| Secret leakage | `SecretStr` settings; regex + exact-match redaction in logs and stored errors; internal error details not shown to users | `test_worker_storage_security.py` (redaction) |
| Resource exhaustion | Request body limit, per-owner rate limit, max claims per job, max sampled frames, bounded VLM calls per claim, job timeout, inference semaphore, container memory/CPU limits | `test_api.py::test_rate_limit`, `test_video.py::test_vlm_calls_are_bounded_per_claim`, `test_worker_storage_security.py::test_timeout_is_enforced` |
| Unintended external upload of media | External VLM requires both `RC_VLM_PROVIDER=anthropic` and `RC_ALLOW_EXTERNAL_PROVIDERS=true`; `/api/v1/system/info` and the UI state where processing happens | `model_manager.py` |

## Processing location

| Component | Default | Where it runs |
|---|---|---|
| Claim extraction (spaCy) | always | local |
| Detector (YOLO11n), grounding (Grounding DINO), VLM (SmolVLM) | real backend | local CPU/GPU |
| Anthropic VLM adapter | **off** | external API — images/crops leave the server; opt-in only |
| Model weight download | first run | Hugging Face Hub / GitHub (weights only, no user data) |

## Operational notes

- Set `RC_SECRET_KEY` to a long random value; `RC_ENVIRONMENT=production` refuses the default key.
- Enable `RC_AUTH_REQUIRED=true` and `RC_API_KEYS` for any shared deployment; put the service behind TLS.
- The in-process rate limiter is per instance; use a gateway for multi-instance deployments.
- Media retention: `RC_MEDIA_RETENTION_DAYS` (default 30). `cleanup_expired_media` deletes stored media, crops and
  frames and keeps the (non-media) result records. Users can delete a job and its media via `DELETE /api/v1/verifications/{id}`.
- Licensing: Ultralytics YOLO is AGPL-3.0. Offering this service over a network with the Ultralytics detector may impose
  AGPL obligations; the detector is behind an interface and can be replaced.
- Model supply chain: Grounding DINO and SmolVLM load from pinned Hugging Face commits (`DEFAULT_MODEL_REVISIONS` in
  `backend/app/core/config.py`; override with `RC_MODEL_REVISIONS`, disable with `RC_PIN_MODEL_REVISIONS=false`), and
  model ids cannot be set through the API. `transformers` 4.57.x has published advisories (PYSEC-2025-217,
  PYSEC-2026-2288/2289/2290/3929/4174). They concern loading malicious model repos, the `Trainer` class,
  checkpoint conversion or `save_pretrained`, none of which this service uses with untrusted input. The fixes ship
  only in transformers 5.x. Upgrade, and re-run the evaluation, before loading models you do not control.
