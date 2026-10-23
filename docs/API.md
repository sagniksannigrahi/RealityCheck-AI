# REST API

Interactive OpenAPI docs: `http://localhost:8000/api/docs` (schema at `/api/openapi.json`).

## Conventions

- **Auth:** when `RC_AUTH_REQUIRED=true`, send `X-API-Key: <key>` (or `Authorization: Bearer <key>`). Each key has its
  own private job history; other owners' jobs return 404.
- **Errors:** every error uses one envelope:
  ```json
  {"error": {"code": "unsupported_media_type", "message": "…", "request_id": "3f2c…", "details": null}}
  ```
  Every response carries `X-Request-ID` (send your own to correlate logs).
- **Pagination:** `?limit=&offset=` → `{"items": [...], "total": N, "limit": L, "offset": O}`.
- **Media URLs** in responses are signed and expire (`RC_SIGNED_URL_TTL_SECONDS`); no filesystem paths are exposed.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/verifications` | Submit image/video (+ description). `202` with `job_id` |
| GET | `/api/v1/verifications/{job_id}` | Status, progress stage, summary, media info, warnings, errors |
| GET | `/api/v1/verifications/{job_id}/claims` | Atomic claims + verdicts (`?verdict=CONTRADICTED`) |
| GET | `/api/v1/verifications/{job_id}/evidence` | Evidence items (`?claim_id=c3`), boxes, crops, timestamps |
| GET | `/api/v1/verifications/{job_id}/frames` | Sampled video frames with timestamps (video jobs only) |
| GET | `/api/v1/verifications/{job_id}/report` | Full report (`?format=json` default, or `markdown`) |
| POST | `/api/v1/verifications/{job_id}/cancel` | Cooperative cancellation |
| DELETE | `/api/v1/verifications/{job_id}` | Delete job, results and stored media |
| GET | `/api/v1/history` | Previous jobs (`status`, `media_type`, `q`, `demo`, pagination) |
| GET | `/api/v1/history/stats` | Jobs by status, claims by verdict |
| GET | `/api/v1/files` | Signed media/crop/frame download |
| GET | `/api/v1/evaluations` | Registered evaluation runs (`experiment`, `dataset`) |
| GET | `/api/v1/evaluations/{run_id}` | One run with all metrics |
| GET | `/api/v1/system/info` | Model backend (real/demo), privacy/processing location, limits |
| GET | `/api/v1/health/live` | Liveness |
| GET | `/api/v1/health/ready` | Readiness (DB, storage, model backend) — 503 when not ready |

### POST /api/v1/verifications (multipart/form-data)

| Field | Required | Notes |
|---|---|---|
| `file` | yes | JPEG/PNG/WebP/GIF ≤ 15 MB, or MP4/MOV/WebM/AVI ≤ 100 MB and ≤ 120 s |
| `mode` | no | `verify_description` (default) or `generate_and_verify` |
| `description` | in `verify_description` | ≤ 4000 characters |
| `settings` | no | JSON `JobSettings`: `abstention`, `disagreement_analysis`, `use_detector`, `use_grounding`, `use_vlm`, `sampling_strategy` (`uniform`/`interval`/`scene_aware`), `max_frames` (2–32) |

Status codes: `202` accepted · `413` too large · `415` unsupported type · `422` validation (invalid image/video,
missing description, bad settings) · `401` auth · `429` rate limited (`Retry-After`).

## Examples

```bash
# submit an image + description
curl -s -X POST localhost:8000/api/v1/verifications \
  -F file=@datasets/demo/images/COCO_val2014_000000039769.jpg \
  -F 'description=Two cats are sleeping on a pink blanket. A dog is next to them.' | jq

# poll, then fetch claims and the report
curl -s localhost:8000/api/v1/verifications/$JOB | jq '.status, .summary.headline'
curl -s "localhost:8000/api/v1/verifications/$JOB/claims?verdict=CONTRADICTED" | jq '.items[].claim.text'
curl -s "localhost:8000/api/v1/verifications/$JOB/report?format=markdown" > report.md

# generate a description with the VLM and verify it independently
curl -s -X POST localhost:8000/api/v1/verifications -F file=@photo.jpg -F mode=generate_and_verify

# video with uniform sampling of 12 frames
curl -s -X POST localhost:8000/api/v1/verifications -F file=@clip.mp4 \
  -F 'description=A person enters the room and picks up a cup.' \
  -F 'settings={"sampling_strategy":"uniform","max_frames":12}'
```

## Claim result fields

`verdict` (`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`, `REVIEW_REQUIRED`, `UNVERIFIABLE`), `evidence_case`
(`positive`, `contradictory`, `missing`, `ambiguous`, `disagreement`, `nonvisual`), `verification_score` (agreement in
[−1, 1], **not** a probability), `calibrated_probability` (only with an evaluated calibrator), `evidence_quality`,
`supporting_evidence` / `contradicting_evidence` / `considered_evidence` (evidence IDs), `explanation`,
`uncertainty_reasons`, `review_recommended`, `decision_trace`, `models`, `duration_ms`.
