# Database

PostgreSQL in Docker Compose; SQLite for local development and tests. Schema managed by Alembic
(`backend/migrations/versions/0001_initial_schema.py`). All primary keys are UUIDs; all tables have timestamps;
foreign keys cascade from jobs to derived records.

| Table | Purpose | Key columns / indexes |
|---|---|---|
| `media_assets` | Uploaded media metadata and opaque storage key | `owner_id` (ix), `sha256` (ix), `storage_key` (unique), `expires_at` (ix), `deleted_at` |
| `verification_jobs` | Job + queue state | `owner_id`, `status`, `media_type`, `created_at` (ix); (`owner_id`,`created_at`), (`status`,`created_at`); `locked_by/locked_at` (worker lease), `attempts`, `cancel_requested`, `settings`, `summary`, `warnings`, `timings_ms`, `demo_mode`, `error_code/message` |
| `claims` | Atomic claims (full typed payload as JSON) | (`job_id`,`claim_key`) unique; `claim_type` (ix) |
| `claim_results` | Verdict per claim | `claim_id` unique; `verdict` (ix); score, calibrated probability, quality, explanation, uncertainty, evidence ID lists, decision trace, model metadata |
| `evidence_records` | Every observation, incl. absences | (`job_id`,`evidence_key`) unique; `claim_id` (ix), `source` (ix); stance, strength, raw score, box, frame index, timestamp, crop key, details |
| `video_frames` | Sampled frames | (`job_id`,`frame_index`) unique; timestamp, storage key, sampling reason, scene-change score |
| `model_executions` | Per-job model calls and time | component, model name/version, device, calls, total ms, demo/external flags |
| `processing_errors` | Redacted errors per attempt | stage, error type, message, attempt |
| `evaluation_runs` | One row per (experiment run, system) | `run_id` unique; experiment, system, dataset (ix); seed, simulated flag, config, model versions, metrics JSON, output path |
| `evaluation_metrics` | Flat metric rows for querying | (`run_id`,`name`,`category`) unique; value, CI low/high |

## Commands

```bash
.venv/bin/alembic upgrade head          # uses RC_DATABASE_URL (default: SQLite in ./data)
.venv/bin/alembic downgrade base
.venv/bin/alembic check                 # fails if models and migrations drift
.venv/bin/alembic revision --autogenerate -m "describe change"
```

The API also runs `upgrade head` on startup (single-instance convenience). With several API replicas, run migrations
as a separate deployment step.

## Retention

Media expires after `RC_MEDIA_RETENTION_DAYS`. `app.services.jobs.cleanup_expired_media` deletes stored media and
derived crops/frames, sets `media_assets.deleted_at`, and keeps the textual results for audit.
