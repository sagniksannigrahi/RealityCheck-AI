# Troubleshooting

| Symptom | Cause / fix |
|---|---|
| UI shows **DEMO MODE** | ML dependencies are not installed or `RC_MODEL_BACKEND=demo`. Install with `uv sync --extra ml` and set `RC_MODEL_BACKEND=real`; `GET /api/v1/system/info` shows the active backend. |
| `/api/v1/health/ready` returns 503 "ML dependencies missing" | `RC_MODEL_BACKEND=real` without the `ml` extra. Install it or switch to `auto`/`demo`. |
| Job fails with `model_unavailable` | Weights could not be downloaded/loaded (offline, disk full) or the external provider is not enabled. Run `python scripts/download_models.py`; check `RC_ALLOW_EXTERNAL_PROVIDERS` when using `RC_VLM_PROVIDER=anthropic`. |
| Jobs stay `queued` | No worker is running. Local: `RC_WORKER_MODE=inline` (default) runs one inside the API. Docker: check `docker compose logs worker`. |
| Jobs are slow | CPU inference: ≈1.5 s per grounding phrase and ≈0.4 s per VLM question. Disable grounding per job, keep `RC_VLM_IMAGE_SPLITTING=false`, or use a GPU (`RC_DEVICE=cuda`). |
| Worker killed / out of memory | Peak ≈4.2 GB with all models. Raise the worker's `mem_limit`, keep `RC_MAX_CONCURRENT_INFERENCE=1`, or set `RC_MODEL_IDLE_UNLOAD_SECONDS`. In Docker Desktop, give the VM at least 8 GB. |
| `invalid_video` on upload | The OpenCV build in the image cannot decode the codec, or limits (duration/resolution) are exceeded. Re-encode to H.264 MP4. |
| Media links return 403 | Signed URLs expire (`RC_SIGNED_URL_TTL_SECONDS`, default 1 h); reload the job page. Changing `RC_SECRET_KEY` invalidates all links. |
| Media links return 404 | The retention policy deleted the media; results remain available. |
| `alembic` errors on SQLite `ALTER` | Migrations use batch mode for SQLite; delete `data/realitycheck.db` in development if it predates the migrations. |
| Many claims are `INSUFFICIENT_EVIDENCE` | Expected when evidence is weak, objects are outside the detector vocabulary, the image is blurry/dark, or (demo mode) the image is not one of the bundled demo scenes. The claim panel lists the reasons. |
| Frontend cannot reach the API in dev | Start the backend on port 8000; `npm run dev` proxies `/api` there (`RC_BACKEND_URL` to change). |
| Playwright cannot find Chrome | Install Chrome, or run `npx playwright install chromium` and set `E2E_BROWSER_CHANNEL=chromium`. |
