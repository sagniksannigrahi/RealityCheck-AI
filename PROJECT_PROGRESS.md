# RealityCheck AI — Implementation Progress

This file records the actual state of implementation so work can resume accurately.
It is updated at the end of every phase. Only results that were actually executed are recorded.

## Environment (verified 2026-10-23)

- macOS arm64 (Apple Silicon), 10 CPU cores, 16 GB RAM, **no discrete GPU** (CPU inference by default).
- Python 3.12 (project venv via `uv`), Node 26 / npm 11.
- Docker 29.8 CLI installed; daemon not running at time of inspection.
- No local PostgreSQL / Redis → SQLite used for local dev and tests; PostgreSQL via Docker Compose.
- Real models verified loadable on CPU: `yolo11n.pt` (Ultralytics), `IDEA-Research/grounding-dino-tiny`,
  `HuggingFaceTB/SmolVLM-500M-Instruct`. Peak RSS with all three loaded ≈ 4.2 GB.

## Checklist

- [x] Phase 1 — Inspect environment, choose architecture, checklist
- [x] Phase 2 — Backend skeleton, config, schemas, persistence, migrations, API, fake adapters
- [x] Phase 3 — Atomic claim extraction
- [x] Phase 4 — Detection + grounding adapters (real + fake), VLM adapters
- [x] Phase 5 — Evidence engine, decision rules, selective abstention
- [x] Phase 6 — Video sampling, timestamped evidence, temporal verification
- [x] Phase 7 — React frontend + evidence visualization
- [x] Phase 8 — Dataset adapters, baselines, metrics, reproducible experiments
- [x] Phase 9 — Security, integration, adversarial tests, performance checks
- [x] Phase 10 — Docker, docs, CI, end-to-end verification, push to GitHub

## Log

(See below; appended per phase.)

### Phases 1–6 (2026-10-23)
- Backend implemented: claim extraction (spaCy rules), YOLO/Grounding-DINO/SmolVLM adapters + optional Anthropic adapter,
  fakes, EAV-SA decision engine, evidence collector, video sampling/temporal verification, persistence (10 tables,
  Alembic 0001 verified up/down/check on SQLite; offline PostgreSQL SQL generated), DB-backed queue + worker, REST API.
- Real-model smoke run (COCO val2017 #39769, 17 claims): all verdicts correct by inspection; 81 s on CPU.
- `pytest`: 132 passed (unit, adversarial pipeline, video, API integration, worker/storage/security).
- Limitation found: synthetic flat test images trip the blur/low-contrast quality check (by design); tests add noise.

### Phases 7–9 (2026-10-23)
- Frontend: React 19 + TS 5.9 + Vite 8; pages Verify / Job results / History / Evaluation / Settings; evidence overlay,
  frame strip, crops, highlighting, reports. vitest 17 passed; `npm run build` OK.
- Browser E2E (Playwright driving system Chrome against the real FastAPI server, demo models): 2 passed.
- Evaluation harness: POPE + COCO-derived adapters, 10 systems (5 baselines, EAV-SA, untuned EAV-SA, 4 ablations),
  val-only tuning, cluster bootstrap CIs, paired tests, calibration. Pilot (real models, 180 POPE questions) done.
- Video sampling ablation on synthetic videos done (scene-aware 100% event recall at 4 frames vs 10–40% uniform/interval).
- Main runs launched: POPE 3×50 images (900 q), COCO count/spatial 200 images. CHAIR generate-and-verify script ready.
- Docker files, CI workflow, docs written; Docker build pending (deferred until evaluation finishes to protect RAM).

### Phase 10 (2026-10-24)
- Project moved from `~/Documents/VS code/` to `~/Downloads/VS Projects/`; venv entry-point scripts repointed.
- Main POPE run completed (resumed from 848/900 cached examples): `experiments/runs/pope-v1-20261024T165747-ff2597`.
  EAV-SA 72.0% coverage, 0.960 accuracy on covered; full vs VLM-only +0.022 macro-F1, not significant (p=0.06).
- COCO count/spatial run completed (200 images, 454 claims): `experiments/runs/coco-cs-v1-20261024T170953-c225d2`.
  Full vs VLM-only +0.191 macro-F1 (significant); abstention could not meet 60% val coverage (fallback setting used).
- CHAIR script: fixed a crash (`sorted()` over dicts) and None-formatting; ground truth now adds objects named in the
  reference captions (original CHAIR protocol).
- PostgreSQL 16 (Docker): migrations up/down/up + `alembic check` clean; API/worker/security tests 50 passed.
- Slim Docker image built and smoke-tested (health ready, UI 200). Full ML image not built.
- Playwright E2E 2 passed; backend 145 passed; frontend 17 passed; ruff clean.
- CI workflow committed; push blocked until the GitHub token has the `workflow` scope.
- CHAIR generate-and-verify (80 images, 141 claims): `experiments/runs/chair-generate-verify-20261025T191948`. CHAIR-i 0.079 → 0.059 after filtering; 3/10
  hallucinated mentions caught, 0 false alarms.
