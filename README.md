# RealityCheck AI

**Evidence-grounded verification of claims in AI-generated (or human-written) image and video descriptions.**

RealityCheck splits a description into atomic claims ("a dog is visible", "the dog is brown", "the man is holding the
ball"…), collects *independent* visual evidence for each claim — an object detector, an open-vocabulary grounding model,
pixel-level color analysis, box geometry, a separate vision-language check, and frame-by-frame analysis for video — and
returns a per-claim verdict with the evidence behind it:

| Verdict | Meaning |
|---|---|
| `SUPPORTED` | Visual evidence sufficiently supports the claim |
| `CONTRADICTED` | Reliable visual evidence conflicts with the claim |
| `INSUFFICIENT_EVIDENCE` | The system abstains: evidence is missing, weak or ambiguous |
| `REVIEW_REQUIRED` | Independent evidence sources disagree strongly |
| `UNVERIFIABLE` | Subjective, mental-state, speculative or instruction-like text — never forced into a binary verdict |

A missing detection is **never** treated as proof that an object is absent, box overlap never establishes an
interaction, and a single frame never establishes an event.

> **Status.** Research prototype. It runs end to end with real local models on CPU, is covered by automated tests, and
> has been evaluated on a POPE subset and on COCO-derived counting/spatial claims (results below). It is **not**
> production-hardened and makes no claim of novelty or benchmark superiority beyond the measured numbers. See
> [Limitations](#limitations).

![Verification result page (demo mode)](docs/images/job-page.png)

*Screenshot from the automated browser test in DEMO mode (fake models; boxes come from COCO annotations).*

---

## Contents
[Features](#features) · [Quick start](#quick-start) · [Using it](#using-it) · [Configuration](#configuration) ·
[Models](#models--hardware) · [Evaluation results](#evaluation-results) · [Tests](#tests) · [Project layout](#project-layout) ·
[Limitations](#limitations) · [Docs](#documentation)

## Features

- **Two modes:** verify a supplied description, or generate one with a VLM and verify it independently (the verifier only
  sees narrow per-claim questions, never the generated paragraph).
- **Atomic claim extraction** (deterministic, spaCy dependency rules): existence, identity, color/other attributes,
  counts (incl. "several", "a group of"), actions, spatial relations, interactions (holding, wearing, riding…), scene,
  quoted text (OCR-style), temporal events, negation, hedging, subjective/mental-state claims, prompt-injection detection,
  claim dependencies.
- **Independent evidence engine:** YOLO11n (closed vocabulary, never "absent" for out-of-vocabulary classes), Grounding
  DINO tiny (one phrase per pass), HSV pixel color analysis, 2-D box geometry, SmolVLM-500M yes/no on the full image or a
  crop (answers treated as hypotheses), optional Anthropic VLM adapter (opt-in, external).
- **EAV-SA decision engine:** per-source reliability tiers, absence semantics, disagreement → review, configurable
  selective abstention, presupposition propagation, optional Platt calibration fitted on a validation split.
- **Video:** OpenCV validation and probing; uniform, fixed-interval or scene-change-aware sampling with preserved
  timestamps; per-claim frame selection with bounded model calls; any-frame aggregation; motion analysis from box
  trajectories; event rules (enter/leave/appear/disappear, pick up/put down, sit down/stand up, open/close, fall).
- **Evidence UI (React + TypeScript):** upload with settings, live job status, claim list with verdict badges, bounding
  boxes over the original media (solid = supporting, dashed = contradicting), evidence crops, highlighted claim text,
  frame strip with timestamps, uncertainty and disagreement indicators, decision trace, JSON/Markdown report download,
  history with search/filters, and an evaluation dashboard.
- **Backend:** FastAPI, Pydantic v2, SQLAlchemy 2 + Alembic, PostgreSQL (SQLite for dev), DB-backed job queue with retries,
  timeouts, cancellation and stale-lock recovery, signed media URLs, API-key auth with per-owner isolation, rate
  limiting, structured JSON logs with request/job IDs and secret redaction, media retention.
- **Research harness:** POPE and COCO-derived dataset adapters, 5 baselines + 4 ablations over cached evidence,
  val-only threshold selection, cluster-bootstrap CIs, paired tests, calibration (ECE/Brier), runtime/memory.

## Quick start

Requirements: Python 3.12 + [uv](https://docs.astral.sh/uv/), Node 20+ (22 recommended). Docker is optional.

### Option A — local, real models (CPU, ~1.7 GB of weights)

```bash
scripts/setup.sh                       # uv venv + ML extras, frontend build, DB migration, model download
.venv/bin/uvicorn app.main:app --app-dir backend --port 8000
# open http://localhost:8000   (API docs: http://localhost:8000/api/docs)
```

Manual equivalent:

```bash
uv sync --extra ml --extra dev
(cd frontend && npm ci && npm run build)
.venv/bin/python scripts/download_models.py
RC_MODEL_BACKEND=real .venv/bin/uvicorn app.main:app --app-dir backend --port 8000
```

The API runs an in-process worker by default (`RC_WORKER_MODE=inline`), stores data in `./data/` (SQLite + media)
and loads models lazily on the first job.

### Option B — Docker Compose (PostgreSQL + API + worker)

```bash
cp .env.example .env                   # optional; set RC_SECRET_KEY for anything shared
docker compose up --build              # http://localhost:8000
# demo-only slim image (no PyTorch): INSTALL_ML=false RC_MODEL_BACKEND=demo docker compose up --build
```

Memory limits are sized for a 16 GB laptop: Postgres 512 MB, API 1 GB (no models), worker 6 GB / 4 CPUs.
The worker downloads model weights into the `models` volume on first start.

### Option C — demo mode (no ML dependencies)

```bash
uv sync --extra dev && (cd frontend && npm ci && npm run build)
RC_MODEL_BACKEND=demo .venv/bin/uvicorn app.main:app --app-dir backend --port 8000
```

Demo mode uses deterministic fake models and is labelled **DEMO** throughout the UI, API and reports. Only the three
bundled images in `datasets/demo/images/` produce meaningful demo verdicts (their "detections" are COCO ground-truth
boxes); any other image mostly yields `INSUFFICIENT_EVIDENCE`. Demo output is never model evaluation.

### Frontend development server

```bash
cd frontend && npm run dev             # http://localhost:5173, proxies /api to http://127.0.0.1:8000
```

## Using it

1. **Verify** — drop an image or video, type a description (or choose *Generate, then verify*), optionally adjust the
   evidence components, abstention, disagreement analysis and video sampling, then start.
2. **Results** — the summary shows verdict counts and coverage. Selecting a claim highlights its sentence and terms in
   the description, draws its supporting/contradicting regions on the original media (or states that no localized
   evidence exists), and lists every evidence item with crops, scores (model-native, not probabilities), timestamps and
   model versions, plus uncertainty reasons and the decision trace. Download the report as JSON or Markdown.
3. **History** — search and filter previous jobs. **Evaluation** — compare systems from registered experiment runs.

API examples: [docs/API.md](docs/API.md).

## Configuration

All settings are `RC_*` environment variables (or `.env`); see [`.env.example`](.env.example) for the full annotated
list. The most important:

| Variable | Default | Purpose |
|---|---|---|
| `RC_MODEL_BACKEND` | `auto` | `real`, `demo`, or `auto` (real if ML deps import) |
| `RC_DATABASE_URL` | SQLite in `./data` | e.g. `postgresql+psycopg://user:pass@host/db` |
| `RC_SECRET_KEY` | dev value | signs media URLs; required in production |
| `RC_AUTH_REQUIRED` / `RC_API_KEYS` | `false` / empty | API-key auth, per-key job isolation |
| `RC_WORKER_MODE` | `inline` | `external` = run `python -m app.workers.worker` separately |
| `RC_DEVICE` | `cpu` | `cpu`, `mps`, `cuda`, `auto` |
| `RC_VLM_IMAGE_SPLITTING` | `false` | SmolVLM tiling (≈10× slower on CPU) |
| `RC_VLM_PROVIDER` / `RC_ALLOW_EXTERNAL_PROVIDERS` | `smolvlm` / `false` | external Anthropic VLM is opt-in only |
| `RC_POLICY_PATH` | — | JSON `VerificationPolicy` override (thresholds, components) |
| `RC_CALIBRATION_PATH` | — | Platt calibrator from an evaluation run → `calibrated_probability` |
| `RC_MEDIA_RETENTION_DAYS` | `30` | stored media lifetime |

## Models & hardware

| Role | Model | Download | License |
|---|---|---|---|
| Detector | Ultralytics YOLO11n | 5.4 MB | AGPL-3.0 |
| Grounding | IDEA-Research/grounding-dino-tiny | ≈690 MB | Apache-2.0 |
| VLM (verifier + generator) | HuggingFaceTB/SmolVLM-500M-Instruct | ≈1.0 GB | Apache-2.0 |
| Claim parsing | spaCy en_core_web_sm 3.8 | 12 MB | MIT |

Runs on CPU (measured on Apple Silicon, 16 GB RAM): ≈4.2 GB peak RSS with all models; ≈0.4 s per VLM question
(tiling off) and ≈1.5 s per grounding phrase. Details, GPU notes and model swapping: [docs/MODELS.md](docs/MODELS.md).
Note the AGPL-3.0 license of Ultralytics if you offer this as a network service.

## Evaluation results

All runs use real local models on CPU (YOLO11n, Grounding DINO tiny, SmolVLM-500M with tiling off). Images are split
70/30 into test/val by image; thresholds are selected on val only and every number below is on the held-out **test**
split. CIs are 95% cluster-bootstrap over images (1000 resamples). "Coverage" = share of claims the system decided
rather than abstained on.

**POPE (COCO), 3 subsets × 50 images = 900 object-existence questions (137 unique images); 624 on test.**
Full report: [`experiments/runs/pope-v1-20261024T165747-ff2597/report.md`](experiments/runs/pope-v1-20261024T165747-ff2597/report.md).

| System | Coverage | Accuracy (covered) | Macro-F1 [95% CI] |
|---|---|---|---|
| VLM only | 100% | 0.849 | 0.849 [0.812, 0.880] |
| Detector + rules | 100% | 0.853 | 0.850 [0.822, 0.877] |
| VLM + grounding | 100% | 0.864 | 0.863 [0.829, 0.894] |
| Full pipeline (forced decision) | 100% | 0.872 | 0.871 [0.843, 0.897] |
| **EAV-SA (with abstention)** | 72.0% | 0.960 | 0.959 [0.941, 0.977] |
| EAV-SA, untuned default thresholds | 76.3% | 0.943 | 0.942 [0.922, 0.961] |

- The full pipeline's gain over VLM-only is small and **not significant** (+0.022 macro-F1, CI [−0.001, 0.048], p=0.06).
  The pilot run (180 questions) had suggested +0.061; the larger sample does not confirm that.
- Selective abstention is the clear effect: on the 72% of claims EAV-SA decides, accuracy rises by +0.088 over the
  forced-decision pipeline (CI [0.064, 0.114]). Abstentions counted as errors, effective accuracy is 0.691.
- Ablations: removing grounding hurts (−0.041 macro-F1, significant); removing the detector or the disagreement check
  makes no significant difference; without the VLM the system never outputs CONTRADICTED (a missing detection is not
  proof of absence), so it abstains on almost all negatives.

**COCO-derived counting and left/right claims, 200 images = 454 claims (286 count + 38 spatial on test).**
Full report: [`experiments/runs/coco-cs-v1-20261024T170953-c225d2/report.md`](experiments/runs/coco-cs-v1-20261024T170953-c225d2/report.md).

| System | Coverage | Accuracy (covered) | Macro-F1 [95% CI] |
|---|---|---|---|
| VLM only | 100% | 0.580 | 0.580 [0.542, 0.620] |
| Detector + rules | 92.0% | 0.728 | 0.708 [0.655, 0.757] |
| VLM + grounding | 100% | 0.802 | 0.802 [0.760, 0.844] |
| Full pipeline (forced decision) | 100% | 0.775 | 0.771 [0.727, 0.814] |
| EAV-SA (with abstention) | 61.1% | 0.808 | 0.806 [0.754, 0.856] |
| EAV-SA, untuned default thresholds | 44.1% | 0.888 | 0.888 [0.836, 0.935] |

- SmolVLM alone is near chance at counting (0.580). Adding independent localizers gives the largest gain measured
  in this project: full pipeline vs VLM-only +0.191 macro-F1 (CI [0.140, 0.243]).
- Abstention helps less here. No threshold setting reached the required 60% validation coverage, so tuning fell back
  to the highest-coverage setting. The untuned defaults are more accurate but decide only 44% of claims.
- Grounding-based counts are capped at "weak" evidence, so removing grounding leaves EAV-SA's count verdicts unchanged
  on this data.
- Spatial claims are few (38 test claims); every abstaining variant was 100% accurate on those it decided.

**CHAIR generate-and-verify, 80 COCO images, 141 claims from SmolVLM captions** ([report](experiments/runs/chair-generate-verify-20261025T191948/report.md)):
captions hallucinated 7.9% of object mentions (CHAIR-i). Removing mentions the verifier contradicted lowered that to 5.9%
with no false alarms, but only 3 of the 10 hallucinated mentions were caught (5 abstained). The sample has few positives.

**Video sampling ablation (synthetic videos, sampling only, no models):** with a 4-frame budget, scene-aware sampling
captured a frame inside 100% of brief (0.3–1.2 s) abrupt events vs 10–40% for uniform/interval sampling
([report](experiments/runs/video-sampling-synthetic-20261023T170939/report.md)). These are best-case, high-contrast
synthetic events.

Protocol, datasets and licenses: [docs/EVALUATION.md](docs/EVALUATION.md).

## Tests

```bash
.venv/bin/pytest -q                    # backend: unit, integration, adversarial, security, evaluation (fake models)
.venv/bin/pytest -q -m slow            # optional: tests that load real models (none required for CI)
.venv/bin/ruff check backend scripts
cd frontend && npm test                # component/page tests (vitest)
cd frontend && npx playwright test     # browser E2E against the real server (demo models; Chrome)
E2E_MODEL_BACKEND=real npx playwright test   # same flow with real models (slow, ~1-3 min)
```

Latest results on the development machine (macOS, Python 3.12, Node 26):
backend **145 passed** (3 real-model tests deselected by default), ruff clean; frontend **17 passed**, production build OK;
Playwright E2E **2 passed** (real server, demo models, Google Chrome). Against PostgreSQL 16 (Docker): Alembic
upgrade → downgrade → upgrade with no drift, and the API/worker/security suites **50 passed**. The slim Docker image
builds and passes the health/UI smoke test. CI runs all of the above.

## Project layout

```
backend/app/         FastAPI app: api, core, schemas, claim_extraction, detection, grounding, vlm, analysis,
                     verification (collector, decision, policy, pipeline), video, persistence, workers,
                     evaluation, observability, services (storage, jobs, model manager, fakes)
backend/migrations/  Alembic migrations
backend/tests/       pytest suite (fake models; no network, no paid APIs)
frontend/            React + TypeScript (Vite) UI, vitest tests, Playwright E2E
datasets/demo/       bundled demo images (CC BY / CC BY-SA, see ATTRIBUTION.md) and scene manifests
datasets/raw/        downloaded datasets (git-ignored): POPE, COCO val2014 annotations/images
experiments/runs/    evaluation outputs: config.json, metrics.json, predictions_*.jsonl, report.md
scripts/             setup, model download, evaluation runners, dataset/demo helpers
docs/                architecture, API, database, evaluation, research method, security, models, troubleshooting
```

## Limitations

- **Speed on CPU:** ≈2–3 s per claim with grounding; a 15-claim image takes ~30–60 s, videos longer.
- **Claim extraction is rule-based** (spaCy): complex syntax, long-range coreference and idioms can be mis-parsed; there is no LLM-based extractor.
- **Small models:** YOLO11n covers only the 80 COCO classes; Grounding DINO tiny produces false positives for absent objects; SmolVLM-500M is weak at counting, fine-grained attributes and actions. Generator and verifier share SmolVLM weights by default (mitigated, not removed, by the independent sources).
- **Model-free evidence is approximate:** pixel color depends on lighting and region quality; 2-D geometry cannot assess depth relations (behind / in front of), which fall back to the VLM.
- **No OCR engine:** quoted-text claims are checked by the VLM only.
- **Video:** no real video benchmark was run; temporal rules cover a fixed verb set; motion analysis ignores camera motion and uses naive per-frame tracking.
- **Evaluation scope:** POPE subset and COCO-derived synthetic claims; COCO annotations are incomplete (label noise); thresholds were tuned on a small validation split and may not transfer.
- **Not verified here:** the full ML Docker image and the Compose stack with real models (only the slim demo image was built and smoke-tested), the S3 adapter, and the Anthropic adapter against the live API (no credentials).
- **Operations:** in-process rate limiting; DB-polling queue (fine for single-node scale); no malware scanning of uploads.
- **License:** Ultralytics YOLO is AGPL-3.0.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) (diagrams, module map, design decisions)
- [Research method & related work](docs/RESEARCH.md)
- [Evaluation protocol & results](docs/EVALUATION.md)
- [API](docs/API.md) · [Database](docs/DATABASE.md) · [Security & privacy](docs/SECURITY.md)
- [Models & hardware](docs/MODELS.md) · [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Implementation progress log](PROJECT_PROGRESS.md)

## License

Code: MIT. Model weights, datasets and bundled images keep their own licenses (see the tables above and
`datasets/demo/ATTRIBUTION.md`).
