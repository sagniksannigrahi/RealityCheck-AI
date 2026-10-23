# Models, weights and hardware

| Role | Default model | Size (download) | License | Source |
|---|---|---|---|---|
| Object detector | Ultralytics **YOLO11n** (`yolo11n.pt`, COCO-80 classes) | 5.4 MB | AGPL-3.0 | github.com/ultralytics/assets |
| Open-vocabulary grounding | **IDEA-Research/grounding-dino-tiny** | ≈690 MB | Apache-2.0 | huggingface.co |
| VLM verifier & generator | **HuggingFaceTB/SmolVLM-500M-Instruct** | ≈1.0 GB | Apache-2.0 | huggingface.co |
| Claim parsing | spaCy **en_core_web_sm** 3.8.0 | 12 MB | MIT | github.com/explosion/spacy-models |
| Optional external VLM | Anthropic `claude-opus-5-5` (configurable) | — | API terms | opt-in, not used in tests |

Download everything with `python scripts/download_models.py` (the Docker worker does this on first start into the
`models` volume).

## Measured on the development machine (Apple Silicon, 10 cores, 16 GB RAM, CPU inference)

| Measurement | Value |
|---|---|
| Peak RSS, all three models loaded | ≈4.2 GB |
| YOLO11n, one 640×480 image | ≈70 ms |
| Grounding DINO tiny, one phrase | ≈1.5 s |
| SmolVLM-500M yes/no, image tiling **off** (default) | ≈0.36 s / question |
| SmolVLM-500M yes/no, image tiling on | ≈3.6 s / question |
| 17-claim image, end to end (tiling on) | 81 s |

Tiling (`RC_VLM_IMAGE_SPLITTING`) splits the image into sub-crops (≈880 vs 88 input tokens). The evaluation in
`docs/EVALUATION.md` used tiling **off**.

## Hardware guidance

- **CPU-only, 16 GB RAM:** supported (default). Keep `RC_MAX_CONCURRENT_INFERENCE=1`; Docker worker limit 6 GB.
- **8 GB RAM:** use `RC_MODEL_BACKEND=demo`, or disable grounding per job, or set `RC_MODEL_IDLE_UNLOAD_SECONDS`.
- **GPU:** set `RC_DEVICE=cuda` (or `mps` on Apple Silicon; some Grounding DINO ops may fall back to CPU). The Docker image
  installs CPU-only PyTorch wheels; build a CUDA image separately if needed.

## Swapping models

Adapters implement small protocols (`detection/base.py`, `grounding/base.py`, `vlm/base.py`). Other Ultralytics weights
(e.g. `yolo11s.pt`) work via `RC_DETECTOR_WEIGHTS`; other Grounding DINO checkpoints via `RC_GROUNDING_MODEL_ID`; other
SmolVLM sizes via `RC_VLM_MODEL_ID` / `RC_GENERATOR_MODEL_ID`. Re-run the evaluation after any change; thresholds were
selected for the default models.

## Demo mode

`RC_MODEL_BACKEND=demo` (or `auto` without ML dependencies) uses deterministic fakes. For the three bundled images in
`datasets/demo/` the fakes return COCO **ground-truth** boxes; for any other image they return nothing, so verdicts are
mostly `INSUFFICIENT_EVIDENCE`. Every demo job, report and evaluation run is labelled DEMO / simulated.
