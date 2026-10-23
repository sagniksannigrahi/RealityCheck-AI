# Evaluation

All numbers in this document were produced by executing `scripts/run_evaluation.py` /
`scripts/run_video_sampling_experiment.py` in this repository; raw outputs (config, per-example predictions, metrics,
report) are in `experiments/runs/<run_id>/`. Nothing here is estimated or copied from papers.

## Datasets (verified sources and licenses)

| Dataset | Source | License | Used for | Label mapping |
|---|---|---|---|---|
| **POPE** (COCO split) | github.com/RUCAIBox/POPE (`output/coco/coco_pope_{random,popular,adversarial}.json`) | MIT (repo) | object-existence claims | `yes` → SUPPORTED, `no` → CONTRADICTED. Negatives are objects *not annotated* in COCO, sampled randomly / by frequency / by co-occurrence. |
| **COCO val2014** images + instances/captions annotations | cocodataset.org | annotations CC BY 4.0; images under their Flickr licenses | POPE images; COCO-derived counting / left-right claims; CHAIR; demo scenes | see below |
| HallusionBench | github.com/tianyi-lab/HallusionBench | BSD-3-Clause | **not used** | It targets visual reasoning / illusion / chart QA, which does not map onto atomic visual claims; a generic JSONL adapter (`load_jsonl`) is provided for converted datasets. |

**COCO-derived claims** (`load_coco_count_spatial`) are *synthetic claims over real images*: for a class with 1–6
annotated instances, "There are N X" is SUPPORTED and "There are N+1 X" is CONTRADICTED; for two singly-annotated classes
with clear horizontal separation, the true left/right relation is SUPPORTED and the swapped one CONTRADICTED. Images whose
counted class includes crowd or tiny (<0.5% area) instances are flagged `ambiguous` and reported separately. These test
counting and geometry, not natural caption errors.

**Annotation caveat (both datasets):** COCO annotations are incomplete; a POPE "no" object or an unannotated instance may
still be visible. Our reported errors therefore include some label noise.

## Protocol

1. Examples are sampled deterministically (seed 13): POPE draws *N* images per subset (each image has 6 questions);
   COCO-derived draws images from instances_val2014.
2. **Split by image** with a salted hash (30% val / 70% test), so no image appears in both splits.
3. Evidence is collected **once** per example with all components on (YOLO11n, Grounding DINO tiny, SmolVLM-500M with
   tiling off, pixel color, geometry) and cached (`experiments/cache/*.jsonl`).
4. Each system is a decision function over the cached evidence. Thresholds are selected **on val only** from a small
   declared grid (VLM/detector/grounding strong thresholds × decision threshold; τ for single-threshold baselines);
   abstaining systems must keep ≥60% val coverage. Selected thresholds are frozen and applied to test.
5. Metrics on test: coverage / abstention rate; accuracy on covered claims and *effective* accuracy (abstentions counted
   as errors); per-class precision/recall/F1, macro-F1 (= mean of SUPPORTED and CONTRADICTED F1) and micro-F1 (= accuracy
   on covered claims); hallucination-detection precision/recall/F1 with CONTRADICTED as positive; FPR/FNR; confusion
   matrices (incl. ABSTAIN column); per-category and ambiguous-subset metrics; ECE (10 bins) and Brier for P(SUPPORTED)
   (VLM P(yes) for `vlm_only`; Platt-calibrated agreement score fitted on val for EAV systems); mean model latency;
   evidence-collection wall time and peak RSS.
6. Uncertainty: 95% **cluster-bootstrap** CIs over images (1000 resamples); paired cluster-bootstrap differences between
   systems on identical examples (two-sided p from sign flips).

Reproduce:

```bash
python scripts/run_evaluation.py --dataset pope --images-per-subset 50 --backend real --experiment pope-coco-v1 --register
python scripts/run_evaluation.py --dataset coco-count-spatial --coco-images 200 --backend real --experiment coco-count-spatial-v1 --register
python scripts/run_video_sampling_experiment.py --videos 60 --seed 7
```

## Results

Main runs (POPE 3 × 50 images; COCO-derived counting/spatial, 200 images; CHAIR generate-and-verify) are in
progress and will be added here. Completed so far:

- **Pilot POPE run** (180 questions, real models): see the README table and
  `experiments/runs/pope-pilot-20261023T165117-fe6ebf/report.md`.
- **Video sampling ablation** (synthetic): `experiments/runs/video-sampling-synthetic-20261023T170939/report.md`.
