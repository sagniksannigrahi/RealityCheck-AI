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
python scripts/run_evaluation.py --dataset pope --images-per-subset 50 --backend real --experiment pope-v1 --register
python scripts/run_evaluation.py --dataset coco-count-spatial --coco-images 200 --backend real --experiment coco-cs-v1 --register
python scripts/run_chair_experiment.py --images 80 --seed 21
python scripts/run_video_sampling_experiment.py --videos 60 --seed 7
```

## Results

All on the held-out **test** split; full tables (hallucination P/R/F1, FPR/FNR, ECE, confusion matrices, selected
thresholds, per-category) are in each run's `report.md`.

### POPE (COCO) — `experiments/runs/pope-v1-20261024T165747-ff2597`

900 questions from 137 images (3 subsets × 50 images; subsets share images); 624 test / 276 val.

| System | Coverage | Acc (covered) | Acc (eff.) | Macro-F1 [95% CI] |
|---|---|---|---|---|
| vlm_only | 100% | 0.849 | 0.849 | 0.849 [0.812, 0.880] |
| detector_rules | 100% | 0.853 | 0.853 | 0.850 [0.822, 0.877] |
| vlm_grounding | 100% | 0.864 | 0.864 | 0.863 [0.829, 0.894] |
| full | 100% | 0.872 | 0.872 | 0.871 [0.843, 0.897] |
| eav_sa | 72.0% | 0.960 | 0.691 | 0.959 [0.941, 0.977] |
| eav_sa_default | 76.3% | 0.943 | 0.720 | 0.942 [0.922, 0.961] |
| ablate_no_grounding | 76.0% | 0.922 | 0.700 | 0.919 [0.889, 0.945] |
| ablate_no_detector | 70.4% | 0.954 | 0.671 | 0.954 [0.934, 0.972] |
| ablate_no_disagreement | 70.5% | 0.966 | 0.681 | 0.966 [0.950, 0.982] |
| ablate_no_vlm | 44.7% | 0.914 | 0.409 | 0.478 [0.469, 0.486] |

Paired comparisons: full vs vlm_only +0.022 macro-F1 [−0.001, 0.048], p=0.062 (not significant; the 180-question
pilot's +0.061 did not replicate); eav_sa vs full +0.088 accuracy on covered [0.064, 0.114]; removing grounding
−0.041 [−0.066, −0.020]; removing the detector −0.006 (n.s.); removing disagreement analysis +0.006 (n.s.).
`ablate_no_vlm` never predicts CONTRADICTED because a missing detection is not treated as proof of absence.
Evidence for 848 of the 900 examples was collected in an earlier session and resumed from the cache, so the
recorded collection time (101.8 s) covers only the last 52; peak RSS 5.0 GB.

### COCO-derived counting / left-right — `experiments/runs/coco-cs-v1-20261024T170953-c225d2`

454 claims from 200 images; 324 test (286 count, 38 spatial) / 130 val.

| System | Coverage | Acc (covered) | Acc (eff.) | Macro-F1 [95% CI] |
|---|---|---|---|---|
| vlm_only | 100% | 0.580 | 0.580 | 0.580 [0.542, 0.620] |
| detector_rules | 92.0% | 0.728 | 0.670 | 0.708 [0.655, 0.757] |
| vlm_grounding | 100% | 0.802 | 0.802 | 0.802 [0.760, 0.844] |
| full | 100% | 0.775 | 0.775 | 0.771 [0.727, 0.814] |
| eav_sa | 61.1% | 0.808 | 0.494 | 0.806 [0.754, 0.856] |
| eav_sa_default | 44.1% | 0.888 | 0.392 | 0.888 [0.836, 0.935] |
| ablate_no_grounding | 61.1% | 0.808 | 0.494 | 0.806 [0.754, 0.856] |
| ablate_no_detector | 56.2% | 0.731 | 0.410 | 0.694 [0.630, 0.754] |
| ablate_no_disagreement | 62.3% | 0.812 | 0.506 | 0.810 [0.760, 0.859] |
| ablate_no_vlm | 27.8% | 0.978 | 0.272 | 0.962 [0.873, 1.000] |

Paired comparisons: full vs vlm_only +0.191 macro-F1 [0.140, 0.243]; full vs detector_rules +0.063 [0.029, 0.102];
full vs vlm_grounding −0.032 (n.s.); eav_sa vs full +0.033 accuracy on covered [−0.001, 0.068], p=0.056.

Notes:
- **Infeasible tuning.** No grid point kept ≥60% coverage on val for the abstaining systems (negative val objective in
  the report), so tuning picked the highest-coverage point. This is why tuned `eav_sa` is less accurate than the
  untuned defaults here.
- **Grounding and counts.** Grounding-based counts are capped at weak evidence, so with all classes inside the
  detector vocabulary, removing grounding leaves every count verdict unchanged.
- The detector is what makes counting work: removing it costs −0.112 macro-F1 [−0.156, −0.072].
- Spatial claims are few (38 test). `ablate_no_vlm` looks best on macro-F1 but decides only 27.8% of claims.
- Evidence collection 580.6 s for 454 claims (≈1.3 s/claim); peak RSS 5.5 GB.

### CHAIR generate-and-verify

`experiments/runs/chair-generate-verify-20261025T191948` — 80 COCO val2014 images (seed 21). SmolVLM writes a caption; object-existence claims that map to a COCO
class are labelled SUPPORTED if the class is in the instance annotations or named in the 5 reference captions (as in
CHAIR), else CONTRADICTED. EAV-SA uses the **untuned default thresholds** (no tuning on this data).

| Measure | Value |
|---|---|
| Claims checked | 141 (131 gold SUPPORTED, 10 gold CONTRADICTED) |
| CHAIR-i / CHAIR-s of raw captions | 0.079 / 0.138 (139 object mentions) |
| CHAIR-i after removing mentions the verifier contradicts | 0.059 |
| EAV-SA coverage / accuracy on covered | 0.901 / 0.984 |
| Hallucination precision / recall (covered) / effective recall | 1.00 / 0.60 / 0.30 |
| Confusion (gold → pred) | SUPPORTED: 122 S, 0 C, 9 abstain · CONTRADICTED: 2 S, 3 C, 5 abstain |

The verifier raised no false alarms and cut CHAIR-i from 0.079 to 0.059, but it caught only 3 of 10 hallucinated
mentions (5 abstained, 2 missed). With only 10 positives this is anecdotal, not a reliable recall estimate.
Generator and verifier share SmolVLM weights, and COCO labels remain incomplete. Runtime 346.7 s (≈4.3 s/image).

### Earlier runs

- **Pilot POPE run** (180 questions): `experiments/runs/pope-pilot-20261023T165117-fe6ebf/report.md`.
- **Video sampling ablation** (synthetic videos, sampling only): with a 4-frame budget, scene-aware sampling
  captured 100% of brief abrupt events vs 10–40% for uniform/interval sampling —
  `experiments/runs/video-sampling-synthetic-20261023T170939/report.md`.
