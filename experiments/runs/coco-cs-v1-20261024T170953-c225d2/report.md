# Evaluation report — coco-cs-v1

- Run: `coco-cs-v1-20261024T170953-c225d2`  ·  seed 13  ·  bootstrap 1000
- Datasets: coco-count-spatial (instances_val2014); subsets: count, spatial
- Examples: 454 (cached evidence for 454); split label counts: {'test': {'SUPPORTED': 162, 'CONTRADICTED': 162}, 'val': {'SUPPORTED': 65, 'CONTRADICTED': 65}}
- Models: detector=ultralytics/yolo11n.pt (ultralytics-8.4.184+sha256:0ebbc80d4a76); grounder=IDEA-Research/grounding-dino-tiny (transformers-4.57.6); verifier=HuggingFaceTB/SmolVLM-500M-Instruct (transformers-4.57.6+tiling=off)
- **Simulated (fake models): False**

Metrics on the held-out **test** split (thresholds selected on val). P/R/F1 are on covered (non-abstained) claims; 'eff.' metrics count abstentions as errors. Positive class for hallucination detection = CONTRADICTED.

| System | Coverage % | Acc (covered) | Acc (eff.) | Macro-F1 [95% CI] | Halluc. P | Halluc. R | Halluc. F1 | FPR | FNR | ECE | Latency ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vlm_only | 100.0 | 0.580 | 0.580 | 0.580 [0.542, 0.620] | 0.575 | 0.617 | 0.595 | 0.457 | 0.383 | 0.174 | 433.189 |
| detector_rules | 92.0 | 0.728 | 0.670 | 0.708 [0.655, 0.757] | 0.649 | 0.993 | 0.785 | 0.537 | 0.007 | — | 11.243 |
| vlm_grounding | 100.0 | 0.802 | 0.802 | 0.802 [0.760, 0.844] | 0.814 | 0.784 | 0.799 | 0.179 | 0.216 | 0.061 | 1262.109 |
| full | 100.0 | 0.775 | 0.775 | 0.771 [0.727, 0.814] | 0.717 | 0.907 | 0.801 | 0.358 | 0.093 | 0.067 | 1273.353 |
| eav_sa | 61.1 | 0.808 | 0.494 | 0.806 [0.754, 0.856] | 0.728 | 0.958 | 0.827 | 0.330 | 0.042 | 0.067 | 1273.353 |
| eav_sa_default | 44.1 | 0.888 | 0.392 | 0.888 [0.836, 0.935] | 0.797 | 0.983 | 0.881 | 0.181 | 0.017 | 0.078 | 1273.353 |
| ablate_no_grounding | 61.1 | 0.808 | 0.494 | 0.806 [0.754, 0.856] | 0.728 | 0.958 | 0.827 | 0.330 | 0.042 | 0.096 | 444.433 |
| ablate_no_detector | 56.2 | 0.731 | 0.410 | 0.694 [0.630, 0.754] | 0.695 | 0.942 | 0.800 | 0.551 | 0.058 | 0.084 | 1262.109 |
| ablate_no_disagreement | 62.3 | 0.812 | 0.506 | 0.810 [0.760, 0.859] | 0.728 | 0.958 | 0.827 | 0.318 | 0.042 | 0.067 | 1273.353 |
| ablate_no_vlm | 27.8 | 0.978 | 0.272 | 0.962 [0.873, 1.000] | 0.938 | 0.938 | 0.938 | 0.014 | 0.062 | 0.090 | 840.163 |

## Paired comparisons (test split, cluster bootstrap)

| Comparison | Metric | Δ | 95% CI | p |
|---|---|---|---|---|
| full_vs_vlm_only | macro_f1 | 0.191 | [0.140, 0.243] | 0.000 |
| full_vs_detector_rules | macro_f1 | 0.063 | [0.029, 0.102] | 0.000 |
| full_vs_vlm_grounding | macro_f1 | -0.032 | [-0.080, 0.014] | 0.164 |
| eav_sa_vs_full | accuracy_covered | 0.033 | [-0.001, 0.068] | 0.056 |
| eav_sa_vs_vlm_only | accuracy_covered | 0.228 | [0.178, 0.279] | 0.000 |
| ablate_no_grounding_vs_eav_sa | macro_f1 | 0.000 | [0.000, 0.000] | 1.000 |
| ablate_no_detector_vs_eav_sa | macro_f1 | -0.112 | [-0.156, -0.072] | 0.000 |
| ablate_no_disagreement_vs_eav_sa | macro_f1 | 0.005 | [0.001, 0.011] | 0.024 |
| ablate_no_vlm_vs_eav_sa | macro_f1 | 0.156 | [0.075, 0.226] | 0.002 |

## Per-category (test)

- vlm_only / count: n=286, coverage=100.0%, acc(covered)=0.580, macro-F1=0.579
- vlm_only / spatial_relation: n=38, coverage=100.0%, acc(covered)=0.579, macro-F1=0.568
- detector_rules / count: n=286, coverage=100.0%, acc(covered)=0.717, macro-F1=0.693
- detector_rules / spatial_relation: n=38, coverage=31.6%, acc(covered)=1.000, macro-F1=1.000
- vlm_grounding / count: n=286, coverage=100.0%, acc(covered)=0.790, macro-F1=0.790
- vlm_grounding / spatial_relation: n=38, coverage=100.0%, acc(covered)=0.895, macro-F1=0.894
- full / count: n=286, coverage=100.0%, acc(covered)=0.762, macro-F1=0.756
- full / spatial_relation: n=38, coverage=100.0%, acc(covered)=0.868, macro-F1=0.868
- eav_sa / count: n=286, coverage=59.4%, acc(covered)=0.776, macro-F1=0.774
- eav_sa / spatial_relation: n=38, coverage=73.7%, acc(covered)=1.000, macro-F1=1.000
- eav_sa_default / count: n=286, coverage=41.3%, acc(covered)=0.864, macro-F1=0.863
- eav_sa_default / spatial_relation: n=38, coverage=65.8%, acc(covered)=1.000, macro-F1=1.000
- ablate_no_grounding / count: n=286, coverage=59.4%, acc(covered)=0.776, macro-F1=0.774
- ablate_no_grounding / spatial_relation: n=38, coverage=73.7%, acc(covered)=1.000, macro-F1=1.000
- ablate_no_detector / count: n=286, coverage=53.8%, acc(covered)=0.682, macro-F1=0.627
- ablate_no_detector / spatial_relation: n=38, coverage=73.7%, acc(covered)=1.000, macro-F1=1.000
- ablate_no_disagreement / count: n=286, coverage=60.8%, acc(covered)=0.782, macro-F1=0.780
- ablate_no_disagreement / spatial_relation: n=38, coverage=73.7%, acc(covered)=1.000, macro-F1=1.000
- ablate_no_vlm / count: n=286, coverage=21.0%, acc(covered)=0.967, macro-F1=0.492
- ablate_no_vlm / spatial_relation: n=38, coverage=78.9%, acc(covered)=1.000, macro-F1=1.000

## Confusion matrices (rows = gold, cols = predicted)

**vlm_only**: SUPPORTED: SUPPORTED=88, CONTRADICTED=74, ABSTAIN=0; CONTRADICTED: SUPPORTED=62, CONTRADICTED=100, ABSTAIN=0
**detector_rules**: SUPPORTED: SUPPORTED=69, CONTRADICTED=80, ABSTAIN=13; CONTRADICTED: SUPPORTED=1, CONTRADICTED=148, ABSTAIN=13
**vlm_grounding**: SUPPORTED: SUPPORTED=133, CONTRADICTED=29, ABSTAIN=0; CONTRADICTED: SUPPORTED=35, CONTRADICTED=127, ABSTAIN=0
**full**: SUPPORTED: SUPPORTED=104, CONTRADICTED=58, ABSTAIN=0; CONTRADICTED: SUPPORTED=15, CONTRADICTED=147, ABSTAIN=0
**eav_sa**: SUPPORTED: SUPPORTED=69, CONTRADICTED=34, ABSTAIN=59; CONTRADICTED: SUPPORTED=4, CONTRADICTED=91, ABSTAIN=67
**eav_sa_default**: SUPPORTED: SUPPORTED=68, CONTRADICTED=15, ABSTAIN=79; CONTRADICTED: SUPPORTED=1, CONTRADICTED=59, ABSTAIN=102
**ablate_no_grounding**: SUPPORTED: SUPPORTED=69, CONTRADICTED=34, ABSTAIN=59; CONTRADICTED: SUPPORTED=4, CONTRADICTED=91, ABSTAIN=67
**ablate_no_detector**: SUPPORTED: SUPPORTED=35, CONTRADICTED=43, ABSTAIN=84; CONTRADICTED: SUPPORTED=6, CONTRADICTED=98, ABSTAIN=58
**ablate_no_disagreement**: SUPPORTED: SUPPORTED=73, CONTRADICTED=34, ABSTAIN=55; CONTRADICTED: SUPPORTED=4, CONTRADICTED=91, ABSTAIN=67
**ablate_no_vlm**: SUPPORTED: SUPPORTED=73, CONTRADICTED=1, ABSTAIN=88; CONTRADICTED: SUPPORTED=1, CONTRADICTED=15, ABSTAIN=146

## Selected thresholds (val)

- vlm_only: {'tau': 0.35} (val objective 0.636)
- detector_rules: {'tau': 0.55} (val objective 0.770)
- vlm_grounding: {'vlm_strong': 0.95, 'detector_strong': 0.4, 'grounding_strong': 0.35, 'decision_threshold': 0.25} (val objective 0.799)
- full: {'vlm_strong': 0.75, 'detector_strong': 0.4, 'grounding_strong': 0.35, 'decision_threshold': 0.25} (val objective 0.767)
- eav_sa: {'vlm_strong': 0.75, 'detector_strong': 0.4, 'grounding_strong': 0.35, 'decision_threshold': 0.25} (val objective -0.454)
- eav_sa_default: {} (val objective —)
- ablate_no_grounding: {'vlm_strong': 0.75, 'detector_strong': 0.4, 'grounding_strong': 0.35, 'decision_threshold': 0.25} (val objective -0.454)
- ablate_no_detector: {'vlm_strong': 0.75, 'detector_strong': 0.4, 'grounding_strong': 0.35, 'decision_threshold': 0.25} (val objective -0.477)
- ablate_no_disagreement: {'vlm_strong': 0.75, 'detector_strong': 0.4, 'grounding_strong': 0.35, 'decision_threshold': 0.25} (val objective -0.438)
- ablate_no_vlm: {'vlm_strong': 0.75, 'detector_strong': 0.4, 'grounding_strong': 0.35, 'decision_threshold': 0.25} (val objective -0.669)

Runtime: evidence collection 580.6 s; peak RSS 5546.4 MB.
