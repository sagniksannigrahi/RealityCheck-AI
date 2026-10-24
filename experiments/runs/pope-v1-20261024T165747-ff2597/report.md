# Evaluation report — pope-v1

- Run: `pope-v1-20261024T165747-ff2597`  ·  seed 13  ·  bootstrap 1000
- Datasets: pope-coco (RUCAIBox/POPE@main); subsets: adversarial, popular, random
- Examples: 900 (cached evidence for 900); split label counts: {'test': {'SUPPORTED': 312, 'CONTRADICTED': 312}, 'val': {'SUPPORTED': 138, 'CONTRADICTED': 138}}
- Models: detector=ultralytics/yolo11n.pt (ultralytics-8.4.184+sha256:0ebbc80d4a76); grounder=IDEA-Research/grounding-dino-tiny (transformers-4.57.6); verifier=HuggingFaceTB/SmolVLM-500M-Instruct (transformers-4.57.6+tiling=off)
- **Simulated (fake models): False**

Metrics on the held-out **test** split (thresholds selected on val). P/R/F1 are on covered (non-abstained) claims; 'eff.' metrics count abstentions as errors. Positive class for hallucination detection = CONTRADICTED.

| System | Coverage % | Acc (covered) | Acc (eff.) | Macro-F1 [95% CI] | Halluc. P | Halluc. R | Halluc. F1 | FPR | FNR | ECE | Latency ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vlm_only | 100.0 | 0.849 | 0.849 | 0.849 [0.812, 0.880] | 0.813 | 0.907 | 0.858 | 0.208 | 0.093 | 0.156 | 476.526 |
| detector_rules | 100.0 | 0.853 | 0.853 | 0.850 [0.822, 0.877] | 0.782 | 0.978 | 0.869 | 0.272 | 0.022 | — | 4.555 |
| vlm_grounding | 100.0 | 0.864 | 0.864 | 0.863 [0.829, 0.894] | 0.808 | 0.955 | 0.875 | 0.228 | 0.045 | 0.039 | 2197.683 |
| full | 100.0 | 0.872 | 0.872 | 0.871 [0.843, 0.897] | 0.812 | 0.968 | 0.883 | 0.224 | 0.032 | 0.051 | 2202.239 |
| eav_sa | 72.0 | 0.960 | 0.691 | 0.959 [0.941, 0.977] | 0.937 | 0.992 | 0.964 | 0.077 | 0.008 | 0.051 | 2202.239 |
| eav_sa_default | 76.3 | 0.943 | 0.720 | 0.942 [0.922, 0.961] | 0.915 | 0.989 | 0.950 | 0.112 | 0.011 | 0.046 | 2202.239 |
| ablate_no_grounding | 76.0 | 0.922 | 0.700 | 0.919 [0.889, 0.945] | 0.877 | 1.000 | 0.935 | 0.176 | 0.000 | 0.032 | 481.081 |
| ablate_no_detector | 70.4 | 0.954 | 0.671 | 0.954 [0.934, 0.972] | 0.934 | 0.988 | 0.960 | 0.086 | 0.012 | 0.047 | 2197.683 |
| ablate_no_disagreement | 70.5 | 0.966 | 0.681 | 0.966 [0.950, 0.982] | 0.946 | 0.991 | 0.968 | 0.062 | 0.009 | 0.040 | 2202.239 |
| ablate_no_vlm | 44.7 | 0.914 | 0.409 | 0.478 [0.469, 0.486] | — | 0.000 | 0.000 | 0.000 | 1.000 | 0.027 | 1725.713 |

## Paired comparisons (test split, cluster bootstrap)

| Comparison | Metric | Δ | 95% CI | p |
|---|---|---|---|---|
| full_vs_vlm_only | macro_f1 | 0.022 | [-0.001, 0.048] | 0.062 |
| full_vs_detector_rules | macro_f1 | 0.020 | [-0.003, 0.041] | 0.070 |
| full_vs_vlm_grounding | macro_f1 | 0.008 | [-0.015, 0.030] | 0.478 |
| eav_sa_vs_full | accuracy_covered | 0.088 | [0.064, 0.114] | 0.000 |
| eav_sa_vs_vlm_only | accuracy_covered | 0.111 | [0.080, 0.145] | 0.000 |
| ablate_no_grounding_vs_eav_sa | macro_f1 | -0.041 | [-0.066, -0.020] | 0.000 |
| ablate_no_detector_vs_eav_sa | macro_f1 | -0.006 | [-0.016, 0.003] | 0.184 |
| ablate_no_disagreement_vs_eav_sa | macro_f1 | 0.006 | [-0.000, 0.015] | 0.084 |
| ablate_no_vlm_vs_eav_sa | macro_f1 | -0.482 | [-0.500, -0.461] | 0.000 |

## Per-category (test)

- vlm_only / object_existence: n=624, coverage=100.0%, acc(covered)=0.849, macro-F1=0.849
- detector_rules / object_existence: n=624, coverage=100.0%, acc(covered)=0.853, macro-F1=0.850
- vlm_grounding / object_existence: n=624, coverage=100.0%, acc(covered)=0.864, macro-F1=0.863
- full / object_existence: n=624, coverage=100.0%, acc(covered)=0.872, macro-F1=0.871
- eav_sa / object_existence: n=624, coverage=72.0%, acc(covered)=0.960, macro-F1=0.959
- eav_sa_default / object_existence: n=624, coverage=76.3%, acc(covered)=0.943, macro-F1=0.942
- ablate_no_grounding / object_existence: n=624, coverage=76.0%, acc(covered)=0.922, macro-F1=0.919
- ablate_no_detector / object_existence: n=624, coverage=70.4%, acc(covered)=0.954, macro-F1=0.954
- ablate_no_disagreement / object_existence: n=624, coverage=70.5%, acc(covered)=0.966, macro-F1=0.966
- ablate_no_vlm / object_existence: n=624, coverage=44.7%, acc(covered)=0.914, macro-F1=0.478

## Confusion matrices (rows = gold, cols = predicted)

**vlm_only**: SUPPORTED: SUPPORTED=247, CONTRADICTED=65, ABSTAIN=0; CONTRADICTED: SUPPORTED=29, CONTRADICTED=283, ABSTAIN=0
**detector_rules**: SUPPORTED: SUPPORTED=227, CONTRADICTED=85, ABSTAIN=0; CONTRADICTED: SUPPORTED=7, CONTRADICTED=305, ABSTAIN=0
**vlm_grounding**: SUPPORTED: SUPPORTED=241, CONTRADICTED=71, ABSTAIN=0; CONTRADICTED: SUPPORTED=14, CONTRADICTED=298, ABSTAIN=0
**full**: SUPPORTED: SUPPORTED=242, CONTRADICTED=70, ABSTAIN=0; CONTRADICTED: SUPPORTED=10, CONTRADICTED=302, ABSTAIN=0
**eav_sa**: SUPPORTED: SUPPORTED=192, CONTRADICTED=16, ABSTAIN=104; CONTRADICTED: SUPPORTED=2, CONTRADICTED=239, ABSTAIN=71
**eav_sa_default**: SUPPORTED: SUPPORTED=190, CONTRADICTED=24, ABSTAIN=98; CONTRADICTED: SUPPORTED=3, CONTRADICTED=259, ABSTAIN=50
**ablate_no_grounding**: SUPPORTED: SUPPORTED=173, CONTRADICTED=37, ABSTAIN=102; CONTRADICTED: SUPPORTED=0, CONTRADICTED=264, ABSTAIN=48
**ablate_no_detector**: SUPPORTED: SUPPORTED=180, CONTRADICTED=17, ABSTAIN=115; CONTRADICTED: SUPPORTED=3, CONTRADICTED=239, ABSTAIN=70
**ablate_no_disagreement**: SUPPORTED: SUPPORTED=198, CONTRADICTED=13, ABSTAIN=101; CONTRADICTED: SUPPORTED=2, CONTRADICTED=227, ABSTAIN=83
**ablate_no_vlm**: SUPPORTED: SUPPORTED=255, CONTRADICTED=0, ABSTAIN=57; CONTRADICTED: SUPPORTED=24, CONTRADICTED=0, ABSTAIN=288

## Selected thresholds (val)

- vlm_only: {'tau': 0.1} (val objective 0.836)
- detector_rules: {'tau': 0.15} (val objective 0.886)
- vlm_grounding: {'vlm_strong': 0.95, 'detector_strong': 0.4, 'grounding_strong': 0.35, 'decision_threshold': 0.25} (val objective 0.869)
- full: {'vlm_strong': 0.95, 'detector_strong': 0.4, 'grounding_strong': 0.35, 'decision_threshold': 0.25} (val objective 0.887)
- eav_sa: {'vlm_strong': 0.95, 'detector_strong': 0.4, 'grounding_strong': 0.35, 'decision_threshold': 0.35} (val objective 0.959)
- eav_sa_default: {} (val objective —)
- ablate_no_grounding: {'vlm_strong': 0.95, 'detector_strong': 0.4, 'grounding_strong': 0.35, 'decision_threshold': 0.35} (val objective 0.933)
- ablate_no_detector: {'vlm_strong': 0.95, 'detector_strong': 0.4, 'grounding_strong': 0.45, 'decision_threshold': 0.5} (val objective 0.953)
- ablate_no_disagreement: {'vlm_strong': 0.95, 'detector_strong': 0.4, 'grounding_strong': 0.35, 'decision_threshold': 0.35} (val objective 0.959)
- ablate_no_vlm: {'vlm_strong': 0.75, 'detector_strong': 0.4, 'grounding_strong': 0.35, 'decision_threshold': 0.25} (val objective -0.518)

Runtime: evidence collection 101.8 s; peak RSS 5048.4 MB.
