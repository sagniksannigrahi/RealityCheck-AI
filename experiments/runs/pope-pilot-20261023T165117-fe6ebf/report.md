# Evaluation report — pope-pilot

- Run: `pope-pilot-20261023T165117-fe6ebf`  ·  seed 13  ·  bootstrap 200
- Datasets: pope-coco (RUCAIBox/POPE@main); subsets: adversarial, popular, random
- Examples: 180 (cached evidence for 180); split label counts: {'test': {'SUPPORTED': 66, 'CONTRADICTED': 66}, 'val': {'SUPPORTED': 24, 'CONTRADICTED': 24}}
- Models: detector=ultralytics/yolo11n.pt (ultralytics-8.4.184+sha256:0ebbc80d4a76); grounder=IDEA-Research/grounding-dino-tiny (transformers-4.57.6); verifier=HuggingFaceTB/SmolVLM-500M-Instruct (transformers-4.57.6+tiling=off)
- **Simulated (fake models): False**

Metrics on the held-out **test** split (thresholds selected on val). P/R/F1 are on covered (non-abstained) claims; 'eff.' metrics count abstentions as errors. Positive class for hallucination detection = CONTRADICTED.

| System | Coverage % | Acc (covered) | Acc (eff.) | Macro-F1 [95% CI] | Halluc. P | Halluc. R | Halluc. F1 | FPR | FNR | ECE | Latency ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vlm_only | 100.0 | 0.841 | 0.841 | 0.839 [0.764, 0.916] | 0.785 | 0.939 | 0.855 | 0.258 | 0.061 | 0.163 | 474.311 |
| detector_rules | 100.0 | 0.871 | 0.871 | 0.869 [0.829, 0.916] | 0.795 | 1.000 | 0.886 | 0.258 | 0.000 | — | 4.674 |
| vlm_grounding | 100.0 | 0.902 | 0.902 | 0.901 [0.853, 0.954] | 0.835 | 1.000 | 0.910 | 0.197 | 0.000 | 0.075 | 2221.265 |
| full | 100.0 | 0.902 | 0.902 | 0.901 [0.845, 0.947] | 0.835 | 1.000 | 0.910 | 0.197 | 0.000 | 0.087 | 2225.939 |
| eav_sa | 73.5 | 0.948 | 0.697 | 0.948 [0.903, 0.988] | 0.907 | 1.000 | 0.951 | 0.104 | 0.000 | 0.081 | 2225.939 |
| eav_sa_default | 72.0 | 0.947 | 0.682 | 0.947 [0.901, 0.989] | 0.911 | 1.000 | 0.953 | 0.114 | 0.000 | 0.085 | 2225.939 |
| ablate_no_grounding | 77.3 | 0.912 | 0.705 | 0.909 [0.850, 0.959] | 0.859 | 1.000 | 0.924 | 0.191 | 0.000 | 0.082 | 478.985 |
| ablate_no_detector | 77.3 | 0.941 | 0.727 | 0.941 [0.905, 0.980] | 0.898 | 1.000 | 0.946 | 0.122 | 0.000 | 0.091 | 2221.265 |
| ablate_no_disagreement | 74.2 | 0.959 | 0.712 | 0.959 [0.917, 0.991] | 0.922 | 1.000 | 0.959 | 0.078 | 0.000 | 0.090 | 2225.939 |
| ablate_no_vlm | 44.7 | 0.898 | 0.402 | — [—, —] | — | 0.000 | — | 0.000 | 1.000 | 0.047 | 1751.628 |

## Paired comparisons (test split, cluster bootstrap)

| Comparison | Metric | Δ | 95% CI | p |
|---|---|---|---|---|
| full_vs_vlm_only | mf1 | 0.061 | [0.022, 0.107] | 0.000 |
| full_vs_detector_rules | mf1 | 0.032 | [-0.032, 0.087] | 0.270 |
| full_vs_vlm_grounding | mf1 | 0.000 | [-0.032, 0.031] | 1.000 |
| eav_sa_vs_full | acc_cov | 0.047 | [0.011, 0.096] | 0.000 |
| eav_sa_vs_vlm_only | acc_cov | 0.108 | [0.051, 0.162] | 0.000 |
| ablate_no_grounding_vs_eav_sa | mf1 | -0.039 | [-0.076, -0.009] | 0.000 |
| ablate_no_detector_vs_eav_sa | mf1 | -0.008 | [-0.035, 0.024] | 0.730 |
| ablate_no_disagreement_vs_eav_sa | mf1 | 0.011 | [-0.000, 0.032] | 0.130 |
| ablate_no_vlm_vs_eav_sa | mf1 | — | [—, —] | — |

## Per-category (test)

- vlm_only / object_existence: n=132, coverage=100.0%, acc(covered)=0.841, macro-F1=0.839
- detector_rules / object_existence: n=132, coverage=100.0%, acc(covered)=0.871, macro-F1=0.869
- vlm_grounding / object_existence: n=132, coverage=100.0%, acc(covered)=0.902, macro-F1=0.901
- full / object_existence: n=132, coverage=100.0%, acc(covered)=0.902, macro-F1=0.901
- eav_sa / object_existence: n=132, coverage=73.5%, acc(covered)=0.948, macro-F1=0.948
- eav_sa_default / object_existence: n=132, coverage=72.0%, acc(covered)=0.947, macro-F1=0.947
- ablate_no_grounding / object_existence: n=132, coverage=77.3%, acc(covered)=0.912, macro-F1=0.909
- ablate_no_detector / object_existence: n=132, coverage=77.3%, acc(covered)=0.941, macro-F1=0.941
- ablate_no_disagreement / object_existence: n=132, coverage=74.2%, acc(covered)=0.959, macro-F1=0.959
- ablate_no_vlm / object_existence: n=132, coverage=44.7%, acc(covered)=0.898, macro-F1=—

## Confusion matrices (rows = gold, cols = predicted)

**vlm_only**: SUPPORTED: SUPPORTED=49, CONTRADICTED=17, ABSTAIN=0; CONTRADICTED: SUPPORTED=4, CONTRADICTED=62, ABSTAIN=0
**detector_rules**: SUPPORTED: SUPPORTED=49, CONTRADICTED=17, ABSTAIN=0; CONTRADICTED: SUPPORTED=0, CONTRADICTED=66, ABSTAIN=0
**vlm_grounding**: SUPPORTED: SUPPORTED=53, CONTRADICTED=13, ABSTAIN=0; CONTRADICTED: SUPPORTED=0, CONTRADICTED=66, ABSTAIN=0
**full**: SUPPORTED: SUPPORTED=53, CONTRADICTED=13, ABSTAIN=0; CONTRADICTED: SUPPORTED=0, CONTRADICTED=66, ABSTAIN=0
**eav_sa**: SUPPORTED: SUPPORTED=43, CONTRADICTED=5, ABSTAIN=18; CONTRADICTED: SUPPORTED=0, CONTRADICTED=49, ABSTAIN=17
**eav_sa_default**: SUPPORTED: SUPPORTED=39, CONTRADICTED=5, ABSTAIN=22; CONTRADICTED: SUPPORTED=0, CONTRADICTED=51, ABSTAIN=15
**ablate_no_grounding**: SUPPORTED: SUPPORTED=38, CONTRADICTED=9, ABSTAIN=19; CONTRADICTED: SUPPORTED=0, CONTRADICTED=55, ABSTAIN=11
**ablate_no_detector**: SUPPORTED: SUPPORTED=43, CONTRADICTED=6, ABSTAIN=17; CONTRADICTED: SUPPORTED=0, CONTRADICTED=53, ABSTAIN=13
**ablate_no_disagreement**: SUPPORTED: SUPPORTED=47, CONTRADICTED=4, ABSTAIN=15; CONTRADICTED: SUPPORTED=0, CONTRADICTED=47, ABSTAIN=19
**ablate_no_vlm**: SUPPORTED: SUPPORTED=53, CONTRADICTED=0, ABSTAIN=13; CONTRADICTED: SUPPORTED=6, CONTRADICTED=0, ABSTAIN=60

## Selected thresholds (val)

- vlm_only: {'tau': 0.1} (val objective 0.812)
- detector_rules: {'tau': 0.15} (val objective 0.831)
- vlm_grounding: {'vlm_strong': 0.95, 'detector_strong': 0.4, 'grounding_strong': 0.35, 'decision_threshold': 0.25} (val objective 0.833)
- full: {'vlm_strong': 0.75, 'detector_strong': 0.4, 'grounding_strong': 0.35, 'decision_threshold': 0.25} (val objective 0.788)
- eav_sa: {'vlm_strong': 0.95, 'detector_strong': 0.4, 'grounding_strong': 0.35, 'decision_threshold': 0.25} (val objective 0.918)
- eav_sa_default: {} (val objective —)
- ablate_no_grounding: {'vlm_strong': 0.95, 'detector_strong': 0.4, 'grounding_strong': 0.35, 'decision_threshold': 0.25} (val objective 0.894)
- ablate_no_detector: {'vlm_strong': 0.75, 'detector_strong': 0.4, 'grounding_strong': 0.45, 'decision_threshold': 0.35} (val objective 0.913)
- ablate_no_disagreement: {'vlm_strong': 0.95, 'detector_strong': 0.4, 'grounding_strong': 0.35, 'decision_threshold': 0.25} (val objective 0.918)
- ablate_no_vlm: {'vlm_strong': 0.75, 'detector_strong': 0.4, 'grounding_strong': 0.35, 'decision_threshold': 0.25} (val objective -0.500)

Runtime: evidence collection 398.8 s; peak RSS 5033.7 MB.
