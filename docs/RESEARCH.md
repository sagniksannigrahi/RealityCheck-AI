# Evidence-Agreement Verification with Selective Abstention (EAV-SA)

This document describes the candidate method implemented in `backend/app/verification/`. It is a **research
hypothesis and an engineering combination of known ideas**, not a claimed novel contribution (see related work).

## Hypothesis

> Combining independently collected, localized visual evidence; claim-specific verification; cross-model
> disagreement analysis; and selective abstention can make hallucination detection more reliable than relying on a
> single vision-language model.

"More reliable" is operationalized as: higher accuracy / macro-F1 on the claims the system decides, at a reported
coverage, with abstentions counted explicitly (see `docs/EVALUATION.md`).

## Method

For each atomic claim *c* with type *t*:

1. **Evidence requirements.** The claim type determines which sources apply (e.g. color → localize subject + pixel
   color + VLM on the crop; spatial relation → localize both entities + box geometry + VLM; count → detector and
   grounding instance counts + VLM).
2. **Source-specific tiers.** Each source maps its native score to `strong` / `weak` / `none` with its own thresholds
   (`VerificationPolicy.thresholds`). Stances: `supports`, `contradicts`, `neutral`, `absent` (searched, not found),
   `not_applicable` (e.g. class outside the detector vocabulary).
3. **Signals.** strong ±1, weak ±0.5, neutral 0; `absent` contributes −0.35 only for existence/count/identity claims.
   Signals are averaged per source, then combined with per-claim-type source weights into an agreement score
   *S* ∈ [−1, 1]. *S* is **not** a probability.
4. **Conflict analysis.** Strong support and strong contradiction from different sources → `REVIEW_REQUIRED`.
   Grounding regions that the detector confidently labels as another class are discounted.
5. **Selective abstention.** `SUPPORTED` needs *S* ≥ τ⁺ and at least one strong supporting item; `CONTRADICTED` needs
   *S* ≤ −τ⁻, at least one strong contradicting item, and at least one *explicit* contradiction (absence alone is never
   enough). Poor image quality (blur, darkness, low contrast or resolution) blocks absence-based contradictions.
   Everything else is `INSUFFICIENT_EVIDENCE`.
6. **Presuppositions.** A claim that depends on a contradicted existence claim ("the ball is blue" when no ball is
   found) is contradicted, unless its own evidence strongly supports it, in which case it goes to review.
7. **Negation** is decided on the positive form and inverted.
8. **Video.** Static claims use any-frame semantics over per-claim selected frames; contradictions need several
   contradicting frames and carry a sampling-gap note. Motion verbs add a box-trajectory check. Event verbs
   (enter/leave/appear/disappear, pick up/put down, sit down/stand up, open/close, fall) require an observed
   transition; a single frame never establishes an event.
9. **Calibration (optional).** A Platt calibrator fitted on a validation split maps *S* to P(supported). The API
   reports `calibrated_probability` only when such a calibrator is configured (`RC_CALIBRATION_PATH`).

## Baselines and ablations (implemented in `backend/app/evaluation/harness.py`)

| System | Description |
|---|---|
| `vlm_only` | VLM P(yes) thresholded (τ tuned on val) |
| `detector_rules` | detector presence/count/geometry with deterministic rules; abstains outside the vocabulary |
| `vlm_grounding` | VLM + grounding (+ geometry/color), forced decision |
| `full` | all components, forced decision |
| `eav_sa` | all components + abstention (thresholds tuned on val) |
| `eav_sa_default` | same, with the pre-declared default thresholds (no tuning) |
| `ablate_no_grounding`, `ablate_no_detector`, `ablate_no_disagreement`, `ablate_no_vlm` | one component removed |

All systems are decision functions over **the same cached observations**, so differences come from the decision
logic and allowed components, not from re-running models. A video-sampling ablation (uniform vs interval vs
scene-aware) is available in the pipeline settings and covered by synthetic tests, but no video benchmark was run (see
limitations).

## Related work (reviewed before writing this section)

The closest prior systems also decompose model output into claims and verify them with visual tools:

- **Woodpecker** (Yin et al., 2023, arXiv:2310.16045): training-free hallucination *correction* with key-concept
  extraction, question formulation, visual knowledge validation using an open-set detector (Grounding DINO) and VQA,
  then rewriting. Reports large POPE gains for MiniGPT-4/mPLUG-Owl.
- **UNIHD / MHaluBench** (Chen et al., ACL 2024): unified, tool-augmented claim-level hallucination *detection*
  (object detection, OCR, search, etc.) and a meta-evaluation benchmark.
- **FaithScore** (Jing et al., Findings of EMNLP 2024): atomic-fact decomposition verified with a visual entailment model.
- Benchmarks: **POPE** (Li et al., EMNLP 2023), **CHAIR** (Rohrbach et al., EMNLP 2018), **HallusionBench**
  (Guan et al., CVPR 2024).
- Selective prediction: selective classification (Geifman & El-Yaniv, NeurIPS 2017); reliable VQA with abstention
  (Whitehead et al., ECCV 2022).

**What differs here (engineering, not a novelty claim):** explicit absence-vs-contradiction semantics, per-source tiers
instead of mixed confidences, a review state for cross-source conflicts, pixel-level color and box-geometry evidence as
model-free sources, temporal rules for video, and an evaluation protocol that reports coverage and abstention alongside
accuracy. Whether these choices improve reliability is an empirical question answered (partially) in
`docs/EVALUATION.md`.
