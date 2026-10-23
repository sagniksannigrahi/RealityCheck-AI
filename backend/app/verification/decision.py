"""Evidence-Agreement Verification with Selective Abstention (EAV-SA) — decision rules.

Inputs: one atomic claim and the evidence items collected for it.
Output: a verdict with an explicit evidence case, a signed agreement score, quality, explanation and trace.

Principles enforced here (each is unit-tested):
1. A missing detection never contradicts on its own ("absence" is weak negative evidence at most).
2. Strong support and strong contradiction from different independent sources -> REVIEW_REQUIRED.
3. A decision needs at least one STRONG item in its direction (configurable).
4. Inadequate evidence -> INSUFFICIENT_EVIDENCE, never a guessed verdict (unless abstention is disabled
   for an ablation, in which case the forced verdict is recorded as such).
5. Non-visual / subjective / instruction-like claims -> UNVERIFIABLE.
6. Negated claims are decided on their positive form and then inverted.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from app.schemas.claims import AtomicClaim, ClaimType
from app.schemas.evidence import (
    EvidenceCase,
    EvidenceItem,
    EvidenceQuality,
    EvidenceSource,
    Stance,
    Strength,
    Verdict,
)
from app.verification.policy import ABSENCE_SENSITIVE, PlattCalibrator, VerificationPolicy

_INVERT = {Verdict.SUPPORTED: Verdict.CONTRADICTED, Verdict.CONTRADICTED: Verdict.SUPPORTED}


@dataclass
class Decision:
    verdict: Verdict
    case: EvidenceCase
    score: float | None
    quality: EvidenceQuality
    supporting: list[str] = field(default_factory=list)
    contradicting: list[str] = field(default_factory=list)
    considered: list[str] = field(default_factory=list)
    explanation: str = ""
    uncertainty: list[str] = field(default_factory=list)
    review: bool = False
    trace: list[str] = field(default_factory=list)
    calibrated_probability: float | None = None
    source_signals: dict[str, float] = field(default_factory=dict)


def item_signal(item: EvidenceItem, claim_type: ClaimType, policy: VerificationPolicy) -> float | None:
    """Map one evidence item to a signal in [-1, 1]; None = not usable for this claim."""
    if item.stance == Stance.NOT_APPLICABLE:
        return None
    if item.stance == Stance.SUPPORTS:
        return 1.0 if item.strength == Strength.STRONG else 0.5
    if item.stance == Stance.CONTRADICTS:
        return -1.0 if item.strength == Strength.STRONG else -0.5
    if item.stance == Stance.ABSENT:
        return -policy.absence_weight if claim_type in ABSENCE_SENSITIVE else 0.0
    return 0.0


def _describe(item: EvidenceItem) -> str:
    src = item.source.value.replace("_", " ")
    tier = f", {item.strength.value}" if item.strength != Strength.NONE else ""
    when = f" @ {item.timestamp_s:.2f}s" if item.timestamp_s is not None else ""
    return f"{src}{when}: {item.description}{tier}"


def decide(
    claim: AtomicClaim,
    items: list[EvidenceItem],
    policy: VerificationPolicy,
    *,
    image_quality: dict[str, float] | None = None,
    calibrator: PlattCalibrator | None = None,
) -> Decision:
    trace: list[str] = []
    uncertainty: list[str] = []
    if claim.hedged:
        uncertainty.append("The description hedges this claim (e.g. 'appears', 'maybe').")

    # ---- non-visual claims ------------------------------------------------------------------
    if not claim.verifiable or claim.claim_type == ClaimType.NONVISUAL:
        reason = claim.unverifiable_reason or "Claim is not visually verifiable."
        return Decision(Verdict.UNVERIFIABLE, EvidenceCase.NONVISUAL, None, EvidenceQuality.NONE,
                        explanation=reason, uncertainty=uncertainty + [reason], trace=["non-visual claim"])

    usable = [i for i in items if policy.source_enabled(i.source)]
    considered = [i.evidence_id for i in usable]
    ctype = claim.claim_type

    if ctype == ClaimType.TEMPORAL_EVENT and not any(i.source == EvidenceSource.TEMPORAL for i in usable):
        why = "A single image cannot establish an event that unfolds over time."
        return Decision(Verdict.INSUFFICIENT_EVIDENCE, EvidenceCase.MISSING, 0.0, EvidenceQuality.NONE,
                        considered=considered, explanation=why, uncertainty=uncertainty + [why],
                        review=False, trace=["temporal claim without multi-frame evidence"])

    weights = policy.weights_for(ctype)

    # ---- cross-source conflict analysis on grounding ----------------------------------------
    adjusted: list[tuple[EvidenceItem, float]] = []
    for it in usable:
        sig = item_signal(it, ctype, policy)
        if sig is None:
            continue
        if (policy.use_disagreement and it.source == EvidenceSource.GROUNDING and sig > 0
                and it.details.get("overlaps_other_class")):
            other = it.details["overlaps_other_class"]
            trace.append(f"grounding region is confidently detected as '{other}' -> grounding support discounted")
            uncertainty.append(f"The grounded region is detected by the detector as a different class ('{other}').")
            sig = 0.0
        adjusted.append((it, sig))

    by_source: dict[str, list[tuple[EvidenceItem, float]]] = defaultdict(list)
    for it, sig in adjusted:
        by_source[it.source.value].append((it, sig))

    source_signal: dict[str, float] = {}
    for src, lst in by_source.items():
        source_signal[src] = sum(s for _, s in lst) / len(lst)
    trace.append("source signals: " + ", ".join(f"{k}={v:+.2f}" for k, v in sorted(source_signal.items())))

    supporting = [it.evidence_id for it, s in adjusted if s > 0]
    contradicting = [it.evidence_id for it, s in adjusted if s < 0 and it.stance == Stance.CONTRADICTS]
    absent = [it for it, _ in adjusted if it.stance == Stance.ABSENT]
    strong_support = [it for it, s in adjusted if s >= 1.0]
    strong_contra = [it for it, s in adjusted if s <= -1.0]

    total_w = sum(weights.get(src, 0.5) for src in source_signal)
    score = (sum(weights.get(src, 0.5) * sig for src, sig in source_signal.items()) / total_w) if total_w else 0.0
    score = max(-1.0, min(1.0, score))
    trace.append(f"weighted agreement score = {score:+.3f}")

    has_signal = any(abs(s) > 0 for s in source_signal.values())
    poor_quality = bool(image_quality) and any(
        image_quality.get(k, 0.0) >= 1.0 for k in ("blurry", "too_dark", "low_resolution", "low_contrast")
    )
    if poor_quality:
        flags = [k for k in ("blurry", "too_dark", "too_bright", "low_resolution", "low_contrast")
                 if image_quality.get(k, 0.0) >= 1.0]
        uncertainty.append("Image quality issues may hide evidence: " + ", ".join(flags) + ".")
    if any(it.details.get("ambiguous") for it, _ in adjusted):
        uncertainty.append("Several regions are plausible referents for this claim.")

    # ---- disagreement between independent sources ------------------------------------------
    disagreement = False
    sup_sources = {it.source for it in strong_support}
    con_sources = {it.source for it in strong_contra}
    if sup_sources and con_sources and (sup_sources - con_sources or con_sources - sup_sources):
        disagreement = True
        trace.append(f"strong disagreement: support from {sorted(s.value for s in sup_sources)}, "
                     f"contradiction from {sorted(s.value for s in con_sources)}")
    absence_vs_strong_vlm = (
        ctype in ABSENCE_SENSITIVE and any(it.source == EvidenceSource.VLM for it in strong_support)
        and len({a.source for a in absent}) >= 2 and not any(
            it.source in (EvidenceSource.DETECTOR, EvidenceSource.GROUNDING) for it in strong_support)
    )
    if absence_vs_strong_vlm:
        uncertainty.append("The VLM reports the object, but neither localizer found it.")
        trace.append("partial disagreement: VLM support vs. localizer absence")

    # ---- verdict --------------------------------------------------------------------------
    verdict: Verdict
    case: EvidenceCase
    if not policy.use_abstention:
        if score > 0:
            verdict = Verdict.SUPPORTED
        elif score < 0:
            verdict = Verdict.CONTRADICTED
        else:
            verdict = policy.forced_tie_verdict
        case = EvidenceCase.POSITIVE if verdict == Verdict.SUPPORTED else EvidenceCase.CONTRADICTORY
        if not has_signal:
            case = EvidenceCase.MISSING
            uncertainty.append("Forced decision without evidence (abstention disabled).")
        elif disagreement:
            case = EvidenceCase.DISAGREEMENT
        trace.append("abstention disabled: forced binary decision")
    elif disagreement and policy.use_disagreement:
        verdict, case = Verdict.REVIEW_REQUIRED, EvidenceCase.DISAGREEMENT
        uncertainty.append("Independent components produced incompatible strong evidence.")
    else:
        can_support = score >= policy.support_threshold and (not policy.require_strong_for_decision or strong_support)
        real_contra = [it for it, s in adjusted if s < 0 and it.stance == Stance.CONTRADICTS]
        can_contra = score <= -policy.contradict_threshold and (
            not policy.require_strong_for_decision or strong_contra
        )
        if can_contra and policy.absence_alone_never_contradicts and not real_contra:
            can_contra = False
            trace.append("absence-only evidence cannot contradict")
        if can_contra and poor_quality and policy.poor_quality_blocks_absence_contradiction and \
                ctype in ABSENCE_SENSITIVE:
            can_contra = False
            trace.append("poor image quality blocks absence-based contradiction")
        if can_support:
            verdict, case = Verdict.SUPPORTED, EvidenceCase.POSITIVE
        elif can_contra:
            verdict, case = Verdict.CONTRADICTED, EvidenceCase.CONTRADICTORY
        else:
            verdict = Verdict.INSUFFICIENT_EVIDENCE
            if not has_signal or (not supporting and not contradicting):
                case = EvidenceCase.MISSING
                uncertainty.append("No sufficiently reliable evidence was found for or against this claim.")
            elif supporting and contradicting:
                case = EvidenceCase.AMBIGUOUS
                uncertainty.append("Evidence is mixed and below the decision thresholds.")
            else:
                case = EvidenceCase.AMBIGUOUS if abs(score) > 0.1 else EvidenceCase.MISSING
                uncertainty.append("Evidence points one way but is too weak for an automated decision.")

    # ---- quality ---------------------------------------------------------------------------
    direction = 1 if verdict == Verdict.SUPPORTED else (-1 if verdict == Verdict.CONTRADICTED else 0)
    # absence-only sources (signal -absence_weight) count as corroboration, not as standalone evidence
    agreeing = {src for src, s in source_signal.items() if direction and s * direction >= 0.3}
    localized = any(it.box is not None and sig * direction > 0 for it, sig in adjusted) if direction else False
    strong_dir = (strong_support if direction > 0 else strong_contra) if direction else []
    if direction == 0:
        quality = EvidenceQuality.NONE if not has_signal else EvidenceQuality.LOW
    elif len(agreeing) >= 2 and strong_dir and localized:
        quality = EvidenceQuality.HIGH
    elif len(agreeing) >= 2 or (strong_dir and localized):
        quality = EvidenceQuality.MEDIUM
    else:
        quality = EvidenceQuality.LOW
        if verdict in (Verdict.SUPPORTED, Verdict.CONTRADICTED):
            uncertainty.append("Verdict rests on a single evidence source.")

    review = verdict == Verdict.REVIEW_REQUIRED or (
        verdict in (Verdict.SUPPORTED, Verdict.CONTRADICTED) and quality == EvidenceQuality.LOW
    ) or absence_vs_strong_vlm

    # ---- explanation -----------------------------------------------------------------------
    pos_lines = [_describe(it) for it, s in adjusted if s > 0][:4]
    neg_lines = [_describe(it) for it, s in adjusted if s < 0][:4]
    head = {
        Verdict.SUPPORTED: "Supported by visual evidence",
        Verdict.CONTRADICTED: "Contradicted by visual evidence",
        Verdict.INSUFFICIENT_EVIDENCE: "Insufficient evidence for an automated verdict",
        Verdict.REVIEW_REQUIRED: "Evidence sources disagree; human review recommended",
    }[verdict]
    parts = [head + "."]
    if pos_lines:
        parts.append("For: " + "; ".join(pos_lines) + ".")
    if neg_lines:
        parts.append("Against: " + "; ".join(neg_lines) + ".")
    if absent and not neg_lines:
        parts.append("Not found by: " + ", ".join(sorted({a.source.value for a in absent})) +
                     " (absence of a detection is not proof of absence).")

    dec = Decision(verdict=verdict, case=case, score=round(score, 4), quality=quality, supporting=supporting,
                   contradicting=contradicting, considered=considered, explanation=" ".join(parts),
                   uncertainty=uncertainty, review=review, trace=trace, source_signals=source_signal)

    # ---- negation ----------------------------------------------------------------------------
    if claim.negated:
        dec.trace.append("claim is negated: verdict computed on the positive form and inverted")
        if dec.verdict in _INVERT:
            dec.verdict = _INVERT[dec.verdict]
            dec.case = EvidenceCase.POSITIVE if dec.verdict == Verdict.SUPPORTED else EvidenceCase.CONTRADICTORY
        dec.supporting, dec.contradicting = dec.contradicting, dec.supporting
        dec.score = -dec.score if dec.score is not None else None
        dec.explanation = "Negated claim (evaluated via its positive form). " + dec.explanation

    if calibrator is not None and dec.score is not None:
        dec.calibrated_probability = round(calibrator(dec.score), 4)
    return dec


def apply_dependencies(claims: list[AtomicClaim], decisions: dict[str, Decision],
                       policy: VerificationPolicy) -> None:
    """Propagate contradictions of presupposed claims (e.g. 'the ball is blue' when no ball exists)."""
    if not policy.propagate_contradictions:
        return
    for c in claims:
        d = decisions.get(c.claim_id)
        if d is None or not c.depends_on or d.verdict == Verdict.UNVERIFIABLE:
            continue
        for dep in c.depends_on:
            dd = decisions.get(dep)
            if dd is None or dd.verdict != Verdict.CONTRADICTED:
                continue
            dep_claim = next((x for x in claims if x.claim_id == dep), None)
            if dep_claim is not None and dep_claim.negated:
                continue  # a contradicted *negated* claim means the thing exists; no presupposition failure
            if dd.quality in (EvidenceQuality.HIGH, EvidenceQuality.MEDIUM):
                if d.verdict == Verdict.SUPPORTED and d.quality == EvidenceQuality.HIGH:
                    d.verdict, d.case, d.review = Verdict.REVIEW_REQUIRED, EvidenceCase.DISAGREEMENT, True
                    d.uncertainty.append(f"Supported, but it presupposes claim {dep}, which is contradicted.")
                elif not c.negated:
                    prev = d.verdict.value
                    d.verdict, d.case = Verdict.CONTRADICTED, EvidenceCase.CONTRADICTORY
                    d.quality = dd.quality if d.quality == EvidenceQuality.NONE else d.quality
                    d.explanation = (f"Contradicted because it presupposes claim {dep}, which is contradicted. "
                                     f"(Own evidence alone: {prev}.) ") + d.explanation
                    d.trace.append(f"presupposition {dep} contradicted -> contradicted")
            else:
                d.uncertainty.append(f"Presupposed claim {dep} is contradicted with low-quality evidence.")
            break
