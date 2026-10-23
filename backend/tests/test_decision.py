"""Verdict logic, abstention rules, absence handling, disagreement and confidence handling."""

import pytest

from app.schemas.claims import AtomicClaim, ClaimType, EntityRef
from app.schemas.evidence import (
    BoundingBox,
    EvidenceCase,
    EvidenceItem,
    EvidenceQuality,
    EvidenceSource,
    Stance,
    Strength,
    Verdict,
)
from app.verification.decision import apply_dependencies, decide
from app.verification.policy import PlattCalibrator, VerificationPolicy

DOG = EntityRef(entity_id="e1", head="dog", phrase="dog", canonical_class="dog")
BOX = BoundingBox(x1=0.1, y1=0.1, x2=0.5, y2=0.5)
P = VerificationPolicy()


def claim(ctype=ClaimType.OBJECT_EXISTENCE, **kw):
    base = dict(claim_id="c1", text="A dog is visible.", claim_type=ctype, subject=DOG)
    base.update(kw)
    return AtomicClaim(**base)


def ev(source, stance, strength=Strength.NONE, box=None, **details):
    return EvidenceItem(evidence_id=f"{source.value}-{stance.value}-{strength.value}", claim_id="c1", source=source,
                        stance=stance, strength=strength, box=box, description="test", details=details)


def test_positive_evidence_from_independent_sources_is_supported_with_high_quality():
    d = decide(claim(), [ev(EvidenceSource.DETECTOR, Stance.SUPPORTS, Strength.STRONG, BOX),
                         ev(EvidenceSource.GROUNDING, Stance.SUPPORTS, Strength.STRONG, BOX),
                         ev(EvidenceSource.VLM, Stance.SUPPORTS, Strength.STRONG)], P)
    assert d.verdict == Verdict.SUPPORTED and d.case == EvidenceCase.POSITIVE
    assert d.quality == EvidenceQuality.HIGH and len(d.supporting) == 3
    assert d.calibrated_probability is None  # no calibrator loaded -> no probability claimed


def test_missing_detection_alone_never_contradicts_existence():
    d = decide(claim(), [ev(EvidenceSource.DETECTOR, Stance.ABSENT), ev(EvidenceSource.GROUNDING, Stance.ABSENT)], P)
    assert d.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert d.case == EvidenceCase.MISSING
    assert d.contradicting == []


def test_absence_plus_strong_vlm_rejection_contradicts():
    d = decide(claim(), [ev(EvidenceSource.DETECTOR, Stance.ABSENT), ev(EvidenceSource.GROUNDING, Stance.ABSENT),
                         ev(EvidenceSource.VLM, Stance.CONTRADICTS, Strength.STRONG)], P)
    assert d.verdict == Verdict.CONTRADICTED and d.case == EvidenceCase.CONTRADICTORY
    assert d.quality in (EvidenceQuality.MEDIUM, EvidenceQuality.HIGH)


def test_strong_cross_source_conflict_requires_review():
    d = decide(claim(), [ev(EvidenceSource.DETECTOR, Stance.SUPPORTS, Strength.STRONG, BOX),
                         ev(EvidenceSource.VLM, Stance.CONTRADICTS, Strength.STRONG)], P)
    assert d.verdict == Verdict.REVIEW_REQUIRED and d.case == EvidenceCase.DISAGREEMENT and d.review


def test_disagreement_analysis_can_be_ablated():
    p = VerificationPolicy(use_disagreement=False)
    d = decide(claim(), [ev(EvidenceSource.DETECTOR, Stance.SUPPORTS, Strength.STRONG, BOX),
                         ev(EvidenceSource.VLM, Stance.CONTRADICTS, Strength.STRONG)], p)
    assert d.verdict != Verdict.REVIEW_REQUIRED


def test_weak_evidence_abstains():
    d = decide(claim(ClaimType.ACTION, text="The dog is running.", predicate="run"),
               [ev(EvidenceSource.VLM, Stance.SUPPORTS, Strength.WEAK)], P)
    assert d.verdict == Verdict.INSUFFICIENT_EVIDENCE  # require_strong_for_decision


def test_no_abstention_forces_binary_decision():
    p = VerificationPolicy(use_abstention=False)
    d = decide(claim(ClaimType.ACTION, text="The dog is running.", predicate="run"),
               [ev(EvidenceSource.VLM, Stance.SUPPORTS, Strength.WEAK)], p)
    assert d.verdict == Verdict.SUPPORTED
    d2 = decide(claim(), [], p)
    assert d2.verdict == p.forced_tie_verdict and d2.case == EvidenceCase.MISSING


def test_negated_claim_is_inverted():
    c = claim(text="There is no dog.", negated=True)
    d = decide(c, [ev(EvidenceSource.DETECTOR, Stance.SUPPORTS, Strength.STRONG, BOX),
                   ev(EvidenceSource.VLM, Stance.SUPPORTS, Strength.STRONG)], P)
    assert d.verdict == Verdict.CONTRADICTED
    assert d.contradicting and not d.supporting


def test_nonvisual_claims_are_unverifiable():
    c = AtomicClaim(claim_id="c1", text="The sunset is beautiful.", claim_type=ClaimType.NONVISUAL, verifiable=False,
                    unverifiable_reason="subjective")
    d = decide(c, [ev(EvidenceSource.VLM, Stance.SUPPORTS, Strength.STRONG)], P)
    assert d.verdict == Verdict.UNVERIFIABLE and d.case == EvidenceCase.NONVISUAL


def test_temporal_claim_on_single_image_is_insufficient():
    c = claim(ClaimType.TEMPORAL_EVENT, text="The man enters the room.", predicate="enter")
    d = decide(c, [ev(EvidenceSource.VLM, Stance.SUPPORTS, Strength.STRONG)], P)
    assert d.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert "single image" in d.explanation


def test_poor_image_quality_blocks_absence_based_contradiction():
    items = [ev(EvidenceSource.DETECTOR, Stance.ABSENT), ev(EvidenceSource.GROUNDING, Stance.ABSENT),
             ev(EvidenceSource.VLM, Stance.CONTRADICTS, Strength.STRONG)]
    d = decide(claim(), items, P, image_quality={"blurry": 1.0})
    assert d.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert any("quality" in u for u in d.uncertainty)


def test_grounding_on_region_detected_as_other_class_is_discounted():
    items = [ev(EvidenceSource.DETECTOR, Stance.ABSENT),
             ev(EvidenceSource.GROUNDING, Stance.SUPPORTS, Strength.STRONG, BOX, overlaps_other_class="cat")]
    d = decide(claim(), items, P)
    assert d.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert any("different class" in u for u in d.uncertainty)


def test_single_source_verdict_is_low_quality_and_flagged_for_review():
    d = decide(claim(ClaimType.SCENE, text="The scene contains a beach.", subject=None),
               [ev(EvidenceSource.VLM, Stance.SUPPORTS, Strength.STRONG)], P)
    assert d.verdict == Verdict.SUPPORTED and d.quality == EvidenceQuality.LOW and d.review


def test_calibrator_produces_probability_only_when_loaded():
    cal = PlattCalibrator(a=4.0, b=0.0, fitted_on="unit-test", n_examples=10)
    d = decide(claim(), [ev(EvidenceSource.VLM, Stance.SUPPORTS, Strength.STRONG)], P, calibrator=cal)
    assert d.calibrated_probability is not None and 0.5 < d.calibrated_probability <= 1.0


def test_scores_are_bounded():
    d = decide(claim(), [ev(EvidenceSource.DETECTOR, Stance.SUPPORTS, Strength.STRONG, BOX)] * 5, P)
    assert -1.0 <= d.score <= 1.0


def test_dependency_propagation_contradicts_presupposing_claims():
    exist = claim()
    color = claim(ClaimType.ATTRIBUTE_COLOR, claim_id="c2", text="The dog is brown.", attribute="brown",
                  depends_on=["c1"])
    d1 = decide(exist, [ev(EvidenceSource.DETECTOR, Stance.ABSENT), ev(EvidenceSource.GROUNDING, Stance.ABSENT),
                        ev(EvidenceSource.VLM, Stance.CONTRADICTS, Strength.STRONG)], P)
    d2 = decide(color, [], P)
    assert d2.verdict == Verdict.INSUFFICIENT_EVIDENCE
    apply_dependencies([exist, color], {"c1": d1, "c2": d2}, P)
    assert d2.verdict == Verdict.CONTRADICTED and "presupposes" in d2.explanation


@pytest.mark.parametrize("threshold", [0.2, 0.5, 0.9])
def test_thresholds_are_configurable(threshold):
    p = VerificationPolicy(support_threshold=threshold, require_strong_for_decision=False)
    d = decide(claim(), [ev(EvidenceSource.VLM, Stance.SUPPORTS, Strength.WEAK)], p)
    assert (d.verdict == Verdict.SUPPORTED) == (0.5 >= threshold)
