"""Temporal verification for video claims.

- Static claims ("a dog is visible", "the car is red") use *any-frame* semantics: per-frame decisions are
  aggregated; a contradiction needs several contradicting frames and no supporting frame, and always
  carries a note that events between sampled frames can be missed.
- Motion verbs get an extra motion analysis from the subject's box trajectory (camera motion is a confound).
- Event claims ("enters", "leaves", "picks up") need a *transition* across frames; one frame never suffices.
"""

from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass

from app.claim_extraction.extractor import gerund
from app.claim_extraction.lexicon import MOTION_VERBS
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
from app.verification.collector import EvidenceCollector, FrameContext
from app.verification.decision import Decision, decide
from app.verification.policy import PlattCalibrator, VerificationPolicy

APPEAR_VERBS = {"enter", "appear", "arrive", "emerge", "come", "return", "join", "approach"}
DISAPPEAR_VERBS = {"leave", "exit", "disappear", "depart", "vanish", "go", "walk away", "run away"}
# verb -> (state question template, expected state before, expected state after)
STATE_CHANGE = {
    "pick up": ("Is the {s} holding the {o}?", False, True),
    "put down": ("Is the {s} holding the {o}?", True, False),
    "drop": ("Is the {s} holding the {o}?", True, False),
    "sit down": ("Is the {s} sitting?", False, True),
    "stand up": ("Is the {s} standing?", False, True),
    "get up": ("Is the {s} standing?", False, True),
    "open": ("Is the {o} open?", False, True),
    "close": ("Is the {o} open?", True, False),
    "fall": ("Is the {s} lying on the ground?", False, True),
}
EXISTENTIAL_TYPES = {ClaimType.OBJECT_EXISTENCE, ClaimType.SCENE, ClaimType.OBJECT_IDENTITY, ClaimType.COUNT}


@dataclass
class VideoClaimResult:
    decision: Decision
    items: list[EvidenceItem]
    frame_verdicts: list[tuple[int, float, str]]  # (frame index, timestamp, verdict)


def _item(claim: AtomicClaim, **kw) -> EvidenceItem:
    return EvidenceItem(evidence_id=f"ev-{uuid.uuid4().hex[:12]}", claim_id=claim.claim_id, **kw)


def frames_for_claim(claim: AtomicClaim, frames: list[FrameContext], collector: EvidenceCollector, k: int) -> list[FrameContext]:
    """Pick up to k frames per claim: frames where the detector sees the subject first, then evenly spaced."""
    if len(frames) <= k:
        return frames
    chosen: list[FrameContext] = []
    s = claim.subject
    if s and s.canonical_class and collector.detector is not None and s.canonical_class in collector.detector.vocabulary:
        scored = []
        for f in frames:
            confs = [d.confidence for d in collector.detections(f) if d.label == s.canonical_class]
            if confs:
                scored.append((max(confs), f))
        scored.sort(key=lambda x: -x[0])
        # spread chosen detector frames over time
        for _, f in scored:
            if len(chosen) >= max(1, k - 1):
                break
            chosen.append(f)
    step = max(1, len(frames) // k)
    for f in frames[::step]:
        if len(chosen) >= k:
            break
        if f not in chosen:
            chosen.append(f)
    return sorted(chosen, key=lambda f: f.frame_index or 0)


def _presence(claim: AtomicClaim, f: FrameContext, collector: EvidenceCollector) -> tuple[bool, bool, object]:
    """(present, strongly_present, best box) for the claim subject in a frame using the localizers."""
    s = claim.subject
    if s is None:
        return False, False, None
    cands = collector.localize(f, s)
    if not cands:
        return False, False, None
    return True, cands[0].strong, cands[0].box


def motion_item(claim: AtomicClaim, frames: list[FrameContext], collector: EvidenceCollector,
                policy: VerificationPolicy) -> EvidenceItem | None:
    track = []
    for f in frames:
        present, strong, box = _presence(claim, f, collector)
        if present and box is not None and f.timestamp_s is not None:
            track.append((f.timestamp_s, box.center, box))
    if len(track) < 3:
        return _item(claim, source=EvidenceSource.MOTION, stance=Stance.NOT_APPLICABLE,
                     model_name="box-trajectory", model_version="1.0",
                     description=f"subject localized in {len(track)} sampled frames; at least 3 are needed for motion analysis")
    speeds = []
    for (t0, c0, _), (t1, c1, _) in zip(track, track[1:], strict=False):
        dt = max(1e-3, t1 - t0)
        speeds.append((((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2) ** 0.5) / dt)
    med = statistics.median(speeds)
    th = policy.thresholds
    if med >= 2 * th.motion_moving:
        st, sg, why = Stance.SUPPORTS, Strength.STRONG, "subject moves substantially between frames"
    elif med >= th.motion_moving:
        st, sg, why = Stance.SUPPORTS, Strength.WEAK, "subject moves between frames"
    elif med <= th.motion_static:
        st, sg, why = Stance.CONTRADICTS, Strength.WEAK, "subject stays in place across frames"
    else:
        st, sg, why = Stance.NEUTRAL, Strength.NONE, "small movement; inconclusive"
    return _item(claim, source=EvidenceSource.MOTION, stance=st, strength=sg, raw_score=round(med, 4),
                 model_name="box-trajectory", model_version="1.0", timestamp_s=track[0][0],
                 description=f"{why} (median speed {med:.3f} image-widths/s over {len(track)} frames)",
                 details={"track": [{"t": t, "center": list(c)} for t, c, _ in track],
                          "caveat": "Camera motion and tracking identity switches are not compensated."})


def _temporal_event(claim: AtomicClaim, frames: list[FrameContext], collector: EvidenceCollector,
                    policy: VerificationPolicy, k_vlm: int) -> list[EvidenceItem]:
    verb = (claim.predicate or "").lower()
    items: list[EvidenceItem] = []
    if len(frames) < 2:
        return [_item(claim, source=EvidenceSource.TEMPORAL, stance=Stance.NOT_APPLICABLE, model_name="temporal-rules",
                      model_version="1.0", description="fewer than two frames; an event cannot be established")]
    timeline = []
    for f in frames:
        present, strong, box = _presence(claim, f, collector)
        timeline.append({"frame": f.frame_index, "t": f.timestamp_s, "present": present, "strong": strong,
                         "box": box.model_dump() if box is not None else None})
    pres = [x["present"] for x in timeline]
    first_t = frames[0].timestamp_s or 0.0
    details = {"timeline": timeline, "verb": verb}

    if verb in APPEAR_VERBS or verb in DISAPPEAR_VERBS:
        appear = verb in APPEAR_VERBS
        seq = pres if appear else pres[::-1]
        if not any(pres):
            return [_item(claim, source=EvidenceSource.TEMPORAL, stance=Stance.ABSENT, model_name="temporal-rules",
                          model_version="1.0", description="subject not localized in any sampled frame", details=details)]
        if all(pres):
            strong = len(pres) >= 3 and first_t <= 0.5 and all(x["strong"] for x in timeline)
            what = "from the first to the last sampled frame"
            return [_item(claim, source=EvidenceSource.TEMPORAL, stance=Stance.CONTRADICTS,
                          strength=Strength.STRONG if strong else Strength.WEAK, model_name="temporal-rules",
                          model_version="1.0",
                          description=f"subject is present {what}; no {'entry/appearance' if appear else 'exit/disappearance'} is observed",
                          details=details)]
        # find a transition absent -> present (for appear) in time order
        first_present = seq.index(True)
        before_absent = first_present > 0 and not any(seq[:first_present])
        after_present = sum(seq[first_present:])
        if before_absent:
            strong_after = sum(1 for x in (timeline if appear else timeline[::-1])[first_present:] if x["strong"])
            strong = strong_after >= 1 and after_present >= 2
            idx = first_present if appear else len(pres) - first_present
            t_change = timeline[min(idx, len(timeline) - 1)]["t"]
            return [_item(claim, source=EvidenceSource.TEMPORAL, stance=Stance.SUPPORTS,
                          strength=Strength.STRONG if strong else Strength.WEAK, model_name="temporal-rules",
                          model_version="1.0", timestamp_s=t_change,
                          description=f"subject {'absent then present' if appear else 'present then absent'} (change near {t_change:.2f}s)",
                          details=details)]
        return [_item(claim, source=EvidenceSource.TEMPORAL, stance=Stance.NEUTRAL, model_name="temporal-rules",
                      model_version="1.0", description="presence fluctuates across frames; no clean transition",
                      details=details)]

    if verb in STATE_CHANGE and collector.vlm is not None and policy.components.vlm:
        template, before, after = STATE_CHANGE[verb]
        s = claim.subject.phrase if claim.subject else "person"
        o = claim.object.phrase if claim.object else "object"
        q = template.format(s=s, o=o)
        present_frames = [f for f, p in zip(frames, pres, strict=True) if p] or frames
        early, late = present_frames[0], present_frames[-1]
        if early is late:
            return [_item(claim, source=EvidenceSource.TEMPORAL, stance=Stance.NOT_APPLICABLE,
                          model_name="temporal-rules", model_version="1.0",
                          description="subject visible in only one frame; state change cannot be assessed")]
        j0, j1 = collector.ask(early, q), collector.ask(late, q)
        th = policy.thresholds

        def state(j):
            if j is None or j.p_yes is None:
                return None if j is None else {"yes": True, "no": False}.get(j.answer)
            if j.p_yes >= th.vlm_weak:
                return True
            if j.p_yes <= 1 - th.vlm_weak:
                return False
            return None

        s0, s1 = state(j0), state(j1)
        details.update({"state_question": q, "early": {"t": early.timestamp_s, "p_yes": j0.p_yes if j0 else None},
                        "late": {"t": late.timestamp_s, "p_yes": j1.p_yes if j1 else None}})
        name = collector.vlm.info.name if collector.vlm else None
        if s0 is None or s1 is None:
            st, sg, why = Stance.NEUTRAL, Strength.NONE, "state could not be determined in one of the frames"
        elif s0 == before and s1 == after:
            strongish = all(j and j.p_yes is not None and (j.p_yes >= th.vlm_strong or j.p_yes <= 1 - th.vlm_strong)
                            for j in (j0, j1))
            st, sg, why = Stance.SUPPORTS, Strength.STRONG if strongish else Strength.WEAK, "expected state change observed"
        elif s0 == s1:
            st, sg, why = Stance.CONTRADICTS, Strength.WEAK, "state does not change between the frames"
        else:
            st, sg, why = Stance.CONTRADICTS, Strength.WEAK, "state changes in the opposite direction"
        return [_item(claim, source=EvidenceSource.TEMPORAL, stance=st, strength=sg, model_name=f"temporal-rules+{name}",
                      model_version="1.0", timestamp_s=late.timestamp_s,
                      description=f"{why}: '{q}' at {early.timestamp_s:.2f}s vs {late.timestamp_s:.2f}s",
                      details=details)]

    if verb.split(" ")[0] in MOTION_VERBS:
        m = motion_item(claim, frames, collector, policy)
        if m is not None:
            m.source = EvidenceSource.TEMPORAL
            return [m]
    items.append(_item(claim, source=EvidenceSource.TEMPORAL, stance=Stance.NOT_APPLICABLE, model_name="temporal-rules",
                       model_version="1.0", description=f"no temporal verification rule for '{verb}'", details=details))
    return items


def verify_video_claim(claim: AtomicClaim, frames: list[FrameContext], collector: EvidenceCollector,
                       policy: VerificationPolicy, *, k_frames: int, max_gap_s: float,
                       calibrator: PlattCalibrator | None = None) -> VideoClaimResult:
    if not claim.verifiable or claim.claim_type == ClaimType.NONVISUAL:
        return VideoClaimResult(decide(claim, [], policy), [], [])

    if claim.claim_type == ClaimType.TEMPORAL_EVENT:
        items = _temporal_event(claim, frames, collector, policy, k_frames)
        dec = decide(claim, items, policy, calibrator=calibrator)
        dec.uncertainty.append(f"Frames were sampled with gaps up to {max_gap_s:.2f}s; brief events can be missed.")
        return VideoClaimResult(dec, items, [])

    chosen = frames_for_claim(claim, frames, collector, k_frames)
    per_frame: list[tuple[FrameContext, Decision, list[EvidenceItem]]] = []
    all_items: list[EvidenceItem] = []
    for f in chosen:
        items = collector.collect(claim, f)
        d = decide(claim, items, policy, image_quality=f.quality)
        per_frame.append((f, d, items))
        all_items.extend(items)

    fv = [(f.frame_index or 0, f.timestamp_s or 0.0, d.verdict.value) for f, d, _ in per_frame]
    sup = [(f, d) for f, d, _ in per_frame if d.verdict == Verdict.SUPPORTED]
    con = [(f, d) for f, d, _ in per_frame if d.verdict == Verdict.CONTRADICTED]
    rev = [(f, d) for f, d, _ in per_frame if d.verdict == Verdict.REVIEW_REQUIRED]
    uncertainty = sorted({u for _, d, _ in per_frame for u in d.uncertainty})
    trace = [f"frame {f.frame_index} @ {f.timestamp_s:.2f}s -> {d.verdict.value} ({d.score:+.2f})"
             for f, d, _ in per_frame if d.score is not None]

    def pick(group):
        best = max(group, key=lambda x: abs(x[1].score or 0))
        return best[1]

    existential = claim.claim_type in EXISTENTIAL_TYPES
    if existential and claim.negated:
        # "there is no X": one frame that clearly shows X contradicts the whole claim
        if con and not rev:
            base, verdict, case = pick(con), Verdict.CONTRADICTED, EvidenceCase.CONTRADICTORY
        elif sup and len(sup) >= min(policy.video_min_contradicting_frames, len(per_frame)) and not rev:
            base, verdict, case = pick(sup), Verdict.SUPPORTED, EvidenceCase.POSITIVE
            uncertainty.append(f"Absence is judged from {len(frames)} sampled frames (max gap {max_gap_s:.2f}s).")
        elif rev:
            base, verdict, case = rev[0][1], Verdict.REVIEW_REQUIRED, EvidenceCase.DISAGREEMENT
        else:
            base = per_frame[0][1] if per_frame else decide(claim, [], policy)
            verdict, case = Verdict.INSUFFICIENT_EVIDENCE, EvidenceCase.MISSING
    elif sup and not con:
        base, verdict, case = pick(sup), Verdict.SUPPORTED, EvidenceCase.POSITIVE
    elif con and not sup:
        if len(con) >= min(policy.video_min_contradicting_frames, len(per_frame)) and not rev:
            base, verdict, case = pick(con), Verdict.CONTRADICTED, EvidenceCase.CONTRADICTORY
        else:
            base, verdict, case = pick(con), Verdict.INSUFFICIENT_EVIDENCE, EvidenceCase.AMBIGUOUS
            uncertainty.append("Contradicted in too few frames for a video-level verdict.")
    elif sup and con:
        frac = len(sup) / (len(sup) + len(con))
        if existential:
            base, verdict, case = pick(sup), Verdict.SUPPORTED, EvidenceCase.POSITIVE
            uncertainty.append("Visible in some sampled frames but not others.")
        elif frac >= policy.video_majority or not policy.use_disagreement:
            base, verdict, case = pick(sup), Verdict.SUPPORTED, EvidenceCase.POSITIVE
            uncertainty.append("Holds in most, but not all, frames where it was checked.")
        elif frac <= 1 - policy.video_majority:
            base, verdict, case = pick(con), Verdict.CONTRADICTED, EvidenceCase.CONTRADICTORY
            uncertainty.append("Contradicted in most frames, but supported in at least one.")
        else:
            base, verdict, case = pick(sup), Verdict.REVIEW_REQUIRED, EvidenceCase.DISAGREEMENT
            uncertainty.append("Frames disagree (temporal inconsistency).")
    elif rev:
        base, verdict, case = rev[0][1], Verdict.REVIEW_REQUIRED, EvidenceCase.DISAGREEMENT
    else:
        base = per_frame[0][1] if per_frame else decide(claim, [], policy)
        verdict, case = Verdict.INSUFFICIENT_EVIDENCE, base.case if per_frame else EvidenceCase.MISSING

    if not policy.use_abstention and verdict in (Verdict.INSUFFICIENT_EVIDENCE, Verdict.REVIEW_REQUIRED):
        mean = statistics.fmean([d.score or 0.0 for _, d, _ in per_frame]) if per_frame else 0.0
        verdict = Verdict.SUPPORTED if mean > 0 else (Verdict.CONTRADICTED if mean < 0 else policy.forced_tie_verdict)

    if verdict == Verdict.CONTRADICTED and existential and not claim.negated:
        uncertainty.append(f"Only {len(frames)} frames were sampled (max gap {max_gap_s:.2f}s); "
                           "the object could appear between sampled frames.")

    # motion analysis for motion verbs
    extra_items: list[EvidenceItem] = []
    if claim.claim_type in (ClaimType.ACTION, ClaimType.INTERACTION) and "motion_verb" in claim.flags \
            and policy.components.motion:
        m = motion_item(claim, frames, collector, policy)
        if m is not None:
            extra_items.append(m)
            if m.stance == Stance.CONTRADICTS and verdict == Verdict.SUPPORTED and policy.use_disagreement:
                verdict, case = Verdict.REVIEW_REQUIRED, EvidenceCase.DISAGREEMENT
                uncertainty.append("The VLM reports the action, but the subject does not move across frames.")
            elif m.stance == Stance.SUPPORTS and verdict == Verdict.CONTRADICTED and policy.use_disagreement:
                verdict, case = Verdict.REVIEW_REQUIRED, EvidenceCase.DISAGREEMENT
                uncertainty.append("The subject moves across frames, but the VLM rejects the action.")
            elif m.stance == Stance.SUPPORTS and verdict == Verdict.INSUFFICIENT_EVIDENCE:
                uncertainty.append(f"The subject moves, but '{gerund(claim.predicate or 'move')}' specifically "
                                   "could not be confirmed.")

    supporting = [i for _, d, _ in per_frame for i in d.supporting] + [
        m.evidence_id for m in extra_items if m.stance == Stance.SUPPORTS]
    contradicting = [i for _, d, _ in per_frame for i in d.contradicting] + [
        m.evidence_id for m in extra_items if m.stance == Stance.CONTRADICTS]
    quality = base.quality if verdict in (Verdict.SUPPORTED, Verdict.CONTRADICTED) else (
        EvidenceQuality.LOW if per_frame and any(d.score for _, d, _ in per_frame) else EvidenceQuality.NONE)
    if verdict == Verdict.SUPPORTED and len(sup) >= 2 and quality == EvidenceQuality.MEDIUM:
        quality = EvidenceQuality.HIGH
    times = ", ".join(f"{f.timestamp_s:.2f}s" for f, _ in (sup if verdict == Verdict.SUPPORTED else con)[:5])
    head = {
        Verdict.SUPPORTED: f"Supported in {len(sup)} of {len(per_frame)} checked frames ({times}).",
        Verdict.CONTRADICTED: f"Contradicted in {len(con)} of {len(per_frame)} checked frames ({times}).",
        Verdict.INSUFFICIENT_EVIDENCE: f"No reliable verdict in {len(per_frame)} checked frames.",
        Verdict.REVIEW_REQUIRED: "Evidence conflicts across frames or sources; human review recommended.",
    }[verdict]
    dec = Decision(
        verdict=verdict, case=case, score=base.score, quality=quality, supporting=supporting,
        contradicting=contradicting, considered=[i.evidence_id for i in all_items + extra_items],
        explanation=head + " Best frame: " + base.explanation, uncertainty=uncertainty,
        review=verdict == Verdict.REVIEW_REQUIRED or quality == EvidenceQuality.LOW,
        trace=trace, calibrated_probability=base.calibrated_probability,
    )
    if calibrator is not None and dec.score is not None:
        dec.calibrated_probability = round(calibrator(dec.score), 4)
    return VideoClaimResult(dec, all_items + extra_items, fv)
