"""Independent evidence collection for one image (or one video frame).

For each claim the collector decides which sources apply (detector, grounding, pixel color analysis,
box geometry, VLM on the full image or on a crop) and records *every* observation as an ``EvidenceItem``,
including absences and not-applicable sources, so the report can show exactly what was checked.

Model calls are cached per frame: the detector runs once per frame, grounding once per phrase, and the
VLM once per (question, region).
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

from PIL import Image

from app.analysis.color import analyze_region
from app.analysis.geometry import evaluate_interaction, evaluate_relation
from app.claim_extraction.extractor import gerund, pluralize
from app.claim_extraction.lexicon import COLOR_WORDS, canonical_class
from app.core.imaging import crop as crop_image
from app.detection.base import Detection
from app.grounding.base import GroundingCandidate
from app.schemas.claims import AtomicClaim, ClaimType, CountComparator, EntityRef
from app.schemas.evidence import BoundingBox, EvidenceItem, EvidenceSource, Stance, Strength
from app.verification.policy import VerificationPolicy
from app.vlm.base import VLMJudgment

CropSink = Callable[[Image.Image, BoundingBox, int | None], str | None]

_REL_TEXT = {"next_to": "next to", "in_front_of": "in front of", "left_of": "to the left of",
             "right_of": "to the right of", "inside": "inside"}


@dataclass
class Candidate:
    box: BoundingBox
    score: float
    source: EvidenceSource
    label: str
    strong: bool


@dataclass
class ComponentStats:
    calls: int = 0
    total_ms: float = 0.0


@dataclass
class FrameContext:
    image: Image.Image
    frame_index: int | None = None
    timestamp_s: float | None = None
    quality: dict[str, float] = field(default_factory=dict)
    detections: list[Detection] | None = None
    grounding: dict[str, list[GroundingCandidate]] = field(default_factory=dict)
    vlm: dict[tuple[str, tuple | None], VLMJudgment] = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size


def question_for(claim: AtomicClaim) -> str:
    """A narrow yes/no question built from structured claim fields (not from raw description text)."""
    s = claim.subject
    sp = s.phrase if s else "object"
    t = claim.claim_type
    if t == ClaimType.OBJECT_EXISTENCE:
        if s and s.plural:
            return f"Are there any {pluralize(sp)} in the image?"
        return f"Is there {'an' if sp[0] in 'aeiou' else 'a'} {sp} in the image?"
    if t == ClaimType.OBJECT_IDENTITY:
        what = claim.attribute or sp
        base = "person" if s and s.canonical_class == "person" else (s.canonical_class or sp if s else "object")
        return f"Is this {base} {'an' if what[0] in 'aeiou' else 'a'} {what}?"
    if t == ClaimType.ATTRIBUTE_COLOR:
        return f"Is the {sp} {claim.attribute}?"
    if t == ClaimType.ATTRIBUTE_OTHER:
        return f"Is the {sp} {claim.attribute}?"
    if t == ClaimType.COUNT:
        n = claim.count or 0
        word = pluralize(sp) if n != 1 else sp
        if claim.count_comparator == CountComparator.AT_LEAST:
            return f"Are there at least {n} {word} in the image?"
        return f"Are there exactly {n} {word} in the image?"
    if t == ClaimType.ACTION:
        return f"Is the {sp} {gerund(claim.predicate or 'do')}?"
    if t == ClaimType.SPATIAL_RELATION and claim.object is not None:
        rel = _REL_TEXT.get(claim.spatial_relation or "", claim.spatial_relation or "near")
        return f"Is the {sp} {rel} the {claim.object.phrase}?"
    if t == ClaimType.INTERACTION and claim.object is not None:
        verb = claim.predicate or "touch"
        if verb == "wear":
            return f"Is the {sp} wearing {'an' if claim.object.phrase[0] in 'aeiou' else 'a'} {claim.object.phrase}?"
        if " " in verb:
            v, prep = verb.split(" ", 1)
            return f"Is the {sp} {gerund(v)} {prep} the {claim.object.phrase}?"
        return f"Is the {sp} {gerund(verb)} the {claim.object.phrase}?"
    if t == ClaimType.SCENE:
        return f"Does the image show {'an' if sp[0] in 'aeiou' else 'a'} {sp}?"
    if t == ClaimType.TEXT_OCR:
        txt = (claim.attribute or "").replace('"', "'")
        return f'Does the image contain the written text "{txt}"?'
    if t == ClaimType.TEMPORAL_EVENT:
        return f"Is the {sp} present?"
    return claim.text


class EvidenceCollector:
    def __init__(self, *, detector=None, grounder=None, vlm=None, policy: VerificationPolicy,
                 crop_sink: CropSink | None = None, inference_guard=None) -> None:
        self.detector = detector
        self.grounder = grounder
        self.vlm = vlm
        self.policy = policy
        self.crop_sink = crop_sink
        self._guard = inference_guard
        self.stats: dict[str, ComponentStats] = defaultdict(ComponentStats)

    # ------------------------------------------------------------------ cached model calls
    def _timed(self, name: str, fn, *args):
        t0 = time.perf_counter()
        if self._guard is not None:
            with self._guard():
                out = fn(*args)
        else:
            out = fn(*args)
        st = self.stats[name]
        st.calls += 1
        st.total_ms += (time.perf_counter() - t0) * 1000
        return out

    def detections(self, ctx: FrameContext) -> list[Detection]:
        if self.detector is None or not self.policy.components.detector:
            return []
        if ctx.detections is None:
            ctx.detections = self._timed("detector", self.detector.detect, ctx.image)
        return ctx.detections

    def ground(self, ctx: FrameContext, phrase: str) -> list[GroundingCandidate]:
        if self.grounder is None or not self.policy.components.grounding:
            return []
        key = phrase.lower().strip()
        if key not in ctx.grounding:
            ctx.grounding[key] = self._timed("grounding", self.grounder.ground, ctx.image, key)
        return ctx.grounding[key]

    def ask(self, ctx: FrameContext, question: str, region: BoundingBox | None = None) -> VLMJudgment | None:
        if self.vlm is None or not self.policy.components.vlm:
            return None
        rkey = None
        img = ctx.image
        if region is not None and self.policy.components.vlm_on_crops:
            rkey = (round(region.x1, 3), round(region.y1, 3), round(region.x2, 3), round(region.y2, 3))
            img = crop_image(ctx.image, region, pad=0.15)
        k = (question, rkey)
        if k not in ctx.vlm:
            ctx.vlm[k] = self._timed("vlm", self.vlm.judge, img, question)
        return ctx.vlm[k]

    # ------------------------------------------------------------------ helpers
    def _item(self, claim: AtomicClaim, ctx: FrameContext, **kw) -> EvidenceItem:
        return EvidenceItem(evidence_id=f"ev-{uuid.uuid4().hex[:12]}", claim_id=claim.claim_id,
                            frame_index=ctx.frame_index, timestamp_s=ctx.timestamp_s, **kw)

    def _crop_key(self, ctx: FrameContext, box: BoundingBox | None) -> str | None:
        if self.crop_sink is None or box is None:
            return None
        try:
            return self.crop_sink(ctx.image, box, ctx.frame_index)
        except Exception:  # noqa: BLE001 - crops are illustrative; never fail verification on them
            return None

    def _model(self, comp) -> tuple[str | None, str | None]:
        info = getattr(comp, "info", None)
        return (info.name, info.version) if info else (None, None)

    def localize(self, ctx: FrameContext, entity: EntityRef, phrase: str | None = None) -> list[Candidate]:
        th = self.policy.thresholds
        cands: list[Candidate] = []
        cls = entity.canonical_class
        if cls and self.detector is not None and self.policy.components.detector and cls in self.detector.vocabulary:
            for d in self.detections(ctx):
                if d.label == cls and d.confidence >= th.detector_weak:
                    cands.append(Candidate(d.box, d.confidence, EvidenceSource.DETECTOR, d.label,
                                           d.confidence >= th.detector_strong))
        for g in self.ground(ctx, phrase or entity.phrase):
            if g.score >= th.grounding_weak:
                if any(c.box.iou(g.box) > 0.5 for c in cands):
                    continue
                cands.append(Candidate(g.box, g.score, EvidenceSource.GROUNDING, g.phrase, g.score >= th.grounding_strong))
        # detector first (closed-vocabulary, typically better calibrated boxes), then by score
        cands.sort(key=lambda c: (c.source != EvidenceSource.DETECTOR, -c.score))
        return cands

    def _vlm_item(self, claim: AtomicClaim, ctx: FrameContext, question: str, region: BoundingBox | None,
                  scope: str) -> EvidenceItem | None:
        j = self.ask(ctx, question, region)
        if j is None:
            return None
        th = self.policy.thresholds
        p = j.p_yes
        name, version = self._model(self.vlm)
        if p is None:  # structured answer without probability (external adapter)
            stance = {"yes": Stance.SUPPORTS, "no": Stance.CONTRADICTS}.get(j.answer, Stance.NEUTRAL)
            strength = Strength.STRONG if stance != Stance.NEUTRAL else Strength.NONE
            if stance == Stance.SUPPORTS and j.conflicts:
                stance, strength = Stance.NEUTRAL, Strength.NONE
        elif p >= th.vlm_strong:
            stance, strength = Stance.SUPPORTS, Strength.STRONG
        elif p >= th.vlm_weak:
            stance, strength = Stance.SUPPORTS, Strength.WEAK
        elif p <= 1 - th.vlm_strong:
            stance, strength = Stance.CONTRADICTS, Strength.STRONG
        elif p <= 1 - th.vlm_weak:
            stance, strength = Stance.CONTRADICTS, Strength.WEAK
        else:
            stance, strength = Stance.NEUTRAL, Strength.NONE
        ptxt = f"P(yes)={p:.2f}" if p is not None else f"answer={j.answer}"
        desc = f"asked '{question}' on {scope} -> {j.answer} ({ptxt})"
        if j.visible_evidence:
            desc += f"; sees: {j.visible_evidence[:160]}"
        return self._item(claim, ctx, source=EvidenceSource.VLM, model_name=name, model_version=version,
                          stance=stance, strength=strength, raw_score=p, box=region if scope == "crop" else None,
                          description=desc,
                          details={"question": question, "answer": j.answer, "scope": scope,
                                   "undetermined": j.undetermined, "relevant_region": j.relevant_region,
                                   "model_reported_conflict": j.conflicts, "hypothesis_only": True,
                                   **({"raw": j.raw} if j.raw else {})})

    # ------------------------------------------------------------------ per-claim collection
    def collect(self, claim: AtomicClaim, ctx: FrameContext) -> list[EvidenceItem]:
        if not claim.verifiable or claim.claim_type in (ClaimType.NONVISUAL, ClaimType.TEMPORAL_EVENT):
            return []
        handler = {
            ClaimType.OBJECT_EXISTENCE: self._existence,
            ClaimType.COUNT: self._count,
            ClaimType.OBJECT_IDENTITY: self._identity,
            ClaimType.ATTRIBUTE_COLOR: self._color,
            ClaimType.ATTRIBUTE_OTHER: self._region_vlm,
            ClaimType.ACTION: self._region_vlm,
            ClaimType.SPATIAL_RELATION: self._spatial,
            ClaimType.INTERACTION: self._interaction,
            ClaimType.SCENE: self._scene,
            ClaimType.TEXT_OCR: self._ocr,
        }.get(claim.claim_type)
        return handler(claim, ctx) if handler else []

    def _detector_presence(self, claim: AtomicClaim, ctx: FrameContext, cls: str | None, label: str) -> EvidenceItem | None:
        if self.detector is None or not self.policy.components.detector:
            return None
        name, version = self._model(self.detector)
        th = self.policy.thresholds
        if not cls or cls not in self.detector.vocabulary:
            return self._item(claim, ctx, source=EvidenceSource.DETECTOR, model_name=name, model_version=version,
                              stance=Stance.NOT_APPLICABLE, label=label,
                              description=f"'{label}' is outside the detector vocabulary; detector cannot assess it.")
        dets = [d for d in self.detections(ctx) if d.label == cls]
        best = max(dets, key=lambda d: d.confidence, default=None)
        if best is not None and best.confidence >= th.detector_weak:
            strong = best.confidence >= th.detector_strong
            return self._item(claim, ctx, source=EvidenceSource.DETECTOR, model_name=name, model_version=version,
                              stance=Stance.SUPPORTS, strength=Strength.STRONG if strong else Strength.WEAK,
                              raw_score=best.confidence, box=best.box, label=cls,
                              crop_key=self._crop_key(ctx, best.box),
                              description=f"detected '{cls}' (confidence {best.confidence:.2f})",
                              details={"instances": len([d for d in dets if d.confidence >= th.detector_weak])})
        return self._item(claim, ctx, source=EvidenceSource.DETECTOR, model_name=name, model_version=version,
                          stance=Stance.ABSENT, label=cls, raw_score=best.confidence if best else None,
                          description=f"no '{cls}' detection above {th.detector_weak:.2f}",
                          details={"searched_class": cls, "max_confidence_seen": best.confidence if best else None})

    def _grounding_presence(self, claim: AtomicClaim, ctx: FrameContext, phrase: str, scene: bool = False) -> EvidenceItem | None:
        if self.grounder is None or not self.policy.components.grounding:
            return None
        name, version = self._model(self.grounder)
        th = self.policy.thresholds
        cands = self.ground(ctx, phrase)
        best = cands[0] if cands else None
        if best is not None and best.score >= th.grounding_weak:
            strong = best.score >= th.grounding_strong and not scene
            details: dict = {"candidates": len([c for c in cands if c.score >= th.grounding_weak])}
            if details["candidates"] > 1:
                details["ambiguous"] = False  # several instances are fine for existence; flagged elsewhere
            # cross-check: is the grounded region confidently detected as some *other* class?
            cls = canonical_class(phrase.split()[-1])
            if self.detector is not None and self.policy.components.detector and cls in self.detector.vocabulary:
                for d in self.detections(ctx):
                    if (d.label != cls and d.confidence >= th.other_class_conflict_conf
                            and d.box.iou(best.box) >= th.other_class_conflict_iou):
                        details["overlaps_other_class"] = d.label
                        details["overlap_confidence"] = round(d.confidence, 3)
                        break
            return self._item(claim, ctx, source=EvidenceSource.GROUNDING, model_name=name, model_version=version,
                              stance=Stance.SUPPORTS, strength=Strength.STRONG if strong else Strength.WEAK,
                              raw_score=best.score, box=best.box, label=phrase,
                              crop_key=self._crop_key(ctx, best.box),
                              description=f"grounded '{phrase}' (score {best.score:.2f})", details=details)
        stance = Stance.NEUTRAL if scene else Stance.ABSENT
        return self._item(claim, ctx, source=EvidenceSource.GROUNDING, model_name=name, model_version=version,
                          stance=stance, label=phrase, raw_score=best.score if best else None,
                          description=f"no region for '{phrase}' above {th.grounding_weak:.2f}",
                          details={"max_score_seen": best.score if best else None})

    def _existence(self, claim: AtomicClaim, ctx: FrameContext) -> list[EvidenceItem]:
        s = claim.subject
        assert s is not None
        items = [
            self._detector_presence(claim, ctx, s.canonical_class, s.phrase),
            self._grounding_presence(claim, ctx, s.phrase),
            self._vlm_item(claim, ctx, question_for(claim), None, "full image"),
        ]
        return [i for i in items if i is not None]

    def _identity(self, claim: AtomicClaim, ctx: FrameContext) -> list[EvidenceItem]:
        s = claim.subject
        assert s is not None
        target = claim.attribute or s.phrase
        target_cls = canonical_class(target)
        items: list[EvidenceItem | None] = []
        region = None
        if target_cls and target_cls != s.canonical_class:
            # "the animal is a dog": the detector can check the specific class directly
            items.append(self._detector_presence(claim, ctx, target_cls, target))
            items.append(self._grounding_presence(claim, ctx, target))
        cands = self.localize(ctx, s, s.canonical_class or s.phrase)
        if cands:
            region = cands[0].box
        items.append(self._vlm_item(claim, ctx, question_for(claim), region, "crop" if region else "full image"))
        return [i for i in items if i is not None]

    def _count(self, claim: AtomicClaim, ctx: FrameContext) -> list[EvidenceItem]:
        s = claim.subject
        assert s is not None
        n = claim.count or 0
        at_least = claim.count_comparator == CountComparator.AT_LEAST
        th = self.policy.thresholds
        items: list[EvidenceItem | None] = []

        def judge_count(n_strong: int, n_weak: int) -> tuple[Stance, Strength, str]:
            if at_least:
                if n_strong >= n:
                    return Stance.SUPPORTS, Strength.STRONG, f"{n_strong} confident instances (>= {n})"
                if n_weak >= n:
                    return Stance.SUPPORTS, Strength.WEAK, f"{n_weak} instances incl. low-confidence (>= {n})"
                if n_weak == 0:
                    return Stance.ABSENT, Strength.NONE, "no instances found"
                return Stance.CONTRADICTS, Strength.WEAK, f"only {n_weak} instances found (< {n}); some may be occluded"
            if n_strong == n and n_weak == n:
                return Stance.SUPPORTS, Strength.STRONG, f"exactly {n} instances found"
            if n_strong <= n <= n_weak:
                return Stance.SUPPORTS, Strength.WEAK, f"{n_strong} confident / {n_weak} total instances bracket {n}"
            if n_strong > n:
                return Stance.CONTRADICTS, Strength.STRONG, f"{n_strong} confident instances (> {n})"
            if n_weak == 0:
                return Stance.ABSENT, Strength.NONE, "no instances found"
            return Stance.CONTRADICTS, Strength.WEAK, f"only {n_weak} instances found (< {n}); some may be occluded"

        if self.detector is not None and self.policy.components.detector:
            name, version = self._model(self.detector)
            cls = s.canonical_class
            if cls and cls in self.detector.vocabulary:
                dets = [d for d in self.detections(ctx) if d.label == cls]
                n_strong = sum(d.confidence >= th.detector_strong for d in dets)
                n_weak = sum(d.confidence >= th.detector_weak for d in dets)
                st, sg, why = judge_count(n_strong, n_weak)
                boxes = [d.box for d in dets if d.confidence >= th.detector_weak]
                items.append(self._item(claim, ctx, source=EvidenceSource.DETECTOR, model_name=name,
                                        model_version=version, stance=st, strength=sg, label=cls,
                                        box=boxes[0] if len(boxes) == 1 else None,
                                        description=f"counted '{cls}': {why}",
                                        details={"n_strong": n_strong, "n_weak": n_weak, "claimed": n,
                                                 "boxes": [b.model_dump() for b in boxes[:20]]}))
            else:
                items.append(self._item(claim, ctx, source=EvidenceSource.DETECTOR, model_name=name,
                                        model_version=version, stance=Stance.NOT_APPLICABLE, label=s.phrase,
                                        description=f"'{s.phrase}' is outside the detector vocabulary"))
        if self.grounder is not None and self.policy.components.grounding:
            name, version = self._model(self.grounder)
            cands = self.ground(ctx, s.phrase)
            n_strong = sum(c.score >= th.grounding_strong for c in cands)
            n_weak = sum(c.score >= th.grounding_weak for c in cands)
            st, sg, why = judge_count(n_strong, n_weak)
            if sg == Strength.STRONG:
                sg = Strength.WEAK  # grounding counts are less reliable than closed-set detection
            items.append(self._item(claim, ctx, source=EvidenceSource.GROUNDING, model_name=name,
                                    model_version=version, stance=st, strength=sg, label=s.phrase,
                                    description=f"grounded '{s.phrase}': {why}",
                                    details={"n_strong": n_strong, "n_weak": n_weak, "claimed": n,
                                             "boxes": [c.box.model_dump() for c in cands if c.score >= th.grounding_weak][:20]}))
        items.append(self._vlm_item(claim, ctx, question_for(claim), None, "full image"))
        return [i for i in items if i is not None]

    def _color(self, claim: AtomicClaim, ctx: FrameContext) -> list[EvidenceItem]:
        s = claim.subject
        assert s is not None
        th = self.policy.thresholds
        items: list[EvidenceItem | None] = []
        cands = self.localize(ctx, s)
        region = cands[0].box if cands else None
        allowed = COLOR_WORDS.get((claim.attribute or "").lower(), {(claim.attribute or "").lower()})
        if self.policy.components.color:
            if region is None and not s.is_scene:
                items.append(self._item(claim, ctx, source=EvidenceSource.COLOR_ANALYSIS, stance=Stance.NOT_APPLICABLE,
                                        model_name="hsv-color-bins", model_version="1.0",
                                        description=f"no localized region for '{s.phrase}'; color not measured"))
            else:
                prof = analyze_region(ctx.image, region)
                frac = prof.fraction(allowed)
                top_color, top_frac = prof.dominant[0]
                ranked = [c for c, _ in prof.dominant[:2]]
                if frac >= th.color_support_strong and any(c in allowed for c in ranked):
                    st, sg = Stance.SUPPORTS, Strength.STRONG
                elif frac >= th.color_support_weak:
                    st, sg = Stance.SUPPORTS, Strength.WEAK
                elif frac <= th.color_contradict_max and top_frac >= th.color_dominant_min:
                    st, sg = Stance.CONTRADICTS, Strength.WEAK
                else:
                    st, sg = Stance.NEUTRAL, Strength.NONE
                top3 = ", ".join(f"{c} {v:.0%}" for c, v in prof.dominant[:3])
                items.append(self._item(
                    claim, ctx, source=EvidenceSource.COLOR_ANALYSIS, model_name="hsv-color-bins",
                    model_version="1.0", stance=st, strength=sg, raw_score=round(frac, 4), box=region,
                    crop_key=self._crop_key(ctx, region) if region else None,
                    description=f"'{claim.attribute}' pixels {frac:.0%} of {'region' if region else 'image'} (top: {top3})",
                    details={"distribution": prof.distribution, "allowed_bins": sorted(allowed),
                             "region_source": cands[0].source.value if cands else None,
                             "limitations": "lighting, shadows and background pixels affect pixel colors"}))
        items.append(self._vlm_item(claim, ctx, question_for(claim), region, "crop" if region else "full image"))
        return [i for i in items if i is not None]

    def _region_vlm(self, claim: AtomicClaim, ctx: FrameContext) -> list[EvidenceItem]:
        s = claim.subject
        region = None
        if s is not None:
            cands = self.localize(ctx, s)
            if cands:
                region = cands[0].box
                if len(cands) > 1 and cands[1].strong:
                    region = None  # several plausible referents: ask about the whole image instead
        it = self._vlm_item(claim, ctx, question_for(claim), region, "crop" if region else "full image")
        return [it] if it else []

    def _pair_candidates(self, claim: AtomicClaim, ctx: FrameContext) -> tuple[list[Candidate], list[Candidate]]:
        assert claim.subject is not None and claim.object is not None
        return self.localize(ctx, claim.subject)[:3], self.localize(ctx, claim.object)[:3]

    def _spatial(self, claim: AtomicClaim, ctx: FrameContext) -> list[EvidenceItem]:
        items: list[EvidenceItem | None] = []
        if claim.object is None:
            return []
        subj_c, obj_c = self._pair_candidates(claim, ctx)
        w, h = ctx.size
        rel = claim.spatial_relation or "near"
        union = None
        if self.policy.components.geometry:
            if not subj_c or not obj_c:
                missing = claim.subject.phrase if not subj_c else claim.object.phrase
                items.append(self._item(claim, ctx, source=EvidenceSource.GEOMETRY, stance=Stance.NOT_APPLICABLE,
                                        model_name="box-geometry", model_version="1.0",
                                        description=f"could not localize '{missing}'; relation not measured"))
            else:
                results = []
                for a in subj_c:
                    for b in obj_c:
                        if a.box.iou(b.box) > 0.9:
                            continue
                        results.append((evaluate_relation(rel, a.box, b.box, w, h), a, b))
                if results:
                    order = {Stance.SUPPORTS: 0, Stance.NEUTRAL: 1, Stance.CONTRADICTS: 2, Stance.NOT_APPLICABLE: 3}
                    best = min(results, key=lambda r: (order[r[0].stance], r[0].strength != Strength.STRONG))
                    gres, a, b = best
                    stances = {r[0].stance for r in results}
                    ambiguous = len(results) > 1 and Stance.SUPPORTS in stances and Stance.CONTRADICTS in stances
                    union = a.box.union(b.box)
                    items.append(self._item(
                        claim, ctx, source=EvidenceSource.GEOMETRY, model_name="box-geometry", model_version="1.0",
                        stance=gres.stance, strength=gres.strength, box=union,
                        crop_key=self._crop_key(ctx, union),
                        description=f"{gres.reason} ({a.label} vs {b.label})",
                        details={"metrics": gres.metrics, "subject_box": a.box.model_dump(),
                                 "object_box": b.box.model_dump(), "pairs_evaluated": len(results),
                                 "ambiguous": ambiguous}))
        items.append(self._vlm_item(claim, ctx, question_for(claim), None, "full image"))
        return [i for i in items if i is not None]

    def _interaction(self, claim: AtomicClaim, ctx: FrameContext) -> list[EvidenceItem]:
        items: list[EvidenceItem | None] = []
        if claim.object is None:
            return []
        subj_c, obj_c = self._pair_candidates(claim, ctx)
        w, h = ctx.size
        union = None
        if subj_c and obj_c:
            pairs = [(a, b) for a in subj_c for b in obj_c if a.box.iou(b.box) < 0.95]
            if pairs:
                a, b = min(pairs, key=lambda p: (evaluate_interaction(claim.predicate or "", p[0].box, p[1].box, w, h)
                                                 .metrics["gap_rel"]))
                union = a.box.union(b.box)
                if self.policy.components.geometry:
                    gres = evaluate_interaction(claim.predicate or "", a.box, b.box, w, h)
                    items.append(self._item(
                        claim, ctx, source=EvidenceSource.GEOMETRY, model_name="box-geometry", model_version="1.0",
                        stance=gres.stance, strength=gres.strength, box=union, crop_key=self._crop_key(ctx, union),
                        description=gres.reason,
                        details={"metrics": gres.metrics, "subject_box": a.box.model_dump(),
                                 "object_box": b.box.model_dump(),
                                 "note": "Box overlap never establishes an interaction by itself."}))
        elif self.policy.components.geometry:
            items.append(self._item(claim, ctx, source=EvidenceSource.GEOMETRY, stance=Stance.NOT_APPLICABLE,
                                    model_name="box-geometry", model_version="1.0",
                                    description="could not localize both participants; proximity not measured"))
        items.append(self._vlm_item(claim, ctx, question_for(claim), union, "crop" if union else "full image"))
        return [i for i in items if i is not None]

    def _scene(self, claim: AtomicClaim, ctx: FrameContext) -> list[EvidenceItem]:
        s = claim.subject
        items: list[EvidenceItem | None] = []
        if s is not None:
            items.append(self._grounding_presence(claim, ctx, s.phrase, scene=True))
        items.append(self._vlm_item(claim, ctx, question_for(claim), None, "full image"))
        return [i for i in items if i is not None]

    def _ocr(self, claim: AtomicClaim, ctx: FrameContext) -> list[EvidenceItem]:
        it = self._vlm_item(claim, ctx, question_for(claim), None, "full image")
        if it is None:
            return []
        it.details["note"] = "No dedicated OCR engine is integrated; text claims rely on the VLM only."
        return [it]
