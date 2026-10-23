"""2-D box geometry for spatial relations and interaction plausibility.

Geometry can *support* or *contradict* simple relations (left/right/above/below/on/inside/near).
It can NOT establish depth relations (behind / in front of) or interactions (holding, riding...):
for interactions it only checks a necessary condition (proximity), so overlapping boxes yield NEUTRAL,
never SUPPORTS.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.evidence import BoundingBox, Stance, Strength


@dataclass
class GeometryResult:
    stance: Stance
    strength: Strength
    reason: str
    metrics: dict[str, float]


def _to_px(b: BoundingBox, w: int, h: int) -> tuple[float, float, float, float]:
    return b.x1 * w, b.y1 * h, b.x2 * w, b.y2 * h


def _gap(a: tuple, b: tuple) -> float:
    """Euclidean gap between two pixel boxes (0 if they overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


def _inter(a: tuple, b: tuple) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _area(a: tuple) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def _diag(a: tuple) -> float:
    return ((a[2] - a[0]) ** 2 + (a[3] - a[1]) ** 2) ** 0.5


def evaluate_relation(rel: str, subj: BoundingBox, obj: BoundingBox, width: int, height: int) -> GeometryResult:
    s, o = _to_px(subj, width, height), _to_px(obj, width, height)
    scx, scy = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
    ocx, ocy = (o[0] + o[2]) / 2, (o[1] + o[3]) / 2
    ref = max(1.0, min(_diag(s), _diag(o)))
    gap = _gap(s, o)
    inter = _inter(s, o)
    contain_s = inter / max(1.0, _area(s))
    hov = max(0.0, min(s[2], o[2]) - max(s[0], o[0])) / max(1.0, min(s[2] - s[0], o[2] - o[0]))
    m = {"gap_rel": round(gap / ref, 3), "subj_in_obj": round(contain_s, 3), "h_overlap": round(hov, 3),
         "dx_rel": round((scx - ocx) / ref, 3), "dy_rel": round((scy - ocy) / ref, 3)}

    def r(st: Stance, sg: Strength, why: str) -> GeometryResult:
        return GeometryResult(st, sg, why, m)

    if rel in {"left_of", "right_of"}:
        sign = 1 if rel == "left_of" else -1
        d = sign * (ocx - scx) / ref  # positive when relation holds
        if (rel == "left_of" and scx < o[0]) or (rel == "right_of" and scx > o[2]):
            return r(Stance.SUPPORTS, Strength.STRONG, "Subject center lies beyond the object's edge in the claimed direction.")
        if d > 0.15:
            return r(Stance.SUPPORTS, Strength.WEAK, "Subject center is offset in the claimed direction.")
        if (rel == "left_of" and scx > o[2]) or (rel == "right_of" and scx < o[0]):
            return r(Stance.CONTRADICTS, Strength.STRONG, "Subject lies on the opposite side of the object.")
        if d < -0.15:
            return r(Stance.CONTRADICTS, Strength.WEAK, "Subject center is offset in the opposite direction.")
        return r(Stance.NEUTRAL, Strength.NONE, "Horizontal positions are too close to decide.")

    if rel in {"above", "below"}:
        sign = 1 if rel == "above" else -1
        d = sign * (ocy - scy) / ref
        if (rel == "above" and s[3] <= o[1] + 0.1 * (o[3] - o[1])) or (rel == "below" and s[1] >= o[3] - 0.1 * (o[3] - o[1])):
            return r(Stance.SUPPORTS, Strength.STRONG, "Subject box is entirely on the claimed side vertically.")
        if d > 0.15:
            return r(Stance.SUPPORTS, Strength.WEAK, "Subject center is vertically offset in the claimed direction.")
        if d < -0.3:
            return r(Stance.CONTRADICTS, Strength.WEAK, "Subject center is vertically offset in the opposite direction.")
        return r(Stance.NEUTRAL, Strength.NONE, "Vertical positions are too close to decide.")

    if rel == "on":
        # subject rests on top of object: horizontally overlapping, subject center above object center,
        # subject bottom within/near the object's upper part
        if hov >= 0.3 and scy < ocy and s[3] >= o[1] - 0.15 * (o[3] - o[1]) and s[3] <= o[3] + 0.05 * (o[3] - o[1]):
            return r(Stance.SUPPORTS, Strength.WEAK,
                     "Boxes are consistent with the subject resting on the object (2-D boxes cannot confirm contact).")
        if hov < 0.05 and gap / ref > 0.3:
            return r(Stance.CONTRADICTS, Strength.WEAK, "Boxes do not overlap horizontally and are separated.")
        if scy > o[3]:
            return r(Stance.CONTRADICTS, Strength.WEAK, "Subject is located below the object.")
        return r(Stance.NEUTRAL, Strength.NONE, "Box layout is inconclusive for 'on'.")

    if rel in {"under"}:
        if hov >= 0.3 and scy > ocy:
            return r(Stance.SUPPORTS, Strength.WEAK, "Subject is below the object with horizontal overlap.")
        if scy < o[1]:
            return r(Stance.CONTRADICTS, Strength.WEAK, "Subject is above the object.")
        return r(Stance.NEUTRAL, Strength.NONE, "Box layout is inconclusive for 'under'.")

    if rel == "inside":
        if contain_s >= 0.85:
            return r(Stance.SUPPORTS, Strength.WEAK, "Subject box lies within the object box (2-D containment).")
        if contain_s <= 0.05:
            return r(Stance.CONTRADICTS, Strength.STRONG if gap / ref > 0.2 else Strength.WEAK,
                     "Subject box is outside the object box.")
        return r(Stance.NEUTRAL, Strength.NONE, "Partial overlap; containment is inconclusive.")

    if rel in {"near", "next_to", "at"}:
        g = gap / ref
        if g <= 0.25:
            return r(Stance.SUPPORTS, Strength.STRONG if g <= 0.05 else Strength.WEAK, "Boxes are adjacent or overlapping.")
        if g >= 2.0:
            return r(Stance.CONTRADICTS, Strength.WEAK, "Boxes are far apart relative to object size.")
        return r(Stance.NEUTRAL, Strength.NONE, "Distance is intermediate.")

    return r(Stance.NOT_APPLICABLE, Strength.NONE,
             f"'{rel}' cannot be assessed from 2-D boxes (requires depth or 3-D reasoning).")


# Interactions for which physical proximity is a necessary condition.
CONTACT_INTERACTIONS = {
    "hold", "carry", "ride", "wear", "eat", "drink", "touch", "hug", "pet", "use", "kick", "hit", "push", "pull",
    "feed", "brush", "wash", "bite", "lick", "grab", "lift", "cut", "swing", "type", "hold up", "play with",
    "sit", "lie", "lean", "kiss", "slice", "pour", "drive", "read",
}


def evaluate_interaction(verb: str, subj: BoundingBox, obj: BoundingBox, width: int, height: int) -> GeometryResult:
    s, o = _to_px(subj, width, height), _to_px(obj, width, height)
    ref = max(1.0, min(_diag(s), _diag(o)))
    gap = _gap(s, o) / ref
    contain_o = _inter(s, o) / max(1.0, _area(o))
    m = {"gap_rel": round(gap, 3), "obj_in_subj": round(contain_o, 3)}
    if verb.split(" ")[0] not in {v.split(" ")[0] for v in CONTACT_INTERACTIONS}:
        return GeometryResult(Stance.NOT_APPLICABLE, Strength.NONE,
                              f"Proximity is not a necessary condition for '{verb}'.", m)
    if verb == "wear":
        if contain_o >= 0.6:
            return GeometryResult(Stance.NEUTRAL, Strength.NONE,
                                  "Item lies within the person's box: consistent with wearing, but not proof.", m)
        if contain_o <= 0.1 and gap > 0.2:
            return GeometryResult(Stance.CONTRADICTS, Strength.WEAK, "Item is outside the person's box.", m)
        return GeometryResult(Stance.NEUTRAL, Strength.NONE, "Partial overlap; inconclusive.", m)
    if gap <= 0.1:
        return GeometryResult(Stance.NEUTRAL, Strength.NONE,
                              "Boxes touch or overlap: consistent with the interaction, but overlap does not prove it.", m)
    if gap >= 1.0:
        return GeometryResult(Stance.CONTRADICTS, Strength.WEAK,
                              "Boxes are clearly separated, so physical contact is unlikely.", m)
    return GeometryResult(Stance.NEUTRAL, Strength.NONE, "Boxes are close but not touching; inconclusive.", m)
