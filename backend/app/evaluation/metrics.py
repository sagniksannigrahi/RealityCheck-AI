"""Evaluation metrics for claim verification with abstention.

Conventions
- Gold labels: SUPPORTED or CONTRADICTED (examples with other gold labels are reported separately).
- Predictions: SUPPORTED, CONTRADICTED or ABSTAIN (INSUFFICIENT_EVIDENCE / REVIEW_REQUIRED / UNVERIFIABLE).
- "Hallucination detection" treats CONTRADICTED as the positive class.
- Precision/recall/F1 are computed on *covered* (non-abstained) examples; ``*_effective`` variants count
  abstentions as errors, so selective systems cannot look better merely by abstaining.
- Confidence intervals use a cluster bootstrap over images (claims on the same image are not independent).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

LABELS = ("SUPPORTED", "CONTRADICTED")
PRED_LABELS = ("SUPPORTED", "CONTRADICTED", "ABSTAIN")


@dataclass
class Prediction:
    example_id: str
    cluster: str  # image id
    gold: str
    pred: str  # SUPPORTED | CONTRADICTED | ABSTAIN
    prob_supported: float | None = None
    category: str = "all"
    ambiguous: bool = False
    latency_ms: float = 0.0


def normalize_pred(verdict: str) -> str:
    return verdict if verdict in LABELS else "ABSTAIN"


def _safe(n: float, d: float) -> float | None:
    return n / d if d else None


def confusion(preds: Sequence[Prediction]) -> dict[str, dict[str, int]]:
    m = {g: {p: 0 for p in PRED_LABELS} for g in LABELS}
    for p in preds:
        if p.gold in LABELS:
            m[p.gold][p.pred] += 1
    return m


def _prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    """Precision/recall/F1. A class that is never predicted but exists in gold gets F1 = 0 (precision stays
    undefined); F1 is undefined only when the class has no gold and no predicted examples."""
    prec = _safe(tp, tp + fp)
    rec = _safe(tp, tp + fn)
    if prec is not None and rec is not None:
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    elif tp + fn > 0 or tp + fp > 0:
        f1 = 0.0
    else:
        f1 = None
    return prec, rec, f1


def core_metrics(preds: Sequence[Prediction]) -> dict[str, float | None]:
    preds = [p for p in preds if p.gold in LABELS]
    n = len(preds)
    covered = [p for p in preds if p.pred != "ABSTAIN"]
    out: dict[str, float | None] = {"n": float(n), "n_covered": float(len(covered))}
    out["coverage"] = _safe(len(covered), n)
    out["abstention_rate"] = None if out["coverage"] is None else 1 - out["coverage"]
    correct = sum(p.pred == p.gold for p in covered)
    out["accuracy_covered"] = _safe(correct, len(covered))  # == micro-F1 on covered examples
    out["accuracy_effective"] = _safe(correct, n)
    f1s = []
    for pos in LABELS:
        tp = sum(p.pred == pos and p.gold == pos for p in covered)
        fp = sum(p.pred == pos and p.gold != pos for p in covered)
        fn = sum(p.pred != pos and p.gold == pos for p in covered)
        prec, rec, f1 = _prf(tp, fp, fn)
        key = "hallucination" if pos == "CONTRADICTED" else "supported"
        out[f"{key}_precision"], out[f"{key}_recall"], out[f"{key}_f1"] = prec, rec, f1
        # effective recall: abstentions on gold-positive examples count as misses
        gold_pos = sum(p.gold == pos for p in preds)
        out[f"{key}_recall_effective"] = _safe(tp, gold_pos)
        if f1 is not None:
            f1s.append(f1)
    out["macro_f1"] = sum(f1s) / len(f1s) if len(f1s) == 2 else None
    out["micro_f1"] = out["accuracy_covered"]
    # error rates in the hallucination-detection framing
    sup_cov = [p for p in covered if p.gold == "SUPPORTED"]
    con_cov = [p for p in covered if p.gold == "CONTRADICTED"]
    out["false_positive_rate"] = _safe(sum(p.pred == "CONTRADICTED" for p in sup_cov), len(sup_cov))
    out["false_negative_rate"] = _safe(sum(p.pred == "SUPPORTED" for p in con_cov), len(con_cov))
    out["mean_latency_ms"] = sum(p.latency_ms for p in preds) / n if n else None
    return out


def calibration(preds: Sequence[Prediction], bins: int = 10) -> dict[str, float | None]:
    """ECE and Brier score for P(SUPPORTED); only examples with a probability are used."""
    pts = [(p.prob_supported, 1.0 if p.gold == "SUPPORTED" else 0.0) for p in preds
           if p.prob_supported is not None and p.gold in LABELS]
    if not pts:
        return {"ece": None, "brier": None, "n_calibration": 0.0}
    brier = sum((pr - y) ** 2 for pr, y in pts) / len(pts)
    buckets: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for pr, y in pts:
        buckets[min(bins - 1, int(pr * bins))].append((pr, y))
    ece = sum(len(b) / len(pts) * abs(sum(x for x, _ in b) / len(b) - sum(y for _, y in b) / len(b))
              for b in buckets.values())
    reliability = [{"bin": k, "n": len(b), "mean_prob": sum(x for x, _ in b) / len(b),
                    "frac_supported": sum(y for _, y in b) / len(b)} for k, b in sorted(buckets.items())]
    return {"ece": ece, "brier": brier, "n_calibration": float(len(pts)), "reliability": reliability}  # type: ignore[dict-item]


def cluster_bootstrap(preds: Sequence[Prediction], stat: Callable[[Sequence[Prediction]], float | None],
                      n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float | None, float | None]:
    clusters: dict[str, list[Prediction]] = defaultdict(list)
    for p in preds:
        clusters[p.cluster].append(p)
    keys = sorted(clusters)
    if len(keys) < 2:
        return None, None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample: list[Prediction] = []
        for k in rng.choices(keys, k=len(keys)):
            sample.extend(clusters[k])
        v = stat(sample)
        if v is not None and not math.isnan(v):
            vals.append(v)
    if len(vals) < n_boot * 0.5:
        return None, None
    vals.sort()
    lo = vals[int(alpha / 2 * len(vals))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return lo, hi


def paired_bootstrap_diff(a: Sequence[Prediction], b: Sequence[Prediction],
                          stat: Callable[[Sequence[Prediction]], float | None], n_boot: int = 1000,
                          seed: int = 0) -> dict[str, float | None]:
    """Difference stat(a) - stat(b) on the same examples, with a cluster-bootstrap 95% CI and p-value
    (fraction of resamples where the sign flips, two-sided)."""
    amap = {p.example_id: p for p in a}
    bmap = {p.example_id: p for p in b}
    ids = sorted(set(amap) & set(bmap))
    clusters: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        clusters[amap[i].cluster].append(i)
    keys = sorted(clusters)
    base_a, base_b = stat([amap[i] for i in ids]), stat([bmap[i] for i in ids])
    if base_a is None or base_b is None or len(keys) < 2:
        return {"diff": None, "ci_low": None, "ci_high": None, "p_value": None}
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sel = [i for k in rng.choices(keys, k=len(keys)) for i in clusters[k]]
        sa, sb = stat([amap[i] for i in sel]), stat([bmap[i] for i in sel])
        if sa is not None and sb is not None:
            diffs.append(sa - sb)
    diffs.sort()
    d0 = base_a - base_b
    flips = sum(1 for d in diffs if (d <= 0 if d0 > 0 else d >= 0))
    return {"diff": d0, "ci_low": diffs[int(0.025 * len(diffs))], "ci_high": diffs[int(0.975 * len(diffs)) - 1],
            "p_value": min(1.0, 2 * flips / len(diffs)) if diffs else None}


def full_report(preds: Sequence[Prediction], *, n_boot: int = 1000, seed: int = 0) -> dict:
    core = core_metrics(preds)
    cis = {}
    for name in ("macro_f1", "accuracy_covered", "coverage", "hallucination_f1", "accuracy_effective"):
        lo, hi = cluster_bootstrap(preds, lambda s, n=name: core_metrics(s)[n], n_boot=n_boot, seed=seed)
        cis[name] = {"low": lo, "high": hi}
    per_cat = {}
    for cat in sorted({p.category for p in preds}):
        sub = [p for p in preds if p.category == cat]
        per_cat[cat] = core_metrics(sub)
    amb = [p for p in preds if p.ambiguous]
    return {
        "overall": core,
        "confidence_intervals_95": cis,
        "confusion_matrix": confusion(preds),
        "per_category": per_cat,
        "ambiguous_cases": core_metrics(amb) if amb else None,
        "calibration": calibration(preds),
    }
