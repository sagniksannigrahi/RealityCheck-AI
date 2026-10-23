"""Reproducible evaluation harness: baselines, the EAV-SA method and ablations.

Protocol
1. Load normalized examples; assign an image-level val/test split (no image appears in both).
2. Collect evidence ONCE per example with every component enabled; cache to disk (resumable). All systems are
   decision functions over the same cached observations, so differences come from the decision logic and the
   components each system is allowed to use — not from re-running stochastic models.
3. For each system, select thresholds on the *val* split only (small declared grid); freeze; score on *test*.
4. Report metrics with cluster-bootstrap 95% CIs, paired comparisons, calibration and runtime.
"""

from __future__ import annotations

import itertools
import json
import math
import platform
import resource
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.imaging import decode_image, image_fingerprint, image_quality
from app.evaluation.datasets import EvalExample, split_of
from app.evaluation.metrics import Prediction, full_report, normalize_pred, paired_bootstrap_diff
from app.schemas.claims import ClaimType
from app.schemas.evidence import EvidenceItem, EvidenceSource, Stance, Strength
from app.services.fake_models import FINGERPRINT_KEY
from app.verification.collector import EvidenceCollector, FrameContext
from app.verification.decision import decide
from app.verification.policy import PlattCalibrator, SourceThresholds, VerificationPolicy

# ----------------------------------------------------------------------------------------------- cache


@dataclass
class CachedObservation:
    example_id: str
    items: list[EvidenceItem]
    quality: dict[str, float]
    component_ms: dict[str, float]
    detections: list[dict[str, Any]] = field(default_factory=list)  # raw detector output for the image

    def to_json(self) -> dict[str, Any]:
        return {"example_id": self.example_id, "items": [i.model_dump(mode="json") for i in self.items],
                "quality": self.quality, "component_ms": self.component_ms, "detections": self.detections}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> CachedObservation:
        return cls(example_id=d["example_id"], items=[EvidenceItem.model_validate(i) for i in d["items"]],
                   quality=d["quality"], component_ms=d["component_ms"], detections=d.get("detections", []))


def collect_observations(examples: list[EvalExample], registry, datasets_root: Path, cache_path: Path,
                         log: Callable[[str], None] = print) -> dict[str, CachedObservation]:
    """Run every component once per example (grouped by image) and append results to a JSONL cache."""
    cache: dict[str, CachedObservation] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                o = CachedObservation.from_json(json.loads(line))
                cache[o.example_id] = o
    todo = [e for e in examples if e.example_id not in cache]
    if not todo:
        return cache
    by_img: dict[str, list[EvalExample]] = defaultdict(list)
    for e in todo:
        by_img[e.image_path].append(e)
    policy = VerificationPolicy()
    # collect with permissive tiers; systems re-tier from raw scores
    collector = EvidenceCollector(detector=registry.detector, grounder=registry.grounder, vlm=registry.verifier,
                                  policy=policy, crop_sink=None, inference_guard=registry.inference)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = 0
    with cache_path.open("a") as fh:
        for img_path, exs in sorted(by_img.items()):
            p = datasets_root / img_path
            if not p.exists():
                log(f"missing image {img_path}; skipping {len(exs)} examples")
                continue
            img: Image.Image = decode_image(p.read_bytes())
            img.info[FINGERPRINT_KEY] = image_fingerprint(img)
            ctx = FrameContext(image=img, quality=image_quality(img))
            for e in exs:
                before = {k: v.total_ms for k, v in collector.stats.items()}
                items = collector.collect(e.claim, ctx)
                after = {k: v.total_ms for k, v in collector.stats.items()}
                comp_ms = {k: round(after[k] - before.get(k, 0.0), 2) for k in after}
                dets = [{"label": d.label, "confidence": d.confidence, "box": d.box.model_dump()}
                        for d in (ctx.detections or [])]
                obs = CachedObservation(e.example_id, items, ctx.quality, comp_ms, dets)
                cache[e.example_id] = obs
                fh.write(json.dumps(obs.to_json()) + "\n")
                done += 1
            fh.flush()
            if done and done % 50 < len(exs):
                rate = (time.time() - t0) / done
                log(f"collected {done}/{len(todo)} ({rate:.2f}s/example, ETA {rate * (len(todo) - done) / 60:.1f} min)")
    return cache


# ----------------------------------------------------------------------------------------------- re-tiering


def retier(items: list[EvidenceItem], th: SourceThresholds, claim_type: ClaimType) -> list[EvidenceItem]:
    """Recompute stance/strength of presence-type items from raw scores under thresholds ``th``."""
    out = []
    for it in items:
        it = it.model_copy()  # shallow: only scalar stance/strength fields are modified
        raw = it.raw_score
        presence = claim_type in (ClaimType.OBJECT_EXISTENCE, ClaimType.OBJECT_IDENTITY)
        if it.source == EvidenceSource.VLM and raw is not None:
            if raw >= th.vlm_strong:
                it.stance, it.strength = Stance.SUPPORTS, Strength.STRONG
            elif raw >= th.vlm_weak:
                it.stance, it.strength = Stance.SUPPORTS, Strength.WEAK
            elif raw <= 1 - th.vlm_strong:
                it.stance, it.strength = Stance.CONTRADICTS, Strength.STRONG
            elif raw <= 1 - th.vlm_weak:
                it.stance, it.strength = Stance.CONTRADICTS, Strength.WEAK
            else:
                it.stance, it.strength = Stance.NEUTRAL, Strength.NONE
        elif presence and it.source in (EvidenceSource.DETECTOR, EvidenceSource.GROUNDING) and \
                it.stance in (Stance.SUPPORTS, Stance.ABSENT):
            strong_t, weak_t = ((th.detector_strong, th.detector_weak) if it.source == EvidenceSource.DETECTOR
                                else (th.grounding_strong, th.grounding_weak))
            if raw is not None and raw >= weak_t:
                it.stance = Stance.SUPPORTS
                it.strength = Strength.STRONG if raw >= strong_t else Strength.WEAK
            else:
                it.stance, it.strength = Stance.ABSENT, Strength.NONE
        out.append(it)
    return out


# ----------------------------------------------------------------------------------------------- systems

SOURCE_OF_COMPONENT = {"detector": EvidenceSource.DETECTOR, "grounding": EvidenceSource.GROUNDING,
                       "vlm": EvidenceSource.VLM, "color": EvidenceSource.COLOR_ANALYSIS,
                       "geometry": EvidenceSource.GEOMETRY}
ALL_COMPONENTS = ("detector", "grounding", "vlm", "color", "geometry")


@dataclass
class SystemSpec:
    name: str
    kind: str  # vlm_only | detector_rules | eav
    description: str
    components: tuple[str, ...] = ALL_COMPONENTS
    abstention: bool = True
    disagreement: bool = True
    tune: bool = True


SYSTEMS: dict[str, SystemSpec] = {s.name: s for s in [
    SystemSpec("vlm_only", "vlm_only", "Baseline 1: VLM yes/no probability alone (threshold tuned on val).",
               ("vlm",), abstention=False),
    SystemSpec("detector_rules", "detector_rules",
               "Baseline 2: closed-vocabulary detector + deterministic rules (abstains outside vocabulary).",
               ("detector",), abstention=False),
    SystemSpec("vlm_grounding", "eav", "Baseline 3: VLM + independent open-vocabulary grounding (forced decision).",
               ("vlm", "grounding", "geometry", "color"), abstention=False),
    SystemSpec("full", "eav", "Baseline 4: full evidence pipeline, forced binary decision (no abstention).",
               ALL_COMPONENTS, abstention=False),
    SystemSpec("eav_sa", "eav", "Method: full pipeline with uncertainty-aware selective abstention (EAV-SA).",
               ALL_COMPONENTS, abstention=True),
    SystemSpec("eav_sa_default", "eav", "EAV-SA with default (untuned, pre-declared) thresholds.",
               ALL_COMPONENTS, abstention=True, tune=False),
    SystemSpec("ablate_no_grounding", "eav", "Ablation: EAV-SA without grounding.",
               ("detector", "vlm", "color", "geometry"), abstention=True),
    SystemSpec("ablate_no_detector", "eav", "Ablation: EAV-SA without the closed-vocabulary detector.",
               ("grounding", "vlm", "color", "geometry"), abstention=True),
    SystemSpec("ablate_no_disagreement", "eav", "Ablation: EAV-SA without cross-source disagreement analysis.",
               ALL_COMPONENTS, abstention=True, disagreement=False),
    SystemSpec("ablate_no_vlm", "eav", "Ablation: EAV-SA without the VLM (localizers + rules only).",
               ("detector", "grounding", "color", "geometry"), abstention=True),
]}

GRID = {
    "vlm_strong": [0.75, 0.85, 0.95],
    "detector_strong": [0.4, 0.5, 0.6],
    "grounding_strong": [0.35, 0.45, 0.55],
    "decision_threshold": [0.25, 0.35, 0.5],
}


def make_policy(spec: SystemSpec, params: dict[str, float] | None = None) -> VerificationPolicy:
    p = VerificationPolicy(name=f"eval-{spec.name}")
    for comp in ALL_COMPONENTS:
        setattr(p.components, comp, comp in spec.components)
    p.use_abstention = spec.abstention
    p.use_disagreement = spec.disagreement
    if params:
        p.thresholds.vlm_strong = params.get("vlm_strong", p.thresholds.vlm_strong)
        p.thresholds.vlm_weak = min(p.thresholds.vlm_strong - 0.05, max(0.55, p.thresholds.vlm_weak))
        p.thresholds.detector_strong = params.get("detector_strong", p.thresholds.detector_strong)
        p.thresholds.grounding_strong = params.get("grounding_strong", p.thresholds.grounding_strong)
        if "decision_threshold" in params:
            p.support_threshold = p.contradict_threshold = params["decision_threshold"]
    return p


_RETIER_CACHE: dict[tuple, list[EvidenceItem]] = {}


def predict_eav(ex: EvalExample, obs: CachedObservation, policy: VerificationPolicy) -> tuple[str, float]:
    allowed = {SOURCE_OF_COMPONENT[c] for c in ALL_COMPONENTS if getattr(policy.components, c)}
    th = policy.thresholds
    key = (ex.example_id, th.vlm_strong, th.vlm_weak, th.detector_strong, th.detector_weak, th.grounding_strong,
           th.grounding_weak)
    if key not in _RETIER_CACHE:
        if len(_RETIER_CACHE) > 500_000:
            _RETIER_CACHE.clear()
        _RETIER_CACHE[key] = retier(obs.items, th, ex.claim.claim_type)
    items = [i for i in _RETIER_CACHE[key] if i.source in allowed]
    d = decide(ex.claim, items, policy, image_quality=obs.quality)
    return normalize_pred(d.verdict.value), float(d.score or 0.0)


def _det_boxes(obs: CachedObservation, cls: str | None, thr: float) -> list[dict]:
    return [d for d in obs.detections if d["label"] == cls and d["confidence"] >= thr]


def predict_detector_rules(ex: EvalExample, obs: CachedObservation, thr: float, vocab: set[str]) -> str:
    c = ex.claim
    s = c.subject
    if s is None or s.canonical_class not in vocab:
        return "ABSTAIN"
    dets = _det_boxes(obs, s.canonical_class, thr)
    if c.claim_type == ClaimType.OBJECT_EXISTENCE:
        return "SUPPORTED" if dets else "CONTRADICTED"
    if c.claim_type == ClaimType.COUNT:
        return "SUPPORTED" if len(dets) == c.count else "CONTRADICTED"
    if c.claim_type == ClaimType.SPATIAL_RELATION and c.object is not None:
        if c.object.canonical_class not in vocab:
            return "ABSTAIN"
        o = _det_boxes(obs, c.object.canonical_class, thr)
        if not dets or not o:
            return "ABSTAIN"
        sx = (dets[0]["box"]["x1"] + dets[0]["box"]["x2"]) / 2
        ox = (o[0]["box"]["x1"] + o[0]["box"]["x2"]) / 2
        holds = sx < ox if c.spatial_relation == "left_of" else sx > ox
        return "SUPPORTED" if holds else "CONTRADICTED"
    return "ABSTAIN"


def vlm_prob(obs: CachedObservation) -> float | None:
    v = [i.raw_score for i in obs.items if i.source == EvidenceSource.VLM and i.raw_score is not None]
    return v[0] if v else None


def latency(obs: CachedObservation, components: tuple[str, ...]) -> float:
    name_map = {"detector": "detector", "grounding": "grounding", "vlm": "vlm"}
    return sum(obs.component_ms.get(name_map[c], 0.0) for c in components if c in name_map)


# ----------------------------------------------------------------------------------------------- objective


def _objective(preds: list[Prediction], abstaining: bool, min_coverage: float) -> float:
    from app.evaluation.metrics import core_metrics

    m = core_metrics(preds)
    f1 = m["macro_f1"] or 0.0
    if abstaining and (m["coverage"] or 0.0) < min_coverage:
        return -1.0 + (m["coverage"] or 0.0)  # infeasible: prefer higher coverage
    return f1


def fit_platt(scores: list[float], labels: list[int], iters: int = 2000, lr: float = 0.1) -> tuple[float, float]:
    a, b = 1.0, 0.0
    n = len(scores)
    for _ in range(iters):
        ga = gb = 0.0
        for s, y in zip(scores, labels, strict=True):
            p = 1 / (1 + math.exp(-(a * s + b)))
            ga += (p - y) * s
            gb += p - y
        a -= lr * ga / n
        b -= lr * gb / n
    return a, b


# ----------------------------------------------------------------------------------------------- runner


@dataclass
class SystemResult:
    spec: SystemSpec
    params: dict[str, float]
    val_objective: float | None
    test_preds: list[Prediction]
    report: dict[str, Any]
    calibrator: dict[str, Any] | None = None


def evaluate_system(spec: SystemSpec, examples: list[EvalExample], cache: dict[str, CachedObservation], *,
                    vocab: set[str], min_coverage: float, n_boot: int, seed: int) -> SystemResult:
    val = [e for e in examples if e.metadata.get("split") == "val" and e.example_id in cache]
    test = [e for e in examples if e.metadata.get("split") == "test" and e.example_id in cache]

    def run(exs: list[EvalExample], params: dict[str, float]) -> list[Prediction]:
        preds = []
        for e in exs:
            obs = cache[e.example_id]
            prob = None
            if spec.kind == "vlm_only":
                pv = vlm_prob(obs)
                pred = "ABSTAIN" if pv is None else ("SUPPORTED" if pv >= params["tau"] else "CONTRADICTED")
                prob = pv
            elif spec.kind == "detector_rules":
                pred = predict_detector_rules(e, obs, params["tau"], vocab)
            else:
                pred, score = predict_eav(e, obs, make_policy(spec, params))
                prob = score  # converted with a val-fitted Platt calibrator below
            preds.append(Prediction(e.example_id, e.image_id, e.gold, pred, prob, e.category, e.ambiguous,
                                    latency(obs, spec.components)))
        return preds

    if spec.kind == "vlm_only":
        grid = [{"tau": round(t, 2)} for t in [x / 20 for x in range(2, 19)]]
    elif spec.kind == "detector_rules":
        grid = [{"tau": t} for t in (0.15, 0.25, 0.35, 0.45, 0.55, 0.65)]
    else:
        keys = list(GRID)
        grid = [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(GRID[k] for k in keys))]
    if not spec.tune:
        grid = [{}] if spec.kind == "eav" else grid[:1]
    best, best_obj = grid[0], -math.inf
    if spec.tune and val:
        for params in grid:
            obj = _objective(run(val, params), spec.abstention, min_coverage)
            if obj > best_obj:
                best, best_obj = params, obj
    test_preds = run(test, best)
    calibrator = None
    if spec.kind == "eav":
        # Platt-calibrate the agreement score on val; report calibration of P(supported) on test
        vp = [p for p in run(val, best) if p.gold in ("SUPPORTED", "CONTRADICTED")]
        if len({p.gold for p in vp}) == 2:
            a, b = fit_platt([p.prob_supported or 0.0 for p in vp], [int(p.gold == "SUPPORTED") for p in vp])
            calibrator = {"a": a, "b": b, "fitted_on": "val", "n_examples": len(vp)}
            for p in test_preds:
                p.prob_supported = 1 / (1 + math.exp(-(a * (p.prob_supported or 0.0) + b)))
        else:
            for p in test_preds:
                p.prob_supported = None
    report = full_report(test_preds, n_boot=n_boot, seed=seed)
    return SystemResult(spec, best, None if best_obj == -math.inf else best_obj, test_preds, report, calibrator)


def run_experiment(*, experiment: str, examples: list[EvalExample], registry, datasets_root: Path, out_root: Path,
                   systems: list[str], seed: int, n_boot: int = 1000, min_coverage: float = 0.6,
                   simulated: bool = False, val_fraction: float = 0.3, log: Callable[[str], None] = print) -> dict:
    started = datetime.now(UTC)
    for e in examples:
        e.metadata["split"] = split_of(e.image_id, val_fraction)
    model_sig = "-".join(getattr(registry, c).info.name.replace("/", "_")
                         for c in ("detector", "grounder", "verifier"))
    datasets = sorted({e.dataset for e in examples})
    cache_path = out_root.parent / "cache" / f"{'_'.join(datasets)}__{model_sig}.jsonl"
    t0 = time.time()
    cache = collect_observations(examples, registry, datasets_root, cache_path, log=log)
    collect_s = time.time() - t0
    vocab = set(registry.detector.vocabulary)
    results: dict[str, SystemResult] = {}
    for name in systems:
        log(f"evaluating {name}")
        results[name] = evaluate_system(SYSTEMS[name], examples, cache, vocab=vocab, min_coverage=min_coverage,
                                        n_boot=n_boot, seed=seed)

    from app.evaluation.metrics import core_metrics

    def macro_f1(s):
        return core_metrics(s)["macro_f1"]

    def accuracy_covered(s):
        return core_metrics(s)["accuracy_covered"]

    mf1, acc_cov = macro_f1, accuracy_covered

    comparisons = {}
    pairs = [("full", "vlm_only", mf1), ("full", "detector_rules", mf1), ("full", "vlm_grounding", mf1),
             ("eav_sa", "full", acc_cov), ("eav_sa", "vlm_only", acc_cov)]
    pairs += [(a, "eav_sa", mf1) for a in systems if a.startswith("ablate_")]
    for a, b, fn in pairs:
        if a in results and b in results:
            comparisons[f"{a}_vs_{b}"] = {"metric": fn.__name__, **paired_bootstrap_diff(
                results[a].test_preds, results[b].test_preds, fn, n_boot=n_boot, seed=seed)}

    run_id = f"{experiment}-{started.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    out = out_root / run_id
    out.mkdir(parents=True, exist_ok=True)
    infos = {c: getattr(registry, c).info.model_dump() for c in ("detector", "grounder", "verifier")}
    split_counts = defaultdict(lambda: defaultdict(int))
    for e in examples:
        split_counts[e.metadata["split"]][e.gold] += 1
    config = {
        "experiment": experiment, "run_id": run_id, "seed": seed, "n_boot": n_boot, "min_coverage": min_coverage,
        "val_fraction": val_fraction, "systems": {n: SYSTEMS[n].__dict__ for n in systems}, "grid": GRID,
        "datasets": datasets, "dataset_versions": sorted({e.dataset_version for e in examples}),
        "n_examples": len(examples), "n_cached": sum(e.example_id in cache for e in examples),
        "split_label_counts": {k: dict(v) for k, v in split_counts.items()},
        "subsets": sorted({e.subset for e in examples}), "simulated": simulated, "models": infos,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "machine": platform.machine()},
        "evidence_cache": str(cache_path.relative_to(out_root.parent.parent)) if cache_path.is_relative_to(
            out_root.parent.parent) else cache_path.name,
    }
    metrics = {
        name: {"description": r.spec.description, "selected_params": r.params, "val_objective": r.val_objective,
               "calibrator": r.calibrator, **r.report}
        for name, r in results.items()
    }
    runtime = {"evidence_collection_s": round(collect_s, 1),
               "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if platform.system() == "Darwin" else 1024), 1)}
    summary = {"config": config, "metrics": metrics, "comparisons": comparisons, "runtime": runtime,
               "started_at": started.isoformat(), "finished_at": datetime.now(UTC).isoformat()}
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    for name, r in results.items():
        with (out / f"predictions_{name}.jsonl").open("w") as fh:
            for p in r.test_preds:
                fh.write(json.dumps(p.__dict__) + "\n")
        if name == "eav_sa" and r.calibrator:
            (out / "calibrator_eav_sa.json").write_text(json.dumps(PlattCalibrator(
                a=r.calibrator["a"], b=r.calibrator["b"], fitted_on=f"{run_id}:val",
                n_examples=r.calibrator["n_examples"],
                ece_validation=None).model_dump(), indent=2))
    (out / "report.md").write_text(render_report(summary))
    summary["output_dir"] = str(out)
    return summary


def _f(v: Any, pct: bool = False) -> str:
    if v is None:
        return "—"
    return f"{v * 100:.1f}" if pct else f"{v:.3f}"


def render_report(summary: dict) -> str:
    cfg, mets = summary["config"], summary["metrics"]
    lines = [f"# Evaluation report — {cfg['experiment']}", "",
             f"- Run: `{cfg['run_id']}`  ·  seed {cfg['seed']}  ·  bootstrap {cfg['n_boot']}",
             f"- Datasets: {', '.join(cfg['datasets'])} ({', '.join(cfg['dataset_versions'])}); subsets: {', '.join(cfg['subsets'])}",
             f"- Examples: {cfg['n_examples']} (cached evidence for {cfg['n_cached']}); split label counts: {cfg['split_label_counts']}",
             "- Models: " + "; ".join(f"{k}={v['name']} ({v.get('version')})" for k, v in cfg["models"].items()),
             f"- **Simulated (fake models): {cfg['simulated']}**", "",
             "Metrics on the held-out **test** split (thresholds selected on val). P/R/F1 are on covered "
             "(non-abstained) claims; 'eff.' metrics count abstentions as errors. Positive class for "
             "hallucination detection = CONTRADICTED.", "",
             "| System | Coverage % | Acc (covered) | Acc (eff.) | Macro-F1 [95% CI] | Halluc. P | Halluc. R | Halluc. F1 | FPR | FNR | ECE | Latency ms |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in mets.items():
        o = m["overall"]
        ci = m["confidence_intervals_95"]["macro_f1"]
        lines.append(
            f"| {name} | {_f(o['coverage'], True)} | {_f(o['accuracy_covered'])} | {_f(o['accuracy_effective'])} | "
            f"{_f(o['macro_f1'])} [{_f(ci['low'])}, {_f(ci['high'])}] | {_f(o['hallucination_precision'])} | "
            f"{_f(o['hallucination_recall'])} | {_f(o['hallucination_f1'])} | {_f(o['false_positive_rate'])} | "
            f"{_f(o['false_negative_rate'])} | {_f(m['calibration']['ece'])} | {_f(o['mean_latency_ms'])} |")
    lines += ["", "## Paired comparisons (test split, cluster bootstrap)", "", "| Comparison | Metric | Δ | 95% CI | p |",
              "|---|---|---|---|---|"]
    for k, c in summary["comparisons"].items():
        lines.append(f"| {k} | {c['metric']} | {_f(c['diff'])} | [{_f(c['ci_low'])}, {_f(c['ci_high'])}] | {_f(c['p_value'])} |")
    lines += ["", "## Per-category (test)", ""]
    for name, m in mets.items():
        for cat, o in m["per_category"].items():
            lines.append(f"- {name} / {cat}: n={int(o['n'])}, coverage={_f(o['coverage'], True)}%, "
                         f"acc(covered)={_f(o['accuracy_covered'])}, macro-F1={_f(o['macro_f1'])}")
    lines += ["", "## Confusion matrices (rows = gold, cols = predicted)", ""]
    for name, m in mets.items():
        cm = m["confusion_matrix"]
        lines.append(f"**{name}**: " + "; ".join(f"{g}: " + ", ".join(f"{p}={n}" for p, n in row.items())
                                                 for g, row in cm.items()))
    lines += ["", "## Selected thresholds (val)", ""]
    for name, m in mets.items():
        lines.append(f"- {name}: {m['selected_params']} (val objective {_f(m['val_objective'])})")
    rt = summary["runtime"]
    lines += ["", f"Runtime: evidence collection {rt['evidence_collection_s']} s; peak RSS {rt['peak_rss_mb']} MB.", ""]
    return "\n".join(lines)


def register_run(summary: dict, db) -> list[str]:
    """Store one EvaluationRun per system (+ metric rows) so the dashboard can display it."""
    from app.persistence.orm import EvaluationMetric, EvaluationRun

    cfg = summary["config"]
    ids = []
    for name, m in summary["metrics"].items():
        rid = f"{cfg['run_id']}:{name}"
        run = EvaluationRun(
            run_id=rid, experiment=cfg["experiment"], system=name, dataset="+".join(cfg["datasets"]),
            dataset_version=";".join(cfg["dataset_versions"])[:64], split="test", seed=cfg["seed"],
            simulated=cfg["simulated"], n_examples=int(m["overall"]["n"]),
            config={"selected_params": m["selected_params"], "description": m["description"],
                    "experiment_run": cfg["run_id"], "grid": cfg["grid"], "split_label_counts": cfg["split_label_counts"]},
            model_versions={k: f"{v['name']} {v.get('version') or ''}".strip() for k, v in cfg["models"].items()},
            metrics={**m, "comparisons": {k: v for k, v in summary["comparisons"].items() if k.startswith(name + "_vs")},
                     "runtime": summary["runtime"]},
            output_path=f"experiments/runs/{cfg['run_id']}", started_at=datetime.fromisoformat(summary["started_at"]),
            finished_at=datetime.fromisoformat(summary["finished_at"]))
        db.add(run)
        db.flush()
        o, cis = m["overall"], m["confidence_intervals_95"]
        for key in ("macro_f1", "accuracy_covered", "accuracy_effective", "coverage", "hallucination_f1",
                    "hallucination_precision", "hallucination_recall", "false_positive_rate", "false_negative_rate"):
            ci = cis.get(key, {})
            db.add(EvaluationMetric(run_id=run.id, name=key, category="all", value=o.get(key),
                                    ci_low=ci.get("low"), ci_high=ci.get("high")))
        for cat, co in m["per_category"].items():
            for key in ("macro_f1", "accuracy_covered", "coverage"):
                db.add(EvaluationMetric(run_id=run.id, name=key, category=cat, value=co.get(key)))
        ids.append(rid)
    db.commit()
    return ids
