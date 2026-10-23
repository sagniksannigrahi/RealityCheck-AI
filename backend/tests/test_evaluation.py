"""Evaluation framework tests. Harness runs here use fake models and are SIMULATED by construction."""

import json

import pytest

from app.evaluation.datasets import (
    EvalExample,
    chair,
    coco_object_mentions,
    existence_claim,
    load_pope,
    split_of,
)
from app.evaluation.harness import SYSTEMS, register_run, retier, run_experiment
from app.evaluation.metrics import (
    Prediction,
    calibration,
    cluster_bootstrap,
    confusion,
    core_metrics,
    paired_bootstrap_diff,
)
from app.schemas.claims import ClaimType
from app.schemas.evidence import EvidenceItem, EvidenceSource, Stance, Strength
from app.services.fake_models import FakeScene
from app.services.model_manager import ModelRegistry
from app.verification.policy import SourceThresholds
from tests.doubles import make_image, obj, register_scene


def P(i, gold, pred, prob=None, cluster=None, cat="c"):
    return Prediction(str(i), cluster or str(i), gold, pred, prob, cat)


def test_core_metrics_hand_computed():
    preds = [P(1, "SUPPORTED", "SUPPORTED"), P(2, "SUPPORTED", "CONTRADICTED"), P(3, "CONTRADICTED", "CONTRADICTED"),
             P(4, "CONTRADICTED", "ABSTAIN"), P(5, "CONTRADICTED", "SUPPORTED"), P(6, "SUPPORTED", "ABSTAIN")]
    m = core_metrics(preds)
    assert m["coverage"] == pytest.approx(4 / 6)
    assert m["accuracy_covered"] == pytest.approx(2 / 4) and m["accuracy_effective"] == pytest.approx(2 / 6)
    # hallucination (CONTRADICTED) on covered: tp=1 (3), fp=1 (2), fn=1 (5)
    assert m["hallucination_precision"] == pytest.approx(0.5) and m["hallucination_recall"] == pytest.approx(0.5)
    assert m["hallucination_recall_effective"] == pytest.approx(1 / 3)
    assert m["false_positive_rate"] == pytest.approx(0.5) and m["false_negative_rate"] == pytest.approx(0.5)
    assert m["macro_f1"] == pytest.approx(0.5)
    cm = confusion(preds)
    assert cm["CONTRADICTED"] == {"SUPPORTED": 1, "CONTRADICTED": 1, "ABSTAIN": 1}


def test_calibration_metrics():
    perfect = [P(i, "SUPPORTED" if i % 2 else "CONTRADICTED", "SUPPORTED", 1.0 if i % 2 else 0.0) for i in range(10)]
    c = calibration(perfect)
    assert c["ece"] == pytest.approx(0.0) and c["brier"] == pytest.approx(0.0)
    bad = [P(i, "CONTRADICTED", "SUPPORTED", 0.9) for i in range(10)]
    assert calibration(bad)["ece"] == pytest.approx(0.9)
    assert calibration([P(1, "SUPPORTED", "SUPPORTED")])["ece"] is None


def test_bootstrap_intervals_contain_point_estimate():
    preds = [P(i, "SUPPORTED" if i % 3 else "CONTRADICTED", "SUPPORTED" if i % 4 else "CONTRADICTED", cluster=str(i // 2))
             for i in range(60)]
    lo, hi = cluster_bootstrap(preds, lambda s: core_metrics(s)["accuracy_covered"], n_boot=300, seed=1)
    point = core_metrics(preds)["accuracy_covered"]
    assert lo <= point <= hi
    d = paired_bootstrap_diff(preds, preds, lambda s: core_metrics(s)["accuracy_covered"], n_boot=200)
    assert d["diff"] == 0


def test_split_is_deterministic_and_image_level():
    assert split_of("123") == split_of("123")
    vals = [split_of(str(i)) for i in range(2000)]
    assert 0.25 < vals.count("val") / len(vals) < 0.35


def test_chair_metric():
    assert coco_object_mentions("Two dogs chase a frisbee near a parked car.") == {"dog", "frisbee", "car"}
    r = chair({"a": "A dog and a cat on a couch.", "b": "A man rides a bike."},
              {"a": {"dog", "couch"}, "b": {"person", "bicycle"}})
    assert r["chair_i"] == pytest.approx(1 / 5) and r["chair_s"] == pytest.approx(0.5)


def test_pope_loader(tmp_path):
    rows = [{"question_id": i, "image": f"COCO_val2014_00000000000{i % 2}.jpg",
             "text": f"Is there a {'dog' if i % 2 else 'dining table'} in the image?", "label": "yes" if i % 2 else "no"}
            for i in range(1, 7)]
    (tmp_path / "coco_pope_random.json").write_text("\n".join(json.dumps(r) for r in rows))
    ex = load_pope(tmp_path, ["random"], images_per_subset=5, seed=1)
    assert len(ex) == 6
    t = {e.metadata["object"]: e for e in ex}
    assert t["dining table"].claim.subject.canonical_class == "dining table"
    assert t["dog"].gold == "SUPPORTED" and t["dining table"].gold == "CONTRADICTED"
    assert t["dining table"].original_label == "no"


def test_retier_uses_raw_scores():
    it = EvidenceItem(evidence_id="x", source=EvidenceSource.VLM, stance=Stance.NEUTRAL, raw_score=0.8)
    out = retier([it], SourceThresholds(vlm_strong=0.75, vlm_weak=0.6), ClaimType.OBJECT_EXISTENCE)
    assert out[0].stance == Stance.SUPPORTS and out[0].strength == Strength.STRONG
    assert it.stance == Stance.NEUTRAL  # original untouched
    det = EvidenceItem(evidence_id="d", source=EvidenceSource.DETECTOR, stance=Stance.ABSENT, raw_score=0.2)
    out = retier([det], SourceThresholds(detector_weak=0.15, detector_strong=0.5), ClaimType.OBJECT_EXISTENCE)
    assert out[0].stance == Stance.SUPPORTS and out[0].strength == Strength.WEAK


def _synthetic_examples(tmp_path, scenes, n_images=24):
    """Synthetic existence claims over generated images; fake models answer correctly most of the time."""
    exs = []
    root = tmp_path / "ds"
    (root / "imgs").mkdir(parents=True)
    for i in range(n_images):
        box = (0.1 + (i % 5) * 0.1, 0.2, 0.3 + (i % 5) * 0.1, 0.6)
        png = make_image([("red", box)], size=(320 + i, 240))
        (root / "imgs" / f"{i}.png").write_bytes(png)
        noisy_vlm = {"is there a dog": 0.3} if i % 6 == 0 else {}  # some VLM errors
        register_scene(scenes, png, FakeScene(objects=[obj("dog", box, conf=0.4 + (i % 5) * 0.1)],
                                              vlm_answers=noisy_vlm))
        for j, (name, gold) in enumerate([("dog", "SUPPORTED"), ("cat", "CONTRADICTED"), ("kangaroo", "CONTRADICTED")]):
            exs.append(EvalExample(example_id=f"s{i}-{j}", dataset="synthetic", dataset_version="test", image_id=str(i),
                                   image_path=f"imgs/{i}.png", claim=existence_claim(name), gold=gold,
                                   original_label=gold.lower(), category="object_existence", subset="synthetic"))
    return exs, root


def test_harness_end_to_end_simulated(tmp_path, scenes, client):
    exs, root = _synthetic_examples(tmp_path, scenes)
    reg = ModelRegistry(force_demo=True)
    summary = run_experiment(experiment="unit-sim", examples=exs, registry=reg, datasets_root=root,
                             out_root=tmp_path / "experiments" / "runs", systems=list(SYSTEMS), seed=3, n_boot=50,
                             simulated=True, log=lambda *_: None)
    out = tmp_path / "experiments" / "runs" / summary["config"]["run_id"]
    for f in ("config.json", "metrics.json", "report.md", "predictions_eav_sa.jsonl"):
        assert (out / f).exists(), f
    assert summary["config"]["simulated"] is True and "Simulated (fake models): True" in (out / "report.md").read_text()
    # no image appears in both splits
    splits = {}
    for e in exs:
        splits.setdefault(e.image_id, set()).add(e.metadata["split"])
    assert all(len(s) == 1 for s in splits.values())
    m = summary["metrics"]
    for name, r in m.items():
        o = r["overall"]
        assert o["n"] > 0
        assert o["coverage"] is None or 0 <= o["coverage"] <= 1
    # detector rules must abstain on out-of-vocabulary objects (kangaroo)
    det = m["detector_rules"]["per_category"]["object_existence"]
    assert det["coverage"] < 1.0
    # abstaining method covers less than its forced counterpart
    assert m["eav_sa"]["overall"]["coverage"] <= m["full"]["overall"]["coverage"]
    assert "full_vs_vlm_only" in summary["comparisons"]
    # cache makes a second run reuse observations
    cache_files = list((tmp_path / "experiments" / "cache").glob("*.jsonl"))
    assert cache_files and sum(1 for _ in cache_files[0].open()) == len(exs)

    from app.persistence.db import session_scope

    with session_scope() as db:
        ids = register_run(summary, db)
    assert len(ids) == len(SYSTEMS)
    lst = client.get("/api/v1/evaluations", params={"experiment": "unit-sim"}).json()
    assert lst["total"] == len(SYSTEMS) and all(r["simulated"] for r in lst["items"])
    one = client.get(f"/api/v1/evaluations/{ids[0]}").json()
    assert one["metric_rows"] and one["metrics"]["confusion_matrix"]
