"""Generate-and-verify experiment with CHAIR-style labels (COCO val2014).

1. The generator VLM writes a caption for each sampled image (generate_and_verify mode).
2. Claims are extracted; object-existence claims whose subject maps to a COCO class are labelled from COCO instance
   annotations plus objects named in the reference captions (as in CHAIR): SUPPORTED if present, CONTRADICTED otherwise ("hallucinated mention").
   COCO annotations are incomplete, so some "hallucinated" labels are wrong; results are reported as such.
3. The verifier (EAV-SA, default declared thresholds — no tuning on this data) decides each claim.
4. Reports CHAIR-i / CHAIR-s of the raw captions, hallucination-detection metrics, and CHAIR-i after removing mentions
   the verifier contradicts.

  python scripts/run_chair_experiment.py --images 80 --seed 21
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=80)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()
    os.environ["RC_MODEL_BACKEND"] = "real"
    os.environ.setdefault("RC_DETECTOR_WEIGHTS", str(ROOT / "data/models/yolo11n.pt"))
    from app.claim_extraction.extractor import RuleBasedClaimExtractor
    from app.core.config import reset_settings_cache
    from app.core.imaging import decode_image, image_fingerprint, image_quality

    reset_settings_cache()
    from app.evaluation.datasets import chair, coco_object_mentions, fetch_coco_images
    from app.evaluation.metrics import Prediction, full_report, normalize_pred
    from app.schemas.claims import ClaimType
    from app.services.fake_models import FINGERPRINT_KEY
    from app.services.model_manager import ModelRegistry
    from app.verification.collector import EvidenceCollector, FrameContext
    from app.verification.decision import decide
    from app.verification.policy import VerificationPolicy

    ann = json.loads((ROOT / "datasets/raw/coco/annotations/instances_val2014.json").read_text())
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    gt: dict[int, set[str]] = {}
    for a in ann["annotations"]:
        gt.setdefault(a["image_id"], set()).add(cats[a["category_id"]])
    # As in the original CHAIR (Rohrbach et al., 2018), objects named in the human reference captions also count as
    # present; this reduces (does not remove) label noise from objects missing in the instance annotations.
    caps = json.loads((ROOT / "datasets/raw/coco/annotations/captions_val2014.json").read_text())
    for a in caps["annotations"]:
        if a["image_id"] in gt:
            gt[a["image_id"]] |= coco_object_mentions(a["caption"])
    imgs = [i for i in ann["images"] if i["id"] in gt]
    rng = random.Random(args.seed)
    sample = rng.sample(sorted(imgs, key=lambda i: i["id"]), args.images)
    files = [i["file_name"] for i in sample]
    fetch_coco_images(files, ROOT / "datasets/raw/coco/val2014")

    reg = ModelRegistry(force_demo=False)
    ex = RuleBasedClaimExtractor()
    policy = VerificationPolicy()
    collector = EvidenceCollector(detector=reg.detector, grounder=reg.grounder, vlm=reg.verifier, policy=policy,
                                  inference_guard=reg.inference)
    captions, preds, rows = {}, [], []
    t0 = time.time()
    for k, im in enumerate(sample):
        p = ROOT / "datasets/raw/coco/val2014" / im["file_name"]
        if not p.exists():
            continue
        img = decode_image(p.read_bytes())
        img.info[FINGERPRINT_KEY] = image_fingerprint(img)
        cap = reg.generator.describe(img)
        key = str(im["id"])
        captions[key] = cap
        ctx = FrameContext(image=img, quality=image_quality(img))
        res = ex.extract(cap)
        for c in res.claims:
            if c.claim_type != ClaimType.OBJECT_EXISTENCE or c.negated or not c.subject or not c.subject.canonical_class:
                continue
            gold = "SUPPORTED" if c.subject.canonical_class in gt[im["id"]] else "CONTRADICTED"
            d = decide(c, collector.collect(c, ctx), policy, image_quality=ctx.quality)
            pred = normalize_pred(d.verdict.value)
            preds.append(Prediction(f"{key}-{c.claim_id}", key, gold, pred, None, "generated_existence"))
            rows.append({"image_id": im["id"], "caption": cap, "claim": c.text, "class": c.subject.canonical_class,
                         "gold": gold, "verdict": d.verdict.value, "score": d.score})
        if (k + 1) % 10 == 0:
            print(f"{k + 1}/{len(sample)} images, {len(preds)} claims, {(time.time() - t0) / (k + 1):.1f}s/image", flush=True)

    raw = chair(captions, {str(k): v for k, v in gt.items()})
    # CHAIR after filtering: drop mentions the verifier contradicts
    kept_mentions = hallucinated_kept = 0
    for key, cap in captions.items():
        removed = {r["class"] for r in rows if str(r["image_id"]) == key and r["verdict"] == "CONTRADICTED"}
        objs = coco_object_mentions(cap) - removed
        kept_mentions += len(objs)
        hallucinated_kept += len(objs - gt[int(key)])
    report = full_report(preds, n_boot=1000, seed=args.seed)
    run_id = f"chair-generate-verify-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    out = ROOT / "experiments/runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {"run_id": run_id, "images": len(captions), "seed": args.seed, "policy": "default (untuned)",
                   "models": {c: getattr(reg, c).info.model_dump() for c in ("detector", "grounder", "verifier", "generator")},
                   "note": "Labels = COCO instance annotations + objects named in the 5 reference captions (still incomplete); generator and verifier share SmolVLM weights."},
        "chair_raw": raw,
        "chair_after_filtering": {"chair_i": hallucinated_kept / kept_mentions if kept_mentions else None,
                                  "objects_mentioned": kept_mentions},
        "verification": report, "seconds": round(time.time() - t0, 1),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    with (out / "claims.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    o = report["overall"]

    def f3(x: float | None) -> str:
        return "—" if x is None else f"{x:.3f}"

    md = [f"# Generate-and-verify with CHAIR labels — {run_id}", "", summary["config"]["note"], "",
          f"- Images: {len(captions)}; object-existence claims from generated captions: {int(o['n'])}",
          f"- Raw captions: CHAIR-i = {f3(raw['chair_i'])}, CHAIR-s = {f3(raw['chair_s'])} ({int(raw['objects_mentioned'])} mentions)",
          f"- After removing verifier-contradicted mentions: CHAIR-i = {f3(summary['chair_after_filtering']['chair_i'])} "
          f"({kept_mentions} mentions kept)",
          f"- EAV-SA (default thresholds): coverage {f3(o['coverage'])}, accuracy on covered {f3(o['accuracy_covered'])}, "
          f"hallucination precision {o['hallucination_precision']}, recall {o['hallucination_recall']}, "
          f"effective recall {o['hallucination_recall_effective']}",
          f"- Confusion (rows gold): {report['confusion_matrix']}", f"- Runtime: {summary['seconds']} s"]
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
