"""Run a reproducible evaluation experiment.

Examples
  # POPE (COCO), 3 subsets x 40 images x 6 questions, real local models (CPU):
  python scripts/run_evaluation.py --dataset pope --images-per-subset 40 --backend real --experiment pope-v1 --register

  # COCO-derived counting + left/right claims over 150 images:
  python scripts/run_evaluation.py --dataset coco-count-spatial --coco-images 150 --backend real --experiment coco-cs-v1

  # Harness smoke test with fake models (results are SIMULATED and labelled as such):
  python scripts/run_evaluation.py --dataset jsonl:datasets/synthetic/smoke.jsonl --backend demo --experiment smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="pope | coco-count-spatial | jsonl:<path>")
    ap.add_argument("--subsets", default="adversarial,popular,random")
    ap.add_argument("--images-per-subset", type=int, default=40)
    ap.add_argument("--coco-images", type=int, default=150)
    ap.add_argument("--backend", choices=["real", "demo"], default="real")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--systems", default="all")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--min-coverage", type=float, default=0.6)
    ap.add_argument("--vlm-tiling", action="store_true", help="enable SmolVLM image splitting (slower)")
    ap.add_argument("--register", action="store_true", help="store results in the application database")
    ap.add_argument("--no-fetch", action="store_true", help="do not download missing COCO images")
    args = ap.parse_args()

    os.environ["RC_MODEL_BACKEND"] = args.backend
    if args.vlm_tiling:
        os.environ["RC_VLM_IMAGE_SPLITTING"] = "true"
    os.environ.setdefault("RC_DETECTOR_WEIGHTS", str(ROOT / "data" / "models" / "yolo11n.pt"))

    from app.core.config import reset_settings_cache

    reset_settings_cache()
    from app.evaluation import datasets as ds
    from app.evaluation.harness import SYSTEMS, register_run, run_experiment
    from app.services.model_manager import ModelRegistry

    droot = ROOT / "datasets"
    if args.dataset == "pope":
        examples = ds.load_pope(droot / "raw" / "pope", args.subsets.split(","), args.images_per_subset, args.seed)
    elif args.dataset == "coco-count-spatial":
        inst = droot / "raw" / "coco" / "annotations" / "instances_val2014.json"
        pope_ids = None
        examples = ds.load_coco_count_spatial(inst, pope_ids, args.coco_images, args.seed)
    elif args.dataset.startswith("jsonl:"):
        examples = ds.load_jsonl(ROOT / args.dataset.split(":", 1)[1])
    else:
        ap.error("unknown dataset")
    print(f"{len(examples)} examples from {len({e.image_path for e in examples})} images")

    coco_files = sorted({Path(e.image_path).name for e in examples if e.image_path.startswith("raw/coco/val2014/")})
    if coco_files and not args.no_fetch:
        print(f"ensuring {len(coco_files)} COCO images are cached ...")
        hashes = ds.fetch_coco_images(coco_files, droot / "raw" / "coco" / "val2014")
        missing = [f for f, h in hashes.items() if not h]
        print(f"images ready ({len(missing)} missing)")
        (droot / "raw" / "coco" / "val2014_manifest.json").write_text(json.dumps(hashes, indent=1, sort_keys=True))

    registry = ModelRegistry(force_demo=args.backend == "demo")
    systems = list(SYSTEMS) if args.systems == "all" else args.systems.split(",")
    summary = run_experiment(experiment=args.experiment, examples=examples, registry=registry, datasets_root=droot,
                             out_root=ROOT / "experiments" / "runs", systems=systems, seed=args.seed,
                             n_boot=args.n_boot, min_coverage=args.min_coverage, simulated=args.backend == "demo")
    print(Path(summary["output_dir"]) / "report.md")
    print((Path(summary["output_dir"]) / "report.md").read_text()[:4000])
    if args.register:
        from app.core.config import get_settings
        from app.main import run_migrations
        from app.persistence.db import init_engine, session_scope

        init_engine()
        run_migrations(get_settings().database_url)
        with session_scope() as db:
            ids = register_run(summary, db)
        print(f"registered {len(ids)} evaluation runs")


if __name__ == "__main__":
    main()
