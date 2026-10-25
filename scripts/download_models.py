"""Download model weights used by the real backend (≈1.7 GB total) into the configured caches.

  YOLO11n (Ultralytics, AGPL-3.0)                     ~5.4 MB   -> RC_DETECTOR_WEIGHTS
  IDEA-Research/grounding-dino-tiny (Apache-2.0)       ~690 MB  -> HF cache
  HuggingFaceTB/SmolVLM-500M-Instruct (Apache-2.0)     ~1.0 GB  -> HF cache
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--if-missing", action="store_true", help="exit quietly when weights already exist")
    args = ap.parse_args()
    from app.core.config import get_settings
    from app.services.model_manager import ml_dependencies_available

    s = get_settings()
    if not ml_dependencies_available():
        print("ML dependencies not installed; nothing to download (demo mode only).")
        return
    weights = Path(s.detector_weights)
    if not weights.is_absolute():
        weights = ROOT / weights
    if not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        from ultralytics import YOLO

        cwd = os.getcwd()
        os.chdir(weights.parent)
        try:
            YOLO(weights.name)  # downloads the official asset into the current directory
        finally:
            os.chdir(cwd)
        if not weights.exists() and Path(weights.name).exists():
            shutil.move(weights.name, weights)
        print(f"YOLO weights: {weights}")
    elif not args.if_missing:
        print(f"YOLO weights present: {weights}")
    from huggingface_hub import snapshot_download

    for repo in {s.grounding_model_id, s.vlm_model_id, s.generator_model_id}:
        rev = s.revision_for(repo)
        path = snapshot_download(repo, revision=rev, cache_dir=str(s.model_cache_dir) if s.model_cache_dir else None)
        print(f"{repo}@{rev or 'latest'}: {path}")


if __name__ == "__main__":
    main()
