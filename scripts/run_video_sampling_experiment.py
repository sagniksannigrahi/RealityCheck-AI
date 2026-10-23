"""Video sampling ablation on SYNTHETIC videos (model-independent).

Question: how often does each sampling strategy include at least one frame inside a brief event?
Each synthetic 10 s, 10 fps video shows a static scene; an event (a colored object) appears for D seconds at a random
time. We run the real OpenCV sampler and record whether any sampled frame falls inside the event window.
This measures the sampling component only — not detection, grounding or VLM accuracy.

  python scripts/run_video_sampling_experiment.py --videos 60 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.video.sampling import probe_video, sample_frames  # noqa: E402


def make(path: Path, start: float, dur: float, fps: int = 10, seconds: int = 10, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    w, h = 320, 240
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    grad = np.repeat(np.repeat(np.linspace(70, 190, w, dtype=np.float32)[None, :, None], h, 0), 3, 2)
    for i in range(fps * seconds):
        t = i / fps
        f = grad.copy()
        if start <= t < start + dur:
            cv2.rectangle(f, (100, 60), (220, 200), (40, 40, 220), -1)
        f = np.clip(f + rng.normal(0, 5, f.shape), 0, 255).astype(np.uint8)
        vw.write(f)
    vw.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    durations = [0.3, 0.6, 1.2]
    budgets = [4, 8, 16]
    strategies = ["uniform", "interval", "scene_aware"]
    results: dict = {}
    with tempfile.TemporaryDirectory() as td:
        for vi in range(args.videos):
            dur = durations[vi % len(durations)]
            start = round(rng.uniform(0.5, 9.5 - dur), 1)
            p = Path(td) / f"v{vi}.mp4"
            make(p, start, dur, seed=vi)
            info = probe_video(p)
            for strat in strategies:
                for k in budgets:
                    frames = sample_frames(p, info, strategy=strat, max_frames=k, interval_s=10 / k)
                    hit = any(start <= f.timestamp_s < start + dur for f in frames)
                    key = f"{strat}|{k}|{dur}"
                    r = results.setdefault(key, {"hits": 0, "n": 0, "frames": 0})
                    r["hits"] += int(hit)
                    r["n"] += 1
                    r["frames"] += len(frames)
    rows = []
    for key, r in sorted(results.items()):
        strat, k, dur = key.split("|")
        rows.append({"strategy": strat, "max_frames": int(k), "event_s": float(dur), "event_recall": r["hits"] / r["n"],
                     "videos": r["n"], "mean_frames_used": r["frames"] / r["n"]})
    run_id = f"video-sampling-synthetic-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    out = ROOT / "experiments" / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": run_id, "seed": args.seed, "videos": args.videos, "durations_s": durations, "budgets": budgets,
            "note": "SYNTHETIC videos; measures sampling coverage only (no models)."}
    (out / "metrics.json").write_text(json.dumps({"config": meta, "rows": rows}, indent=2))
    lines = [f"# Video sampling ablation (synthetic) — {run_id}", "", meta["note"], "",
             "| Strategy | Max frames | Event length (s) | Event recall | Frames used (mean) | Videos |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['strategy']} | {r['max_frames']} | {r['event_s']} | {r['event_recall']:.2f} | {r['mean_frames_used']:.1f} | {r['videos']} |"
              for r in rows]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
