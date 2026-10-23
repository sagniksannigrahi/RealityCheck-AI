"""Video probing, validation and frame sampling with OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class VideoDecodeError(ValueError):
    pass


@dataclass
class VideoInfo:
    fps: float
    frame_count: int
    duration_s: float
    width: int
    height: int
    fourcc: str


@dataclass
class SampledFrame:
    index: int
    timestamp_s: float
    image: Image.Image
    reason: str  # first | uniform | interval | scene_change
    scene_change_score: float | None = None


def probe_video(path: Path | str) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoDecodeError("Video could not be opened by the decoder.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        code = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        fourcc = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)).strip("\x00") if code else ""
        ok, _ = cap.read()
        if not ok:
            raise VideoDecodeError("Video contains no decodable frames.")
        if fps <= 0 or fps > 1000:
            raise VideoDecodeError(f"Implausible frame rate: {fps}.")
        if n <= 0:
            # some containers do not report frame count: count by grabbing (bounded)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            n = 0
            while cap.grab():
                n += 1
                if n > 200_000:
                    break
        if w <= 0 or h <= 0:
            raise VideoDecodeError("Video has invalid dimensions.")
        return VideoInfo(fps=fps, frame_count=n, duration_s=n / fps, width=w, height=h, fourcc=fourcc)
    finally:
        cap.release()


def validate_video(info: VideoInfo, *, max_seconds: float, max_side: int) -> None:
    if info.duration_s > max_seconds:
        raise VideoDecodeError(f"Video is {info.duration_s:.1f}s long; the limit is {max_seconds:.0f}s.")
    if max(info.width, info.height) > max_side:
        raise VideoDecodeError(f"Video resolution {info.width}x{info.height} exceeds the {max_side}px limit.")
    if info.frame_count < 1:
        raise VideoDecodeError("Video has no frames.")


def _hist(frame_bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h


def _to_pil(frame_bgr: np.ndarray, max_side: int) -> Image.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    return img


def scene_change_scores(path: Path | str, info: VideoInfo, max_scan: int = 400) -> list[tuple[int, float]]:
    """(frame index, histogram distance to previous scanned frame) for a bounded scan of the video."""
    step = max(1, info.frame_count // max_scan)
    cap = cv2.VideoCapture(str(path))
    out: list[tuple[int, float]] = []
    prev = None
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    h = _hist(frame)
                    d = 0.0 if prev is None else float(cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA))
                    out.append((idx, d))
                    prev = h
            idx += 1
    finally:
        cap.release()
    return out


def select_indices(info: VideoInfo, strategy: str, max_frames: int, interval_s: float,
                   scene_threshold: float, path: Path | str | None = None) -> dict[int, tuple[str, float | None]]:
    n = info.frame_count
    max_frames = max(1, min(max_frames, n))
    chosen: dict[int, tuple[str, float | None]] = {0: ("first", None)}
    if strategy == "interval":
        step = max(1, int(round(interval_s * info.fps)))
        for i in range(0, n, step):
            if len(chosen) >= max_frames:
                break
            chosen.setdefault(i, ("interval", None))
    else:
        if strategy == "scene_aware" and path is not None:
            scores = scene_change_scores(path, info)
            cuts = sorted([(i, s) for i, s in scores if s >= scene_threshold], key=lambda x: -x[1])
            budget = max(1, max_frames // 2)
            min_gap = max(1, int(0.25 * info.fps))
            for i, s in cuts:
                if len([v for v in chosen.values() if v[0] == "scene_change"]) >= budget:
                    break
                if all(abs(i - j) >= min_gap for j in chosen):
                    chosen[i] = ("scene_change", round(s, 4))
        remaining = max_frames - len(chosen)
        if remaining > 0:
            for i in np.linspace(0, n - 1, num=remaining + 2).round().astype(int)[1:]:
                if len(chosen) >= max_frames:
                    break
                chosen.setdefault(int(i), ("uniform", None))
    return dict(sorted(chosen.items()))


def sample_frames(path: Path | str, info: VideoInfo, *, strategy: str = "scene_aware", max_frames: int = 16,
                  interval_s: float = 1.0, scene_threshold: float = 0.35, max_side: int = 1280) -> list[SampledFrame]:
    wanted = select_indices(info, strategy, max_frames, interval_s, scene_threshold, path)
    cap = cv2.VideoCapture(str(path))
    out: list[SampledFrame] = []
    idx = 0
    last = max(wanted) if wanted else -1
    try:
        while idx <= last:
            ok = cap.grab()
            if not ok:
                break
            if idx in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    reason, score = wanted[idx]
                    out.append(SampledFrame(index=idx, timestamp_s=round(idx / info.fps, 3),
                                            image=_to_pil(frame, max_side), reason=reason,
                                            scene_change_score=score))
            idx += 1
    finally:
        cap.release()
    if not out:
        raise VideoDecodeError("No frames could be decoded at the sampled positions.")
    return out


def coverage(frames: list[SampledFrame], duration_s: float) -> dict[str, float]:
    ts = [0.0] + [f.timestamp_s for f in frames] + [duration_s]
    gaps = [b - a for a, b in zip(ts, ts[1:], strict=False)]
    return {"max_gap_s": round(max(gaps), 3) if gaps else 0.0, "frames": float(len(frames)),
            "duration_s": round(duration_s, 3)}
