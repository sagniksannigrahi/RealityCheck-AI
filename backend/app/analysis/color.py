"""Deterministic pixel-level color analysis (independent of any learned model).

Pixels in the central part of a region are binned into 11 basic color names in HSV space.
Limitations: lighting, white balance, shadows and background pixels inside the box all affect the result,
which is why the decision engine only treats this as one evidence source among several.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from app.schemas.evidence import BoundingBox

COLOR_BINS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray"]


@dataclass
class ColorProfile:
    distribution: dict[str, float]
    pixels: int

    @property
    def dominant(self) -> list[tuple[str, float]]:
        return sorted(self.distribution.items(), key=lambda kv: kv[1], reverse=True)

    def fraction(self, colors: set[str]) -> float:
        return float(sum(self.distribution.get(c, 0.0) for c in colors))


def classify_hsv(hsv: np.ndarray) -> np.ndarray:
    """Map an (N,3) OpenCV HSV array (H 0-179, S/V 0-255) to color-bin indices."""
    h = hsv[:, 0].astype(np.int32)
    s = hsv[:, 1].astype(np.int32)
    v = hsv[:, 2].astype(np.int32)
    out = np.full(h.shape, COLOR_BINS.index("gray"), dtype=np.int32)

    black = v < 50
    white = (s < 40) & (v >= 190)
    gray = (s < 40) & (v >= 50) & (v < 190)
    chroma = ~(black | white | gray)

    hue_bins = np.select(
        [
            (h < 8) | (h >= 170),
            (h >= 8) & (h < 20),
            (h >= 20) & (h < 33),
            (h >= 33) & (h < 85),
            (h >= 85) & (h < 130),
            (h >= 130) & (h < 150),
            (h >= 150) & (h < 170),
        ],
        [
            COLOR_BINS.index("red"),
            COLOR_BINS.index("orange"),
            COLOR_BINS.index("yellow"),
            COLOR_BINS.index("green"),
            COLOR_BINS.index("blue"),
            COLOR_BINS.index("purple"),
            COLOR_BINS.index("pink"),
        ],
        default=COLOR_BINS.index("gray"),
    )
    # dark warm hues read as brown
    warm = (h < 33) | (h >= 170)
    brown = chroma & warm & (v < 150) & (s >= 60)
    # pale reds read as pink
    pale_red = chroma & ((h < 8) | (h >= 170)) & (s < 130) & (v >= 170)
    out = np.where(chroma, hue_bins, out)
    out = np.where(brown, COLOR_BINS.index("brown"), out)
    out = np.where(pale_red, COLOR_BINS.index("pink"), out)
    out = np.where(black, COLOR_BINS.index("black"), out)
    out = np.where(white, COLOR_BINS.index("white"), out)
    out = np.where(gray, COLOR_BINS.index("gray"), out)
    return out


def analyze_region(image: Image.Image, box: BoundingBox | None = None, inner: float = 0.2,
                   max_pixels: int = 40_000) -> ColorProfile:
    """Color distribution of the central part of ``box`` (shrunk by ``inner`` on each side)."""
    arr = np.asarray(image.convert("RGB"))
    hgt, wid = arr.shape[:2]
    if box is not None:
        x1, y1, x2, y2 = box.to_pixels(wid, hgt)
        dx, dy = int((x2 - x1) * inner), int((y2 - y1) * inner)
        x1, y1, x2, y2 = x1 + dx, y1 + dy, max(x1 + dx + 1, x2 - dx), max(y1 + dy + 1, y2 - dy)
        arr = arr[y1:y2, x1:x2]
    if arr.size == 0:
        return ColorProfile(distribution={c: 0.0 for c in COLOR_BINS}, pixels=0)
    n = arr.shape[0] * arr.shape[1]
    if n > max_pixels:
        scale = (max_pixels / n) ** 0.5
        arr = cv2.resize(arr, (max(1, int(arr.shape[1] * scale)), max(1, int(arr.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    bins = classify_hsv(hsv)
    counts = np.bincount(bins, minlength=len(COLOR_BINS)).astype(float)
    total = counts.sum()
    dist = {c: round(float(counts[i] / total), 4) for i, c in enumerate(COLOR_BINS)}
    return ColorProfile(distribution=dist, pixels=int(total))
