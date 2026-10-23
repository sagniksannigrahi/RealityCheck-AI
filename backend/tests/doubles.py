"""Deterministic test doubles and synthetic media generators (no ML models, no network)."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.core.imaging import decode_image, image_fingerprint
from app.detection.base import Detection
from app.grounding.base import GroundingCandidate
from app.schemas.evidence import BoundingBox, ModelInfo
from app.services.fake_models import FakeObject, FakeScene, FakeSceneRegistry
from app.vlm.base import VLMJudgment

# ---------------------------------------------------------------------------------------- images

COLORS = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (30, 170, 40), "yellow": (240, 220, 30),
          "white": (245, 245, 245), "black": (10, 10, 10), "gray": (128, 128, 128)}


def make_image(shapes: list[tuple[str, tuple[float, float, float, float]]], size=(640, 480),
               background=(128, 128, 128), blur: bool = False, dark: bool = False) -> bytes:
    """shapes: [(color name, normalized box)] -> PNG bytes."""
    img = Image.new("RGB", size, background)
    d = ImageDraw.Draw(img)
    w, h = size
    for color, (x1, y1, x2, y2) in shapes:
        d.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], fill=COLORS[color])
    # mild sensor noise so flat synthetic graphics are not mistaken for blurry photos
    rng = np.random.default_rng(0)
    noisy = np.clip(np.asarray(img).astype(np.int16) + rng.normal(0, 6, (h, w, 3)).astype(np.int16), 0, 255)
    img = Image.fromarray(noisy.astype(np.uint8))
    if blur:
        arr = cv2.GaussianBlur(np.asarray(img), (51, 51), 25)
        img = Image.fromarray(arr)
    if dark:
        img = Image.fromarray((np.asarray(img) * 0.08).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def decoded(png: bytes):
    """Decode like the API does and attach the fingerprint used by fake models."""
    from app.services.fake_models import FINGERPRINT_KEY

    img = decode_image(png)
    img.info[FINGERPRINT_KEY] = image_fingerprint(img)
    return img


def register_scene(registry: FakeSceneRegistry, png: bytes, scene: FakeScene) -> str:
    fp = image_fingerprint(decode_image(png))
    registry.register(fp, scene)
    return fp


def obj(label: str, box: tuple[float, float, float, float], conf: float = 0.9, gscore: float = 0.6,
        aliases: list[str] | None = None) -> FakeObject:
    return FakeObject(label=label, box=BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]), confidence=conf,
                      grounding_score=gscore, aliases=aliases or [])


# ---------------------------------------------------------------------------------------- blob doubles
# For video tests: decoded frames have compression noise, so scenes cannot be keyed by fingerprint.
# These doubles find saturated color blobs and map colors to labels deterministically.


def _blobs(image: Image.Image, color: str) -> list[BoundingBox]:
    arr = np.asarray(image.convert("RGB")).astype(int)
    target = np.array(COLORS[color])
    mask = (np.abs(arr - target).sum(axis=2) < 90).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    h, w = mask.shape
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area >= 150:
            out.append(BoundingBox.from_pixels(x, y, x + bw, y + bh, width=w, height=h))
    return out


class BlobDetector:
    def __init__(self, color_to_label: dict[str, str]) -> None:
        self.map = color_to_label
        self.info = ModelInfo(component="detector", name="blob-detector-test-double", version="1", demo=True)
        self.vocabulary = {"person", "dog", "cat", "car", "sports ball", "chair"}
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return [Detection(label=lbl, confidence=0.9, box=b) for c, lbl in self.map.items() for b in _blobs(image, c)]


class BlobGrounder:
    def __init__(self, color_to_label: dict[str, str]) -> None:
        self.map = color_to_label
        self.info = ModelInfo(component="grounding", name="blob-grounder-test-double", version="1", demo=True)

    def ground(self, image, phrase):
        from app.claim_extraction.lexicon import canonical_class

        target = canonical_class(phrase.split()[-1]) or phrase
        return [GroundingCandidate(phrase=phrase, score=0.6, box=b) for c, lbl in self.map.items() if lbl == target
                for b in _blobs(image, c)]


class ScriptedVLM:
    """Answers from a dict of question-substring -> P(yes); existence questions follow blob presence."""

    def __init__(self, answers: dict[str, float] | None = None, color_to_label: dict[str, str] | None = None,
                 default: float = 0.5) -> None:
        self.answers = answers or {}
        self.map = color_to_label or {}
        self.default = default
        self.info = ModelInfo(component="vlm_verifier", name="scripted-vlm-test-double", version="1", demo=True)
        self.questions: list[str] = []

    def judge(self, image, question):
        self.questions.append(question)
        q = question.lower()
        for k in sorted(self.answers, key=len, reverse=True):
            if k.lower() in q:
                p = self.answers[k]
                return VLMJudgment(question=question, answer="yes" if p >= 0.5 else "no", p_yes=p)
        if q.startswith("is there") and self.map:
            from app.claim_extraction.lexicon import canonical_class

            present = {lbl for c, lbl in self.map.items() if _blobs(image, c)}
            for lbl in present:
                if lbl in q or any(canonical_class(w) == lbl for w in q.replace("?", "").split()):
                    return VLMJudgment(question=question, answer="yes", p_yes=0.95)
            return VLMJudgment(question=question, answer="no", p_yes=0.05)
        return VLMJudgment(question=question, answer="uncertain", p_yes=self.default)

    def describe(self, image, prompt=None, max_new_tokens=120):
        return "A red person stands in a gray room."


class StaticRegistry:
    """Minimal ModelRegistry stand-in wiring explicit doubles."""

    def __init__(self, detector=None, grounder=None, verifier=None, generator=None, demo=True) -> None:
        self.detector, self.grounder, self.verifier = detector, grounder, verifier
        self.generator = generator or verifier
        self.demo = demo

    def inference(self):
        from contextlib import nullcontext

        return nullcontext()

    def generator_shares_verifier_weights(self) -> bool:
        return self.generator is self.verifier


# ---------------------------------------------------------------------------------------- videos


def make_video(path: Path, frames_spec: list[list[tuple[str, tuple[float, float, float, float]]]], fps: float = 10.0,
               size=(320, 240), repeat: int = 5) -> Path:
    """Each spec entry is a frame layout; each layout is held for ``repeat`` frames."""
    w, h = size
    rng = np.random.default_rng(0)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert vw.isOpened(), "OpenCV VideoWriter could not open mp4v"
    for layout in frames_spec:
        for _ in range(repeat):
            # textured background: horizontal gradient (realistic contrast), gray in hue
            grad = np.linspace(70, 190, w, dtype=np.float32)[None, :, None]
            frame = np.repeat(np.repeat(grad, h, axis=0), 3, axis=2).astype(np.uint8)
            for color, (x1, y1, x2, y2) in layout:
                r, g, b = COLORS[color]
                cv2.rectangle(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (b, g, r), -1)
            noise = rng.normal(0, 6, frame.shape).astype(np.int16)
            vw.write(np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8))
    vw.release()
    return path
