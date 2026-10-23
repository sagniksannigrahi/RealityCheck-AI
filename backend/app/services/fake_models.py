"""Deterministic fake model adapters.

Used by (a) the automated test suite and (b) DEMO mode when real models are unavailable.

Fakes never look at pixels to invent detections. They return objects only for images that have a
registered *scene* (keyed by the decoded-pixel fingerprint stored in ``image.info['rc_fingerprint']``,
which PIL carries through ``crop``/``copy``). For any other image they return no detections and an
'uncertain' VLM answer, so unknown uploads in demo mode end up INSUFFICIENT_EVIDENCE rather than
receiving fabricated verdicts. All outputs are labelled ``demo=True`` in model metadata.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from app.claim_extraction.lexicon import COCO_CLASSES, canonical_class
from app.detection.base import Detection
from app.grounding.base import GroundingCandidate
from app.schemas.evidence import BoundingBox, ModelInfo
from app.vlm.base import VLMJudgment

FINGERPRINT_KEY = "rc_fingerprint"


@dataclass
class FakeObject:
    label: str
    box: BoundingBox
    confidence: float = 0.9
    aliases: list[str] = field(default_factory=list)
    grounding_score: float = 0.6


@dataclass
class FakeScene:
    objects: list[FakeObject] = field(default_factory=list)
    caption: str = ""
    # lowercase substring of a VLM question -> P(yes); first match wins (longest pattern first)
    vlm_answers: dict[str, float] = field(default_factory=dict)
    default_p_yes: float = 0.5
    name: str = ""

    def names_for(self, obj: FakeObject) -> set[str]:
        return {obj.label.lower(), *(a.lower() for a in obj.aliases)}


class FakeSceneRegistry:
    def __init__(self) -> None:
        self._scenes: dict[str, FakeScene] = {}
        self._lock = threading.Lock()

    def register(self, fingerprint: str, scene: FakeScene) -> None:
        with self._lock:
            self._scenes[fingerprint] = scene

    def clear(self) -> None:
        with self._lock:
            self._scenes.clear()

    def lookup(self, image: Image.Image) -> FakeScene | None:
        fp = image.info.get(FINGERPRINT_KEY)
        return self._scenes.get(fp) if fp else None

    def load_manifest_dir(self, directory: Path) -> int:
        """Load demo scenes: ``*.json`` files with {"fingerprint": ..., "scene": {...}}."""
        n = 0
        if not directory.exists():
            return 0
        for p in sorted(directory.glob("*.json")):
            data = json.loads(p.read_text())
            objs = [
                FakeObject(label=o["label"], box=BoundingBox(**o["box"]), confidence=o.get("confidence", 0.9),
                           aliases=o.get("aliases", []), grounding_score=o.get("grounding_score", 0.6))
                for o in data["scene"].get("objects", [])
            ]
            scene = FakeScene(objects=objs, caption=data["scene"].get("caption", ""),
                              vlm_answers=data["scene"].get("vlm_answers", {}),
                              default_p_yes=data["scene"].get("default_p_yes", 0.5), name=p.stem)
            self.register(data["fingerprint"], scene)
            n += 1
        return n


SCENES = FakeSceneRegistry()


def _singulars(word: str) -> list[str]:
    out = [word]
    if word.endswith("ies"):
        out.append(word[:-3] + "y")
    if word.endswith("es"):
        out.append(word[:-2])
    if word.endswith("s"):
        out.append(word[:-1])
    irregular = {"people": "person", "men": "man", "women": "woman", "children": "child", "mice": "mouse"}
    if word in irregular:
        out.append(irregular[word])
    return out


def _match(name: str, phrase: str) -> bool:
    phrase = phrase.lower().strip()
    name = name.lower()
    if not phrase:
        return False
    head, *_ = phrase.split()[-1:]
    for cand in [phrase, *[" ".join(phrase.split()[:-1] + [h]) for h in _singulars(head)]]:
        cand = cand.strip()
        if name == cand:
            return True
        cp = canonical_class(cand) or canonical_class(cand.split()[-1])
        if cp is not None and (canonical_class(name) == cp or name == cp):
            return True
    return False


class FakeDetector:
    def __init__(self, registry: FakeSceneRegistry = SCENES) -> None:
        self._reg = registry
        self._info = ModelInfo(component="detector", name="fake-detector", version="1.0", demo=True)

    @property
    def info(self) -> ModelInfo:
        return self._info

    @property
    def vocabulary(self) -> set[str]:
        return set(COCO_CLASSES)

    def detect(self, image: Image.Image) -> list[Detection]:
        scene = self._reg.lookup(image)
        if scene is None:
            return []
        out = []
        for o in scene.objects:
            label = canonical_class(o.label) or o.label
            if label in COCO_CLASSES and o.confidence > 0:
                out.append(Detection(label=label, confidence=o.confidence, box=o.box))
        return out


class FakeGrounder:
    def __init__(self, registry: FakeSceneRegistry = SCENES) -> None:
        self._reg = registry
        self._info = ModelInfo(component="grounding", name="fake-grounder", version="1.0", demo=True)

    @property
    def info(self) -> ModelInfo:
        return self._info

    def ground(self, image: Image.Image, phrase: str) -> list[GroundingCandidate]:
        scene = self._reg.lookup(image)
        if scene is None:
            return []
        out = []
        for o in scene.objects:
            if o.grounding_score > 0 and any(_match(n, phrase) for n in scene.names_for(o)):
                out.append(GroundingCandidate(phrase=phrase, score=o.grounding_score, box=o.box))
        return sorted(out, key=lambda c: c.score, reverse=True)


_EXIST_RE = re.compile(r"^(?:is|are) there (?:a|an|any)?\s*(.+?) in the image\??$", re.IGNORECASE)


class FakeVLM:
    def __init__(self, registry: FakeSceneRegistry = SCENES, role: str = "verifier") -> None:
        self._reg = registry
        self._info = ModelInfo(component=f"vlm_{role}", name="fake-vlm", version="1.0", demo=True)

    @property
    def info(self) -> ModelInfo:
        return self._info

    def judge(self, image: Image.Image, question: str) -> VLMJudgment:
        scene = self._reg.lookup(image)
        q = question.lower().strip()
        if scene is None:
            return VLMJudgment(question=question, answer="uncertain", p_yes=0.5, raw={"fake": "unknown_image"})
        p = None
        for pattern in sorted(scene.vlm_answers, key=len, reverse=True):
            if pattern.lower() in q:
                p = scene.vlm_answers[pattern]
                break
        if p is None:
            m = _EXIST_RE.match(q)
            if m:
                target = m.group(1)
                p = 0.92 if any(_match(n, target) for o in scene.objects for n in scene.names_for(o)) else 0.06
            else:
                p = scene.default_p_yes
        answer = "uncertain" if 0.4 < p < 0.6 else ("yes" if p >= 0.5 else "no")
        return VLMJudgment(question=question, answer=answer, p_yes=p, raw={"fake": "scene"})

    def describe(self, image: Image.Image, prompt: str | None = None, max_new_tokens: int = 120) -> str:
        scene = self._reg.lookup(image)
        if scene is None or not scene.caption:
            return ""
        return scene.caption
