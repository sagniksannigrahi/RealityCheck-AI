"""Object detector interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from app.schemas.evidence import BoundingBox, ModelInfo


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: BoundingBox


class ObjectDetector(Protocol):
    """Closed-vocabulary detector. ``vocabulary`` lists every class the detector can report;
    a class outside the vocabulary must never be treated as 'searched for and not found'."""

    @property
    def info(self) -> ModelInfo: ...

    @property
    def vocabulary(self) -> set[str]: ...

    def detect(self, image: Image.Image) -> list[Detection]: ...
