"""Open-vocabulary grounding interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from app.schemas.evidence import BoundingBox, ModelInfo


@dataclass(frozen=True)
class GroundingCandidate:
    phrase: str
    score: float
    box: BoundingBox


class Grounder(Protocol):
    @property
    def info(self) -> ModelInfo: ...

    def ground(self, image: Image.Image, phrase: str) -> list[GroundingCandidate]:
        """Return candidate regions for one phrase, best first. May be empty."""
        ...
