"""Evidence and verification-result schemas."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    """Axis-aligned box in *normalized* image coordinates (0..1), origin top-left."""

    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _order(self) -> BoundingBox:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("box coordinates must satisfy x1<=x2 and y1<=y2")
        return self

    @classmethod
    def from_pixels(cls, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> BoundingBox:
        def c(v: float) -> float:
            return min(1.0, max(0.0, v))

        return cls(x1=c(x1 / width), y1=c(y1 / height), x2=c(x2 / width), y2=c(y2 / height))

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        return (
            int(round(self.x1 * width)),
            int(round(self.y1 * height)),
            int(round(self.x2 * width)),
            int(round(self.y2 * height)),
        )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def iou(self, other: BoundingBox) -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def union(self, other: BoundingBox) -> BoundingBox:
        return BoundingBox(
            x1=min(self.x1, other.x1), y1=min(self.y1, other.y1), x2=max(self.x2, other.x2), y2=max(self.y2, other.y2)
        )


class EvidenceSource(StrEnum):
    DETECTOR = "detector"
    GROUNDING = "grounding"
    VLM = "vlm"
    COLOR_ANALYSIS = "color_analysis"
    GEOMETRY = "geometry"
    TEMPORAL = "temporal"
    MOTION = "motion"
    CLAIM_ANALYSIS = "claim_analysis"


class Stance(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"  # observed, but neither supports nor contradicts
    ABSENT = "absent"  # searched for and not found (NOT the same as contradiction)
    NOT_APPLICABLE = "not_applicable"  # source cannot assess this claim (e.g. class not in detector vocab)


class Strength(StrEnum):
    """Source-specific reliability tier. Tiers come from per-source thresholds, never from comparing
    raw confidences across models."""

    STRONG = "strong"
    WEAK = "weak"
    NONE = "none"


class EvidenceItem(BaseModel):
    evidence_id: str
    claim_id: str | None = None
    source: EvidenceSource
    model_name: str | None = None
    model_version: str | None = None
    stance: Stance
    strength: Strength = Strength.NONE
    raw_score: float | None = Field(
        default=None, description="Source-native score (detector conf, grounding score, VLM P(yes)). Not comparable."
    )
    box: BoundingBox | None = None
    label: str | None = None
    frame_index: int | None = None
    timestamp_s: float | None = None
    crop_key: str | None = Field(default=None, description="Opaque storage key for an evidence crop.")
    description: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class Verdict(StrEnum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    UNVERIFIABLE = "UNVERIFIABLE"  # subjective / non-visual / speculative claims; never forced to binary


class EvidenceCase(StrEnum):
    POSITIVE = "positive"  # A: directly observed
    CONTRADICTORY = "contradictory"  # B: reliable conflicting observation
    MISSING = "missing"  # C: could not locate sufficient evidence
    AMBIGUOUS = "ambiguous"  # D: several interpretations plausible
    DISAGREEMENT = "disagreement"  # E: independent components incompatible
    NONVISUAL = "nonvisual"


class EvidenceQuality(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class ModelInfo(BaseModel):
    component: str
    name: str
    version: str | None = None
    device: str | None = None
    demo: bool = False
    external: bool = False


class ClaimVerification(BaseModel):
    claim_id: str
    claim_text: str
    claim_type: str
    verdict: Verdict
    evidence_case: EvidenceCase
    verification_score: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="Signed evidence-agreement score; NOT a probability."
    )
    calibrated_probability: float | None = Field(
        default=None, description="P(claim supported); present only if a fitted, evaluated calibrator is loaded."
    )
    evidence_quality: EvidenceQuality
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    considered_evidence: list[str] = Field(default_factory=list)
    explanation: str
    uncertainty_reasons: list[str] = Field(default_factory=list)
    review_recommended: bool = False
    models: list[ModelInfo] = Field(default_factory=list)
    decision_trace: list[str] = Field(default_factory=list)
    duration_ms: float = 0.0
