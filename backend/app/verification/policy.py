"""Verification policy: every threshold and switch of the decision engine, in one versioned object.

Raw scores from different models are mapped to *source-specific* reliability tiers (strong / weak) using
the thresholds below. Tiers, not raw confidences, are combined across sources.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from app.schemas.claims import ClaimType
from app.schemas.evidence import EvidenceSource, Verdict


class SourceThresholds(BaseModel):
    detector_strong: float = 0.50
    detector_weak: float = 0.25
    grounding_strong: float = 0.45
    grounding_weak: float = 0.30
    vlm_strong: float = 0.85  # P(yes) >= this -> strong support; <= 1-this -> strong contradiction
    vlm_weak: float = 0.65
    color_support_strong: float = 0.35
    color_support_weak: float = 0.18
    color_contradict_max: float = 0.04
    color_dominant_min: float = 0.45
    motion_moving: float = 0.06  # normalized center displacement per second
    motion_static: float = 0.015
    other_class_conflict_conf: float = 0.50
    other_class_conflict_iou: float = 0.70


class ComponentSwitches(BaseModel):
    detector: bool = True
    grounding: bool = True
    vlm: bool = True
    color: bool = True
    geometry: bool = True
    motion: bool = True
    temporal: bool = True
    vlm_on_crops: bool = True


DEFAULT_SOURCE_WEIGHTS: dict[str, dict[str, float]] = {
    ClaimType.OBJECT_EXISTENCE: {"detector": 1.0, "grounding": 0.8, "vlm": 1.0},
    ClaimType.OBJECT_IDENTITY: {"detector": 0.8, "grounding": 0.5, "vlm": 1.0},
    ClaimType.ATTRIBUTE_COLOR: {"color_analysis": 0.8, "vlm": 1.0},
    ClaimType.ATTRIBUTE_OTHER: {"vlm": 1.0},
    ClaimType.COUNT: {"detector": 1.0, "grounding": 0.5, "vlm": 0.5},
    ClaimType.ACTION: {"vlm": 1.0, "motion": 0.8},
    ClaimType.SPATIAL_RELATION: {"geometry": 1.0, "vlm": 0.8},
    ClaimType.INTERACTION: {"geometry": 0.6, "vlm": 1.0},
    ClaimType.SCENE: {"vlm": 1.0, "grounding": 0.4},
    ClaimType.TEXT_OCR: {"vlm": 1.0},
    ClaimType.TEMPORAL_EVENT: {"temporal": 1.0, "vlm": 0.6},
}

# Claim types for which "searched and not found" is meaningful (weak negative) evidence.
ABSENCE_SENSITIVE = {ClaimType.OBJECT_EXISTENCE, ClaimType.COUNT, ClaimType.OBJECT_IDENTITY}


class VerificationPolicy(BaseModel):
    name: str = "eav-sa"
    version: str = "1.0"
    thresholds: SourceThresholds = Field(default_factory=SourceThresholds)
    components: ComponentSwitches = Field(default_factory=ComponentSwitches)
    source_weights: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {k.value: v for k, v in DEFAULT_SOURCE_WEIGHTS.items()}
    )
    # --- selective abstention ---
    use_abstention: bool = True
    support_threshold: float = 0.35
    contradict_threshold: float = 0.35
    require_strong_for_decision: bool = True
    forced_tie_verdict: Verdict = Verdict.SUPPORTED  # only used when use_abstention=False
    # --- absence handling ---
    absence_weight: float = 0.35
    absence_alone_never_contradicts: bool = True
    poor_quality_blocks_absence_contradiction: bool = True
    # --- conflict analysis ---
    use_disagreement: bool = True
    # --- dependencies ---
    propagate_contradictions: bool = True
    # --- video ---
    video_min_contradicting_frames: int = 2
    video_majority: float = 0.66

    def weights_for(self, claim_type: ClaimType | str) -> dict[str, float]:
        return self.source_weights.get(str(claim_type), {})

    def source_enabled(self, source: EvidenceSource) -> bool:
        c = self.components
        return {
            EvidenceSource.DETECTOR: c.detector,
            EvidenceSource.GROUNDING: c.grounding,
            EvidenceSource.VLM: c.vlm,
            EvidenceSource.COLOR_ANALYSIS: c.color,
            EvidenceSource.GEOMETRY: c.geometry,
            EvidenceSource.MOTION: c.motion,
            EvidenceSource.TEMPORAL: c.temporal,
        }.get(source, True)

    @classmethod
    def load(cls, path: Path | None) -> VerificationPolicy:
        if path is None or not Path(path).exists():
            return cls()
        return cls.model_validate(json.loads(Path(path).read_text()))


class PlattCalibrator(BaseModel):
    """p = sigmoid(a * score + b), fitted on a validation split by the evaluation harness."""

    a: float
    b: float
    fitted_on: str
    n_examples: int
    ece_validation: float | None = None

    def __call__(self, score: float) -> float:
        import math

        z = self.a * score + self.b
        return 1.0 / (1.0 + math.exp(-z))

    @classmethod
    def load(cls, path: Path | None) -> PlattCalibrator | None:
        if path is None or not Path(path).exists():
            return None
        return cls.model_validate(json.loads(Path(path).read_text()))
