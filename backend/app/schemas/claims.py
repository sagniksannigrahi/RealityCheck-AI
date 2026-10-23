"""Atomic claim schemas.

An atomic claim is the smallest independently verifiable statement extracted from a description.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class ClaimType(StrEnum):
    OBJECT_EXISTENCE = "object_existence"
    OBJECT_IDENTITY = "object_identity"
    ATTRIBUTE_COLOR = "attribute_color"
    ATTRIBUTE_OTHER = "attribute_other"
    COUNT = "count"
    ACTION = "action"
    SPATIAL_RELATION = "spatial_relation"
    INTERACTION = "interaction"
    SCENE = "scene"
    TEXT_OCR = "text_ocr"
    TEMPORAL_EVENT = "temporal_event"
    NONVISUAL = "nonvisual"


class EvidenceRequirement(StrEnum):
    """What kind of visual evidence would be needed to verify a claim."""

    LOCALIZE_SUBJECT = "localize_subject"
    LOCALIZE_OBJECT = "localize_object"
    CLASS_DETECTION = "class_detection"
    REGION_COLOR = "region_color"
    REGION_APPEARANCE = "region_appearance"
    INSTANCE_COUNT = "instance_count"
    PAIRWISE_GEOMETRY = "pairwise_geometry"
    INTERACTION_JUDGMENT = "interaction_judgment"
    ACTION_JUDGMENT = "action_judgment"
    SCENE_JUDGMENT = "scene_judgment"
    TEXT_READING = "text_reading"
    MULTI_FRAME = "multi_frame"
    MOTION = "motion"


class CountComparator(StrEnum):
    EXACT = "exact"
    AT_LEAST = "at_least"


class EntityRef(BaseModel):
    """A noun phrase referenced by one or more claims."""

    entity_id: str
    head: str = Field(description="Lemma of the head noun as written, e.g. 'dog', 'man', 'shirt'.")
    phrase: str = Field(description="Short grounding phrase, e.g. 'dog' or 'red shirt'.")
    canonical_class: str | None = Field(
        default=None, description="Normalized object category (COCO-style name) if known, e.g. 'person'."
    )
    category: str | None = Field(default=None, description="Broad category, e.g. 'animal', 'person'.")
    is_scene: bool = False
    plural: bool = False


class SourceSpan(BaseModel):
    start: int
    end: int


class AtomicClaim(BaseModel):
    claim_id: str
    text: str = Field(max_length=500)
    claim_type: ClaimType
    subject: EntityRef | None = None
    predicate: str | None = Field(default=None, description="Verb lemma, relation or attribute name.")
    attribute: str | None = Field(default=None, description="Attribute value, e.g. 'red'.")
    object: EntityRef | None = None
    spatial_relation: str | None = None
    temporal_scope: str | None = Field(
        default=None, description="'any_frame', 'all_frames', 'transition' or None for images."
    )
    count: int | None = Field(default=None, ge=0, le=1000)
    count_comparator: CountComparator | None = None
    negated: bool = False
    hedged: bool = False
    verifiable: bool = True
    unverifiable_reason: str | None = None
    required_evidence: list[EvidenceRequirement] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    source_span: SourceSpan | None = None
    sentence_index: int = 0
    flags: list[str] = Field(default_factory=list, description="e.g. 'instruction_like_text', 'ambiguous'.")

    @field_validator("text")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("claim text must not be empty")
        return v


class ClaimExtractionResult(BaseModel):
    description: str
    claims: list[AtomicClaim]
    entities: list[EntityRef]
    extractor: str
    extractor_version: str
    warnings: list[str] = Field(default_factory=list)
    truncated: bool = False
