"""Grounding DINO adapter via Hugging Face transformers (default: IDEA-Research/grounding-dino-tiny, Apache-2.0).

One phrase is grounded per forward pass: batching several phrases in one prompt makes the model merge
labels (observed during environment checks), which would break claim-to-region attribution.
"""

from __future__ import annotations

from PIL import Image

from app.grounding.base import GroundingCandidate
from app.schemas.evidence import BoundingBox, ModelInfo


def _nms(cands: list[GroundingCandidate], iou_thr: float = 0.6) -> list[GroundingCandidate]:
    kept: list[GroundingCandidate] = []
    for c in sorted(cands, key=lambda x: x.score, reverse=True):
        if all(c.box.iou(k.box) < iou_thr for k in kept):
            kept.append(c)
    return kept


class GroundingDinoGrounder:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", device: str = "cpu",
                 box_threshold: float = 0.20, text_threshold: float = 0.20, max_candidates: int = 8,
                 cache_dir: str | None = None, revision: str | None = None) -> None:
        import torch
        import transformers
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, revision=revision)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, cache_dir=cache_dir, revision=revision).eval()
        self._device = device
        self._model.to(device)
        self._box_thr = box_threshold
        self._text_thr = text_threshold
        self._max = max_candidates
        self._info = ModelInfo(component="grounding", name=model_id,
                               version=f"transformers-{transformers.__version__}", device=device)

    @property
    def info(self) -> ModelInfo:
        return self._info

    def ground(self, image: Image.Image, phrase: str) -> list[GroundingCandidate]:
        text = phrase.strip().lower().rstrip(".") + "."
        inputs = self._processor(images=image, text=text, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        w, h = image.size
        res = self._processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self._box_thr, text_threshold=self._text_thr,
            target_sizes=[(h, w)],
        )[0]
        cands = [
            GroundingCandidate(phrase=phrase, score=float(s), box=BoundingBox.from_pixels(*b.tolist(), width=w, height=h))
            for s, b in zip(res["scores"], res["boxes"], strict=True)
        ]
        return _nms(cands)[: self._max]
