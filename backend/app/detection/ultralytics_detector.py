"""Ultralytics YOLO adapter (default weights: yolo11n.pt, COCO-80 vocabulary, AGPL-3.0)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from app.detection.base import Detection
from app.schemas.evidence import BoundingBox, ModelInfo


class UltralyticsDetector:
    def __init__(self, weights: str = "yolo11n.pt", device: str = "cpu", min_confidence: float = 0.10,
                 imgsz: int = 640) -> None:
        import ultralytics
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._device = device
        self._min_conf = min_confidence
        self._imgsz = imgsz
        names = self._model.names
        self._names = {int(k): v for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
        digest = None
        ckpt = Path(getattr(self._model, "ckpt_path", "") or weights)
        if ckpt.exists():
            digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()[:12]
        self._info = ModelInfo(
            component="detector",
            name=f"ultralytics/{Path(weights).name}",
            version=f"ultralytics-{ultralytics.__version__}" + (f"+sha256:{digest}" if digest else ""),
            device=device,
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    @property
    def vocabulary(self) -> set[str]:
        return set(self._names.values())

    def detect(self, image: Image.Image) -> list[Detection]:
        w, h = image.size
        result = self._model.predict(image, conf=self._min_conf, imgsz=self._imgsz, device=self._device,
                                     verbose=False)[0]
        out: list[Detection] = []
        boxes = result.boxes
        for xyxy, cls, conf in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), strict=True):
            out.append(Detection(label=self._names[int(cls)], confidence=float(conf),
                                 box=BoundingBox.from_pixels(*xyxy, width=w, height=h)))
        return out
