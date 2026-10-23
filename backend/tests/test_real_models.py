"""Real-model adapter checks (marked slow; skipped unless ML dependencies and weights are available).

Run with:  .venv/bin/pytest -q -m slow
"""

from pathlib import Path

import pytest

from app.services.model_manager import ml_dependencies_available

ROOT = Path(__file__).resolve().parents[2]
CATS = ROOT / "datasets/demo/images/COCO_val2014_000000039769.jpg"
WEIGHTS = ROOT / "data/models/yolo11n.pt"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not ml_dependencies_available(), reason="ML extras not installed"),
    pytest.mark.skipif(not WEIGHTS.exists(), reason="run scripts/download_models.py first"),
]


@pytest.fixture(scope="module")
def image():
    from app.core.imaging import decode_image

    return decode_image(CATS.read_bytes())


def test_yolo_adapter(image):
    from app.detection.ultralytics_detector import UltralyticsDetector

    det = UltralyticsDetector(str(WEIGHTS))
    labels = [d.label for d in det.detect(image) if d.confidence > 0.5]
    assert labels.count("cat") == 2
    assert "dog" in det.vocabulary and "kangaroo" not in det.vocabulary
    assert det.info.version and "sha256" in det.info.version


def test_grounding_dino_adapter(image):
    from app.grounding.grounding_dino import GroundingDinoGrounder

    g = GroundingDinoGrounder()
    cats = g.ground(image, "cat")
    assert cats and cats[0].score > 0.4
    assert all(0 <= c.box.x1 <= c.box.x2 <= 1 for c in cats)


def test_smolvlm_adapter(image):
    from app.vlm.smolvlm import SmolVLMAdapter

    v = SmolVLMAdapter()
    assert v.judge(image, "Is there a cat in the image?").p_yes > 0.8
    assert v.judge(image, "Is there a dog in the image?").p_yes < 0.2
    assert len(v.describe(image, max_new_tokens=30)) > 10
