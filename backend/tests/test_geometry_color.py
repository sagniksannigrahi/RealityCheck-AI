import io

from PIL import Image

from app.analysis.color import analyze_region
from app.analysis.geometry import evaluate_interaction, evaluate_relation
from app.schemas.evidence import BoundingBox, Stance, Strength
from tests.doubles import make_image


def B(*v):
    return BoundingBox(x1=v[0], y1=v[1], x2=v[2], y2=v[3])


LEFT, RIGHT = B(0.05, 0.4, 0.25, 0.6), B(0.7, 0.4, 0.9, 0.6)


def test_left_right_relations():
    assert evaluate_relation("left_of", LEFT, RIGHT, 640, 480).stance == Stance.SUPPORTS
    r = evaluate_relation("left_of", RIGHT, LEFT, 640, 480)
    assert r.stance == Stance.CONTRADICTS and r.strength == Strength.STRONG
    assert evaluate_relation("right_of", RIGHT, LEFT, 640, 480).stance == Stance.SUPPORTS


def test_above_below_on_inside_near():
    top, bottom = B(0.4, 0.05, 0.6, 0.25), B(0.4, 0.7, 0.6, 0.9)
    assert evaluate_relation("above", top, bottom, 640, 480).stance == Stance.SUPPORTS
    assert evaluate_relation("below", top, bottom, 640, 480).stance == Stance.CONTRADICTS
    cup, table = B(0.45, 0.35, 0.55, 0.5), B(0.2, 0.48, 0.8, 0.9)
    assert evaluate_relation("on", cup, table, 640, 480).stance == Stance.SUPPORTS
    assert evaluate_relation("inside", B(0.3, 0.3, 0.4, 0.4), B(0.2, 0.2, 0.6, 0.6), 640, 480).stance == Stance.SUPPORTS
    assert evaluate_relation("inside", LEFT, RIGHT, 640, 480).stance == Stance.CONTRADICTS
    assert evaluate_relation("near", B(0.1, 0.1, 0.3, 0.3), B(0.3, 0.1, 0.5, 0.3), 640, 480).stance == Stance.SUPPORTS
    assert evaluate_relation("near", B(0.0, 0.0, 0.05, 0.05), B(0.9, 0.9, 0.95, 0.95), 640, 480).stance == Stance.CONTRADICTS


def test_depth_relations_are_not_assessable_from_2d_boxes():
    for rel in ("behind", "in_front_of", "between"):
        assert evaluate_relation(rel, LEFT, RIGHT, 640, 480).stance == Stance.NOT_APPLICABLE


def test_overlapping_boxes_never_establish_holding():
    man, ball = B(0.3, 0.1, 0.6, 0.9), B(0.5, 0.4, 0.6, 0.5)
    r = evaluate_interaction("hold", man, ball, 640, 480)
    assert r.stance == Stance.NEUTRAL  # consistent, but NOT support
    far = evaluate_interaction("hold", B(0.0, 0.0, 0.1, 0.2), B(0.8, 0.8, 0.9, 0.9), 640, 480)
    assert far.stance == Stance.CONTRADICTS and far.strength == Strength.WEAK
    assert evaluate_interaction("watch", man, ball, 640, 480).stance == Stance.NOT_APPLICABLE


def test_wearing_requires_item_within_person_box():
    person, shirt = B(0.3, 0.1, 0.6, 0.9), B(0.35, 0.3, 0.55, 0.5)
    assert evaluate_interaction("wear", person, shirt, 640, 480).stance == Stance.NEUTRAL
    assert evaluate_interaction("wear", person, B(0.8, 0.8, 0.95, 0.95), 640, 480).stance == Stance.CONTRADICTS


def _img(png):
    return Image.open(io.BytesIO(png)).convert("RGB")


def test_color_analysis_identifies_region_color():
    img = _img(make_image([("red", (0.1, 0.1, 0.4, 0.4)), ("blue", (0.6, 0.6, 0.9, 0.9))]))
    red = analyze_region(img, B(0.1, 0.1, 0.4, 0.4))
    blue = analyze_region(img, B(0.6, 0.6, 0.9, 0.9))
    assert red.dominant[0][0] == "red" and red.fraction({"red"}) > 0.9
    assert blue.dominant[0][0] == "blue"
    whole = analyze_region(img)
    assert whole.dominant[0][0] == "gray"
    assert abs(sum(whole.distribution.values()) - 1.0) < 1e-3


def test_color_bins_basic_colors():
    for name in ["green", "yellow", "white", "black"]:
        img = _img(make_image([(name, (0.0, 0.0, 1.0, 1.0))]))
        assert analyze_region(img).dominant[0][0] == name, name
