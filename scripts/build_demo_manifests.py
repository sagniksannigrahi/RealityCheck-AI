"""Build DEMO-mode scene manifests from COCO val2014 ground-truth annotations.

Demo mode uses deterministic fake models. For these bundled sample images only, the fake detector/grounder
return COCO *ground-truth* boxes and the fake VLM answers existence questions from the same annotations.
This makes the UI demonstrable without ML models; it is NOT model output and is labelled DEMO everywhere.
Only images under licenses permitting redistribution (CC BY 2.0, CC BY-SA 2.0) are bundled.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.imaging import decode_image, image_fingerprint  # noqa: E402

ALLOWED_LICENSES = {4, 5}
DEFAULT_IDS = [39769]  # two cats on a couch (CC BY-SA 2.0)


def main(n_extra: int = 2) -> None:
    ann = json.loads((ROOT / "datasets/raw/coco/annotations/instances_val2014.json").read_text())
    caps = json.loads((ROOT / "datasets/raw/coco/annotations/captions_val2014.json").read_text())
    lic = {lc["id"]: lc for lc in ann["licenses"]}
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    imgs = {i["id"]: i for i in ann["images"]}
    by_img: dict[int, list] = {}
    for a in ann["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    cap_by: dict[int, list[str]] = {}
    for c in caps["annotations"]:
        cap_by.setdefault(c["image_id"], []).append(c["caption"].strip())
    local = ROOT / "datasets/raw/coco/val2014"
    ids = list(DEFAULT_IDS)
    for f in sorted(local.glob("*.jpg")):
        iid = int(f.stem.rsplit("_", 1)[1])
        names = {cats[a["category_id"]] for a in by_img.get(iid, [])}
        if (iid not in ids and imgs[iid]["license"] in ALLOWED_LICENSES and "person" in names
                and 2 <= len(names) <= 5 and len(ids) < 1 + n_extra):
            ids.append(iid)
    out = ROOT / "datasets/demo"
    (out / "images").mkdir(parents=True, exist_ok=True)
    attributions = []
    for iid in ids:
        im = imgs[iid]
        assert im["license"] in ALLOWED_LICENSES, im
        src = local / im["file_name"]
        if not src.exists():
            import urllib.request

            src.write_bytes(urllib.request.urlopen(im["coco_url"], timeout=60).read())  # noqa: S310
        dst = out / "images" / im["file_name"]
        shutil.copyfile(src, dst)
        fp = image_fingerprint(decode_image(dst.read_bytes()))
        w, h = im["width"], im["height"]
        objects = []
        for a in by_img.get(iid, []):
            if a.get("iscrowd") or a["area"] < 0.002 * w * h:
                continue
            x, y, bw, bh = a["bbox"]
            objects.append({"label": cats[a["category_id"]], "box": {"x1": x / w, "y1": y / h, "x2": (x + bw) / w, "y2": (y + bh) / h},
                            "confidence": 0.9, "grounding_score": 0.6})
        scene = {"objects": objects, "caption": cap_by.get(iid, [""])[0], "default_p_yes": 0.5, "vlm_answers": {}}
        (out / f"coco_{iid}.json").write_text(json.dumps({
            "fingerprint": fp, "image": f"images/{im['file_name']}",
            "note": "DEMO scene: boxes are COCO ground-truth annotations, not model output.", "scene": scene}, indent=1))
        attributions.append(f"- `images/{im['file_name']}` — COCO val2014 image {iid}; Flickr: {im['flickr_url']}; "
                            f"license: {lic[im['license']]['name']} ({lic[im['license']]['url']}). "
                            "Annotations: COCO Consortium, CC BY 4.0.")
    (out / "ATTRIBUTION.md").write_text("# Demo image attribution\n\n" + "\n".join(attributions) + "\n")
    print(f"wrote {len(ids)} demo scenes to {out}")


if __name__ == "__main__":
    main()
