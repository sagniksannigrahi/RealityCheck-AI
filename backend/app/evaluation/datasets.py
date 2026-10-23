"""Dataset adapters producing a normalized evaluation format.

Every example keeps its *original* label and dataset-specific semantics in ``original_label`` / ``metadata``;
``gold`` is our mapping onto SUPPORTED / CONTRADICTED / AMBIGUOUS, documented per adapter in
docs/EVALUATION.md. Datasets do not all measure the same task; results are reported per dataset.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.claim_extraction.extractor import pluralize
from app.claim_extraction.lexicon import SYNONYMS, canonical_class, category_of
from app.schemas.claims import AtomicClaim, ClaimType, CountComparator, EntityRef, EvidenceRequirement


@dataclass
class EvalExample:
    example_id: str
    dataset: str
    dataset_version: str
    image_id: str
    image_path: str  # relative to the datasets root
    claim: AtomicClaim
    gold: str  # SUPPORTED | CONTRADICTED | AMBIGUOUS
    original_label: str
    category: str
    subset: str = ""
    ambiguous: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["claim"] = self.claim.model_dump(mode="json")
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> EvalExample:
        d = dict(d)
        d["claim"] = AtomicClaim.model_validate(d["claim"])
        return cls(**d)


def split_of(image_id: str, val_fraction: float = 0.3, salt: str = "rc-split-v1") -> str:
    """Deterministic image-level split (prevents leakage of an image across val/test)."""
    h = int(hashlib.sha256(f"{salt}:{image_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_fraction else "test"


def _entity(name: str, eid: str = "e1", plural: bool = False) -> EntityRef:
    canon = canonical_class(name)
    return EntityRef(entity_id=eid, head=name.split()[-1], phrase=name, canonical_class=canon,
                     category=category_of(canon), plural=plural)


def existence_claim(name: str, cid: str = "c1") -> AtomicClaim:
    art = "an" if name[0] in "aeiou" else "a"
    req = [EvidenceRequirement.LOCALIZE_SUBJECT] + ([EvidenceRequirement.CLASS_DETECTION] if canonical_class(name) else [])
    return AtomicClaim(claim_id=cid, text=f"{art.capitalize()} {name} is visible.", claim_type=ClaimType.OBJECT_EXISTENCE,
                       subject=_entity(name), predicate="exists", required_evidence=req)


# ------------------------------------------------------------------------------------------- POPE

_POPE_Q = re.compile(r"^is there (?:a|an) (.+?) in the im(?:a)?ge\??$", re.IGNORECASE)


def load_pope(raw_dir: Path, subsets: list[str], images_per_subset: int, seed: int,
              images_rel_dir: str = "raw/coco/val2014") -> list[EvalExample]:
    """POPE (Li et al., 2023; MIT license). Labels: 'yes' = object annotated in COCO -> SUPPORTED;
    'no' = object sampled from objects *not* annotated (random/popular/adversarial strategies) -> CONTRADICTED.
    Note: negatives inherit COCO's annotation completeness; an unannotated object might still be visible."""
    examples: list[EvalExample] = []
    for subset in subsets:
        rows = [json.loads(line) for line in (raw_dir / f"coco_pope_{subset}.json").read_text().splitlines() if line.strip()]
        by_img: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_img[r["image"]].append(r)
        imgs = sorted(by_img)
        rng = random.Random(f"{seed}:{subset}")
        chosen = rng.sample(imgs, min(images_per_subset, len(imgs)))
        for img in sorted(chosen):
            for r in by_img[img]:
                m = _POPE_Q.match(r["text"].strip())
                if not m:
                    continue
                obj = m.group(1).strip().lower()
                image_id = img.rsplit("_", 1)[-1].split(".")[0].lstrip("0") or "0"
                examples.append(EvalExample(
                    example_id=f"pope-{subset}-{r['question_id']}", dataset="pope-coco", dataset_version="RUCAIBox/POPE@main",
                    image_id=image_id, image_path=f"{images_rel_dir}/{img}", claim=existence_claim(obj),
                    gold="SUPPORTED" if r["label"] == "yes" else "CONTRADICTED", original_label=r["label"],
                    category=ClaimType.OBJECT_EXISTENCE.value, subset=subset,
                    metadata={"question": r["text"], "object": obj, "coco_file": img}))
    return examples


# ------------------------------------------------------------------------------------------- COCO-derived


def load_coco_count_spatial(instances_json: Path, image_ids: list[int] | None, n_images: int, seed: int,
                            images_rel_dir: str = "raw/coco/val2014") -> list[EvalExample]:
    """Counting and left/right claims derived from COCO instance annotations (CC BY 4.0).

    - Count: for a class with 1-6 non-crowd instances, "There are N X" is SUPPORTED and "There are N+1 X" is
      CONTRADICTED. Images where the class has tiny (<0.5% area) or crowd instances are marked AMBIGUOUS-subset
      (gold kept, ``ambiguous=True``) because annotation completeness is uncertain.
    - Spatial: two classes with exactly one instance each and clear horizontal separation; the true relation is
      SUPPORTED and the swapped one CONTRADICTED.
    These are synthetic claims over real images; they test counting/geometry, not natural caption errors.
    """
    data = json.loads(instances_json.read_text())
    cats = {c["id"]: c["name"] for c in data["categories"]}
    imgs = {im["id"]: im for im in data["images"]}
    anns: dict[int, list[dict]] = defaultdict(list)
    for a in data["annotations"]:
        anns[a["image_id"]].append(a)
    pool = sorted(image_ids if image_ids is not None else imgs)
    rng = random.Random(seed)
    rng.shuffle(pool)
    out: list[EvalExample] = []
    used = 0
    for iid in pool:
        if used >= n_images:
            break
        im = imgs.get(iid)
        if im is None:
            continue
        area = im["width"] * im["height"]
        by_cls: dict[str, list[dict]] = defaultdict(list)
        for a in anns[iid]:
            by_cls[cats[a["category_id"]]].append(a)
        made = False
        fname = im["file_name"]
        # counting
        cands = [c for c, lst in by_cls.items() if 1 <= len(lst) <= 6 and c != "person" or (c == "person" and 1 <= len(lst) <= 4)]
        if cands:
            c = rng.choice(sorted(cands))
            lst = by_cls[c]
            n = len([a for a in lst if not a.get("iscrowd")])
            amb = any(a.get("iscrowd") or a["area"] < 0.005 * area for a in lst)
            for k, gold in ((n, "SUPPORTED"), (n + 1, "CONTRADICTED")):
                ent = _entity(c, plural=k != 1)
                text = f"There is exactly one {c}." if k == 1 else f"There are {k} {pluralize(c)}."
                claim = AtomicClaim(claim_id="c1", text=text, claim_type=ClaimType.COUNT, subject=ent, predicate="count",
                                    count=k, count_comparator=CountComparator.EXACT,
                                    required_evidence=[EvidenceRequirement.INSTANCE_COUNT])
                out.append(EvalExample(
                    example_id=f"coco-count-{iid}-{c.replace(' ', '_')}-{k}", dataset="coco-count-spatial",
                    dataset_version="instances_val2014", image_id=str(iid), image_path=f"{images_rel_dir}/{fname}",
                    claim=claim, gold=gold, original_label=f"annotated_count={n}", category=ClaimType.COUNT.value,
                    subset="count", ambiguous=amb, metadata={"class": c, "annotated_count": n, "claimed": k}))
            made = True
        # spatial left/right
        singles = [c for c, lst in by_cls.items() if len(lst) == 1 and lst[0]["area"] >= 0.01 * area]
        pairs = []
        for i, a in enumerate(sorted(singles)):
            for b in sorted(singles)[i + 1:]:
                ba, bb = by_cls[a][0]["bbox"], by_cls[b][0]["bbox"]
                ax, bx = ba[0] + ba[2] / 2, bb[0] + bb[2] / 2
                # clear separation: centers differ by > 25% of width and boxes overlap horizontally < 10%
                ov = max(0, min(ba[0] + ba[2], bb[0] + bb[2]) - max(ba[0], bb[0]))
                if abs(ax - bx) > 0.25 * im["width"] and ov < 0.1 * min(ba[2], bb[2]):
                    pairs.append((a, b, ax < bx))
        if pairs:
            a, b, a_left = rng.choice(pairs)
            for rel, gold in (("left_of", "SUPPORTED" if a_left else "CONTRADICTED"),
                              ("right_of", "CONTRADICTED" if a_left else "SUPPORTED")):
                rel_txt = "to the left of" if rel == "left_of" else "to the right of"
                claim = AtomicClaim(claim_id="c1", text=f"The {a} is {rel_txt} the {b}.",
                                    claim_type=ClaimType.SPATIAL_RELATION, subject=_entity(a), object=_entity(b, "e2"),
                                    predicate=rel, spatial_relation=rel,
                                    required_evidence=[EvidenceRequirement.LOCALIZE_SUBJECT,
                                                       EvidenceRequirement.LOCALIZE_OBJECT,
                                                       EvidenceRequirement.PAIRWISE_GEOMETRY])
                out.append(EvalExample(
                    example_id=f"coco-spatial-{iid}-{a.replace(' ', '_')}-{b.replace(' ', '_')}-{rel}",
                    dataset="coco-count-spatial", dataset_version="instances_val2014", image_id=str(iid),
                    image_path=f"{images_rel_dir}/{fname}", claim=claim, gold=gold,
                    original_label=f"{a} {'left' if a_left else 'right'} of {b}",
                    category=ClaimType.SPATIAL_RELATION.value, subset="spatial", metadata={"a": a, "b": b}))
            made = True
        if made:
            used += 1
    return out


# ------------------------------------------------------------------------------------------- generic JSONL


def load_jsonl(path: Path) -> list[EvalExample]:
    """Normalized-format JSONL (one EvalExample per line). Used for custom or converted datasets."""
    return [EvalExample.from_json(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def save_jsonl(examples: list[EvalExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e.to_json()) for e in examples) + "\n")


# ------------------------------------------------------------------------------------------- CHAIR


def coco_object_mentions(caption: str) -> set[str]:
    """COCO classes mentioned in a caption (lexicon synonyms, simple plural stripping)."""
    words = re.findall(r"[a-z]+", caption.lower())
    found = set()
    for i in range(len(words)):
        for n in (2, 1):
            if i + n > len(words):
                continue
            phrase = " ".join(words[i:i + n])
            cands = [phrase, phrase[:-1] if phrase.endswith("s") else phrase, phrase[:-2] if phrase.endswith("es") else phrase]
            for c in cands:
                if c in SYNONYMS:
                    found.add(SYNONYMS[c])
                    break
    return found


def chair(captions: dict[str, str], gt_objects: dict[str, set[str]]) -> dict[str, float | None]:
    """CHAIR-i: fraction of mentioned objects not in ground truth; CHAIR-s: fraction of captions with any."""
    mentioned = hallucinated = caps_with = 0
    for img, cap in captions.items():
        objs = coco_object_mentions(cap)
        bad = objs - gt_objects.get(img, set())
        mentioned += len(objs)
        hallucinated += len(bad)
        caps_with += bool(bad)
    return {"chair_i": hallucinated / mentioned if mentioned else None,
            "chair_s": caps_with / len(captions) if captions else None,
            "objects_mentioned": float(mentioned), "captions": float(len(captions))}


# ------------------------------------------------------------------------------------------- images


def fetch_coco_images(files: list[str], dest: Path, base_url: str = "http://images.cocodataset.org/val2014",
                      workers: int = 8) -> dict[str, str]:
    """Download COCO images not yet cached; returns {file: sha256}. Files are validated as JPEG."""
    import urllib.request

    dest.mkdir(parents=True, exist_ok=True)

    def one(f: str) -> tuple[str, str]:
        if not re.fullmatch(r"COCO_val2014_\d{12}\.jpg", f):
            raise ValueError(f"unexpected file name {f}")
        p = dest / f
        if not p.exists() or p.stat().st_size < 1000:
            for _ in range(5):
                try:
                    with urllib.request.urlopen(f"{base_url}/{f}", timeout=60) as r:  # noqa: S310 - fixed dataset host
                        data = r.read()
                    if data[:3] != b"\xff\xd8\xff":
                        raise ValueError("not a JPEG")
                    p.write_bytes(data)
                    break
                except Exception:  # noqa: BLE001, S112 - retry transient network errors
                    continue
        return f, hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""

    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(one, sorted(set(files))))
