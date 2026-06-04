#!/usr/bin/env python3
"""Select targeted pilot images from existing pseudo-label hits.

This is used when random sampling is not diagnostic for difficult classes such
as power line, pole, insulator, and small person. It mines the old SAM3 pseudo
label directory for images where related prompts produced any mask, then emits
an image list for Dataset V2 prompt proposal generation.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_image_index(image_root: Path) -> Dict[tuple[str, str], Path]:
    index = {}
    for p in image_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            index[(p.parent.name.lower(), p.stem.lower())] = p
            index[("ALL", p.stem.lower())] = p
    return index


def recover_image_key(base_with_hash: str) -> str:
    m = re.match(r"^(.*)_([0-9a-fA-F]{10})$", base_with_hash)
    return m.group(1) if m else base_with_hash


def iter_hits(pseudo_root: Path, classes: set[str]) -> Iterable[tuple[str, str, Path]]:
    masks_union = pseudo_root / "masks_union"
    for mask_path in sorted(masks_union.rglob("*.png")):
        if "__" not in mask_path.stem:
            continue
        base_with_hash, prompt = mask_path.stem.rsplit("__", 1)
        prompt = prompt.lower()
        if prompt not in classes:
            continue
        yield mask_path.parent.name.lower(), recover_image_key(base_with_hash), mask_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", default="/home/Groups/group2/Working/TJY/sam3_ir_test/data/ir_images")
    parser.add_argument("--pseudo_root", default="/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/pseudo_lora_b1_step300")
    parser.add_argument("--out_dir", default="/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_targeted_sampling")
    parser.add_argument("--classes", default="pole,insulator,person,vehicle,car")
    parser.add_argument("--max_images", type=int, default=80)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()

    random.seed(args.seed)
    classes = {x.strip().lower() for x in args.classes.split(",") if x.strip()}
    image_root = Path(args.image_root)
    pseudo_root = Path(args.pseudo_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = build_image_index(image_root)
    by_image = defaultdict(lambda: {"classes": set(), "masks": []})
    missing = 0
    for subdir, image_key, mask_path in iter_hits(pseudo_root, classes):
        image = index.get((subdir, image_key.lower())) or index.get(("ALL", image_key.lower()))
        if image is None:
            missing += 1
            continue
        prompt = mask_path.stem.rsplit("__", 1)[1].lower()
        by_image[str(image)]["classes"].add(prompt)
        by_image[str(image)]["masks"].append(str(mask_path))

    rows = []
    for image, meta in by_image.items():
        score = 0
        cls = meta["classes"]
        score += 5 if "insulator" in cls else 0
        score += 4 if "pole" in cls else 0
        score += 2 if "person" in cls else 0
        score += 1 if cls & {"vehicle", "car"} else 0
        rows.append({
            "image": image,
            "classes": sorted(cls),
            "score": score,
            "masks": meta["masks"][:10],
        })

    rows.sort(key=lambda x: (-x["score"], x["image"]))
    selected = rows[: args.max_images]
    image_list = out_dir / "targeted_images.txt"
    detail_path = out_dir / "targeted_images_detail.jsonl"
    summary_path = out_dir / "targeted_images_summary.json"

    image_list.write_text("\n".join(r["image"] for r in selected) + ("\n" if selected else ""), encoding="utf-8")
    with detail_path.open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counter = Counter()
    for r in selected:
        counter.update(r["classes"])
    summary = {
        "selected_images": len(selected),
        "candidate_images": len(rows),
        "missing_images": missing,
        "classes": sorted(classes),
        "selected_class_counter": dict(counter),
        "image_list": str(image_list),
        "detail_jsonl": str(detail_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
