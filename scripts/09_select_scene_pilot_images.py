#!/usr/bin/env python3
"""Select a reproducible scene-specific pilot image set.

The first use case is a small data3 power-scene pilot. It prefers images that
had old SAM3 pseudo-label hits for target classes, then fills the rest from the
same scene directory so every selected image remains auditable by folder name.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_scene_images(image_root: Path, scene_dir: str) -> List[Path]:
    root = image_root / scene_dir
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def recover_image_key(base_with_hash: str) -> str:
    m = re.match(r"^(.*)_([0-9a-fA-F]{10})$", base_with_hash)
    return m.group(1) if m else base_with_hash


def build_scene_index(images: Iterable[Path]) -> Dict[str, Path]:
    out = {}
    for image in images:
        out.setdefault(image.stem.lower(), image)
    return out


def iter_hits(pseudo_root: Path, classes: set[str]) -> Iterable[tuple[str, str, str, Path]]:
    masks_union = pseudo_root / "masks_union"
    for mask_path in sorted(masks_union.rglob("*.png")):
        if "__" not in mask_path.stem:
            continue
        base_with_hash, prompt = mask_path.stem.rsplit("__", 1)
        prompt = prompt.lower()
        if prompt not in classes:
            continue
        yield mask_path.parent.name, recover_image_key(base_with_hash).lower(), prompt, mask_path


def row_for_image(image: Path, scene_dir: str, scene_type: str, source: str, classes: List[str], masks: List[str], score: int) -> Dict[str, Any]:
    return {
        "image": str(image),
        "scene_dir": scene_dir,
        "scene_type": scene_type,
        "selection_source": source,
        "hit_classes": sorted(classes),
        "hit_masks": masks[:10],
        "selection_score": score,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", default="/home/Groups/group2/Working/TJY/sam3_ir_test/data/ir_images")
    parser.add_argument("--pseudo_root", default="/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/pseudo_lora_b1_step300")
    parser.add_argument("--scene_dir", default="data3")
    parser.add_argument("--scene_type", default="power/electric_scene")
    parser.add_argument("--hit_classes", default="pole,insulator,person,vehicle,car")
    parser.add_argument("--out_dir", default="/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_data3_power_pilot_current")
    parser.add_argument("--max_images", type=int, default=50)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()

    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hit_classes = {x.strip().lower() for x in args.hit_classes.split(",") if x.strip()}

    scene_images = iter_scene_images(image_root, args.scene_dir)
    scene_index = build_scene_index(scene_images)
    by_image = defaultdict(lambda: {"classes": set(), "masks": []})
    missing_hits = 0
    for hit_scene, image_key, prompt, mask_path in iter_hits(Path(args.pseudo_root), hit_classes):
        if hit_scene.lower() != args.scene_dir.lower():
            continue
        image = scene_index.get(image_key)
        if image is None:
            missing_hits += 1
            continue
        by_image[str(image)]["classes"].add(prompt)
        by_image[str(image)]["masks"].append(str(mask_path))

    scored = []
    for image_str, meta in by_image.items():
        cls = meta["classes"]
        score = 0
        score += 6 if "insulator" in cls else 0
        score += 5 if "pole" in cls else 0
        score += 3 if "person" in cls else 0
        score += 1 if cls & {"vehicle", "car"} else 0
        scored.append(row_for_image(Path(image_str), args.scene_dir, args.scene_type, "old_pseudo_hit", sorted(cls), meta["masks"], score))
    scored.sort(key=lambda x: (-x["selection_score"], x["image"]))

    selected = scored[: args.max_images]
    selected_images = {r["image"] for r in selected}
    remaining = [p for p in scene_images if str(p) not in selected_images]
    random.Random(args.seed).shuffle(remaining)
    for image in remaining[: max(0, args.max_images - len(selected))]:
        selected.append(row_for_image(image, args.scene_dir, args.scene_type, "scene_random_fill", [], [], 0))

    image_list = out_dir / "image_list.txt"
    manifest = out_dir / "scene_manifest.jsonl"
    summary_path = out_dir / "summary.json"
    image_list.write_text("\n".join(r["image"] for r in selected) + ("\n" if selected else ""), encoding="utf-8")
    with manifest.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    hit_counter = Counter()
    source_counter = Counter()
    for row in selected:
        hit_counter.update(row["hit_classes"])
        source_counter[row["selection_source"]] += 1
    summary = {
        "scene_dir": args.scene_dir,
        "scene_type": args.scene_type,
        "scene_images": len(scene_images),
        "selected_images": len(selected),
        "old_pseudo_hit_candidates": len(scored),
        "missing_hits": missing_hits,
        "hit_classes": sorted(hit_classes),
        "selected_hit_class_counter": dict(hit_counter),
        "selection_source_counter": dict(source_counter),
        "image_list": str(image_list),
        "scene_manifest": str(manifest),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
