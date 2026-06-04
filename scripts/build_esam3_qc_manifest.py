import os
import re
import json
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_image_index(image_root):
    image_root = Path(image_root)
    index = {}

    for p in image_root.rglob("*"):
        if p.suffix.lower() not in IMG_EXTS:
            continue

        rel_parent = p.parent.name
        stem = p.stem

        # 优先按 data1/data2/... + stem 精确匹配
        index[(rel_parent.lower(), stem.lower())] = str(p)

        # 全局兜底
        index[("ALL", stem.lower())] = str(p)

    return index


def recover_image_key(base_with_hash):
    # masks_union 文件名形如：00191_61098dddbe__car.png
    # 原图 stem 通常是：00191
    m = re.match(r"^(.*)_([0-9a-fA-F]{10})$", base_with_hash)
    if m:
        return m.group(1)
    return base_with_hash


def mask_qc(mask_path, prompt):
    mask = np.array(Image.open(mask_path).convert("L"))
    bin_mask = (mask > 127).astype(np.uint8)

    h, w = bin_mask.shape[:2]
    area = int(bin_mask.sum())
    area_ratio = area / max(1, h * w)

    if area == 0:
        return None

    # 类别面积上限：road/building/tree 可能较大，其他目标严格一些
    large_classes = {"road", "building", "tree", "sky", "water"}
    max_area_ratio = 0.95 if prompt.lower() in large_classes else 0.70

    if area_ratio < 0.0002 or area_ratio > max_area_ratio:
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

    # 去掉背景
    comp_areas = []
    for i in range(1, num_labels):
        comp_areas.append(int(stats[i, cv2.CC_STAT_AREA]))

    num_components = len(comp_areas)
    largest_component_ratio = max(comp_areas) / max(1, area) if comp_areas else 0.0

    if num_components > 30:
        return None
    if largest_component_ratio < 0.25:
        return None

    # 简单分级
    quality = "A"
    weight = 1.0

    if num_components > 10 or largest_component_ratio < 0.50:
        quality = "B"
        weight = 0.5

    if area_ratio < 0.0008:
        quality = "B"
        weight = min(weight, 0.5)

    return {
        "area_ratio": area_ratio,
        "num_components": num_components,
        "largest_component_ratio": largest_component_ratio,
        "quality": quality,
        "weight": weight,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", default="/home/Groups/group2/Working/TJY/sam3_ir_test/data/ir_images")
    parser.add_argument("--pseudo_root", default="/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/pseudo_lora_b1_step300")
    parser.add_argument("--out_dir", default="/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/esam3_manifest_qc")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--max_items", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    image_index = build_image_index(args.image_root)
    masks_union = Path(args.pseudo_root) / "masks_union"

    rows = []
    missing = 0
    rejected = 0

    for mask_path in sorted(masks_union.rglob("*.png")):
        subdir = mask_path.parent.name.lower()
        stem = mask_path.stem

        if "__" not in stem:
            continue

        base_with_hash, prompt = stem.rsplit("__", 1)
        image_key = recover_image_key(base_with_hash)

        image_path = image_index.get((subdir, image_key.lower()))
        if image_path is None:
            image_path = image_index.get(("ALL", image_key.lower()))

        if image_path is None:
            missing += 1
            continue

        qc = mask_qc(mask_path, prompt)
        if qc is None:
            rejected += 1
            continue

        rows.append({
            "image": image_path,
            "prompt": prompt,
            "mask": str(mask_path),
            "exists": 1,
            "weight": qc["weight"],
            "quality": qc["quality"],
            "area_ratio": qc["area_ratio"],
            "num_components": qc["num_components"],
            "largest_component_ratio": qc["largest_component_ratio"],
        })

    random.shuffle(rows)
    if args.max_items and args.max_items > 0:
        rows = rows[:args.max_items]

    n_val = max(1, int(len(rows) * args.val_ratio))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_qc.jsonl"
    val_path = out_dir / "val_qc.jsonl"
    summary_path = out_dir / "summary.json"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "total_kept": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "missing_image": missing,
        "rejected_by_qc": rejected,
        "train_jsonl": str(train_path),
        "val_jsonl": str(val_path),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
