#!/usr/bin/env python3
"""Merge SAM3 candidates and Qwen QC scores into high-confidence manifests."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from PIL import Image


LARGE_REGION = {"road", "building", "tree"}
THIN_STRUCTURE = {"power line", "wire", "pole"}
SMALL_TARGET = {"person", "animal", "insulator", "power line", "wire"}
CONFLICT_GROUPS = [
    {"car", "vehicle", "truck"},
    {"power line", "wire"},
    {"person", "human", "pedestrian"},
]


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_qwen(path: Path) -> Dict[str, Dict[str, Any]]:
    return {r["candidate_id"]: r for r in read_jsonl(path)}


def mask_components(mask_path: Path) -> tuple[int, float]:
    if not mask_path.exists():
        return 0, 0.0
    m = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)
    if m.sum() == 0:
        return 0, 0.0
    h, w = m.shape
    visited = np.zeros_like(m, dtype=bool)
    comp_areas = []
    for y in range(h):
        for x in range(w):
            if m[y, x] == 0 or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and m[ny, nx] > 0:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            comp_areas.append(area)
    largest_ratio = max(comp_areas) / max(1, int(m.sum())) if comp_areas else 0.0
    return len(comp_areas), float(largest_ratio)


def class_rule_score(row: Dict[str, Any], qwen: Dict[str, Any]) -> tuple[float, str]:
    cls = row["canonical_prompt"]
    area = float(row.get("area_ratio", 0.0))
    comps = int(row.get("num_components", 0))
    largest = float(row.get("largest_component_ratio", 0.0))
    semantic = float(qwen.get("semantic_match", 0.0))
    leakage = float(qwen.get("background_leakage", 0.5))

    if area <= 0:
        return 0.0, "empty_mask"
    if cls in LARGE_REGION:
        if area > 0.97:
            return 0.2, "large_region_too_large"
        if leakage > 0.65:
            return 0.4, "large_region_background_leakage"
        return 0.95, "large_region_ok"
    if cls in THIN_STRUCTURE:
        if area < 0.00002:
            return 0.25, "thin_structure_too_small"
        if semantic < 0.35:
            return 0.35, "thin_structure_low_semantic"
        return 0.9, "thin_structure_ok"
    if cls == "insulator":
        if area < 0.00001:
            return 0.2, "insulator_too_small"
        if semantic < 0.55:
            return 0.35, "insulator_needs_high_semantic"
        return 0.9, "insulator_ok"
    if cls in SMALL_TARGET:
        if area < 0.00002 and semantic < 0.6:
            return 0.35, "small_target_low_semantic"
    if area > 0.70:
        return 0.25, "object_too_large"
    if comps > 30 and largest < 0.25:
        return 0.35, "object_too_fragmented"
    return 0.9, "object_ok"


def quality_from_score(score: float) -> tuple[str, float, str]:
    if score >= 0.55:
        return "A", 1.0, "hard"
    if score >= 0.35:
        return "B", 0.5, "soft_or_review"
    return "C", 0.0, "drop"


def build_rows(candidates_path: Path, qwen_path: Path) -> tuple[List[Dict[str, Any]], Counter]:
    qwen_by_id = load_qwen(qwen_path)
    rows = []
    reject = Counter()
    for row in read_jsonl(candidates_path):
        cid = row["candidate_id"]
        qwen = qwen_by_id.get(cid)
        if qwen is None:
            reject["missing_qwen"] += 1
            continue
        if not row.get("num_components"):
            comps, largest = mask_components(Path(row["mask"]))
            row["num_components"] = comps
            row["largest_component_ratio"] = largest
        rule_score, rule_reason = class_rule_score(row, qwen)
        sam3_score = max(0.05, min(1.0, float(row.get("sam3_score", 0.0) or 0.0)))
        prompt_consistency = 1.0 if row["canonical_prompt"] in str(row.get("raw_prompt", "")).lower() else 0.85
        q_sem = max(0.0, min(1.0, float(qwen.get("semantic_match", 0.0))))
        coverage = max(0.0, min(1.0, float(qwen.get("mask_coverage", 0.0))))
        context = max(0.0, min(1.0, float(qwen.get("context_consistency", 0.5))))
        leakage_penalty = 1.0 - max(0.0, min(1.0, float(qwen.get("background_leakage", 0.0)))) * 0.5
        final_score = sam3_score * prompt_consistency * q_sem * coverage * rule_score * context * leakage_penalty
        quality, weight, label_mode = quality_from_score(final_score)
        out = {
            "image": row["image"],
            "mask": row["mask"],
            "canonical_prompt": "power line" if row["canonical_prompt"] == "wire" else row["canonical_prompt"],
            "raw_prompt": row["raw_prompt"],
            "source_mode": row["source_mode"],
            "quality": quality,
            "weight": weight,
            "label_mode": label_mode,
            "final_score": round(final_score, 6),
            "sam3_score": row.get("sam3_score", 0.0),
            "qwen_semantic_match": qwen.get("semantic_match", 0.0),
            "qwen_mask_coverage": qwen.get("mask_coverage", 0.0),
            "qwen_background_leakage": qwen.get("background_leakage", 0.0),
            "class_rule_score": rule_score,
            "class_rule_reason": rule_reason,
            "area_ratio": row.get("area_ratio", 0.0),
            "num_components": row.get("num_components", 0),
            "largest_component_ratio": row.get("largest_component_ratio", 0.0),
            "candidate_id": cid,
        }
        if quality == "C":
            reject[rule_reason] += 1
            continue
        rows.append(out)
    return rows, reject


def balanced(rows: List[Dict[str, Any]], max_per_class: int, target_total: int) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["canonical_prompt"]].append(row)
    out = []
    for cls, items in groups.items():
        items.sort(key=lambda x: x["final_score"], reverse=True)
        out.extend(items[:max_per_class])
    out.sort(key=lambda x: x["final_score"], reverse=True)
    if target_total > 0:
        out = out[:target_total]
    return out


def split_by_image(rows: List[Dict[str, Any]], val_ratio: float, seed: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    images = sorted({r["image"] for r in rows})
    random.Random(seed).shuffle(images)
    n_val = max(1, int(len(images) * val_ratio)) if images else 0
    val_images = set(images[:n_val])
    train, val = [], []
    for r in rows:
        (val if r["image"] in val_images else train).append(r)
    return train, val


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--qwen", required=True)
    parser.add_argument("--out_dir", default="/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_prompt_irgpt_sam3_qwen8b")
    parser.add_argument("--max_per_class", type=int, default=3000)
    parser.add_argument("--target_total", type=int, default=10000)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rows, reject = build_rows(Path(args.candidates), Path(args.qwen))
    rows = balanced(rows, args.max_per_class, args.target_total)
    train, val = split_by_image(rows, args.val_ratio, args.seed)
    write_jsonl(out_dir / "train_hq.jsonl", train)
    write_jsonl(out_dir / "val_hq.jsonl", val)

    counter = Counter(r["canonical_prompt"] for r in rows)
    qcounter = Counter(r["quality"] for r in rows)
    summary = {
        "kept_total": len(rows),
        "train": len(train),
        "val": len(val),
        "classes": dict(counter),
        "qualities": dict(qcounter),
        "reject_reasons": dict(reject),
        "train_jsonl": str(out_dir / "train_hq.jsonl"),
        "val_jsonl": str(out_dir / "val_hq.jsonl"),
    }
    (out_dir / "manifest_v2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
