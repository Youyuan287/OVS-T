#!/usr/bin/env python3
"""Filter non-empty SAM3 candidates before Qwen panel generation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from PIL import Image


LARGE_REGION = {"road", "building", "tree"}
SMALL_OR_THIN = {"person", "animal", "insulator", "power line", "wire", "pole"}


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def mask_array(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((1, 1), dtype=bool)
    return np.asarray(Image.open(path).convert("L")) > 127


def mask_iou(a_path: str, b_path: str) -> float:
    a = mask_array(Path(a_path))
    b = mask_array(Path(b_path))
    if a.shape != b.shape:
        b_img = Image.fromarray(b.astype(np.uint8) * 255)
        b = np.asarray(b_img.resize((a.shape[1], a.shape[0]), Image.Resampling.NEAREST)) > 127
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / max(1, union))


def rejection_reason(row: Dict[str, Any]) -> str:
    cls = str(row.get("canonical_prompt", "")).lower()
    area = float(row.get("area_ratio", 0.0) or 0.0)
    if area <= 0.0 or int(row.get("area", 0) or 0) <= 0:
        return "empty_mask"
    if cls in LARGE_REGION:
        if area > 0.97:
            return "large_region_too_large"
        return ""
    if cls not in SMALL_OR_THIN and area > 0.70:
        return "object_too_large"
    return ""


def better(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    a_score = float(a.get("sam3_score", 0.0) or 0.0)
    b_score = float(b.get("sam3_score", 0.0) or 0.0)
    if a_score != b_score:
        return a if a_score > b_score else b
    return a if float(a.get("area_ratio", 0.0) or 0.0) >= float(b.get("area_ratio", 0.0) or 0.0) else b


def dedupe_group(rows: List[Dict[str, Any]], iou_threshold: float) -> tuple[List[Dict[str, Any]], int]:
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for row in sorted(rows, key=lambda r: float(r.get("sam3_score", 0.0) or 0.0), reverse=True):
        replaced = False
        for idx, old in enumerate(kept):
            if mask_iou(row["mask"], old["mask"]) > iou_threshold:
                kept[idx] = better(row, old)
                dropped += 1
                replaced = True
                break
        if not replaced:
            kept.append(row)
    return kept, dropped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out_jsonl", default="")
    parser.add_argument("--summary", default="")
    parser.add_argument("--iou_threshold", type=float, default=0.90)
    parser.add_argument("--max_per_image_class", type=int, default=0)
    args = parser.parse_args()

    in_path = Path(args.candidates)
    out_path = Path(args.out_jsonl) if args.out_jsonl else in_path.with_name("sam3_candidates_nonempty_filtered.jsonl")
    summary_path = Path(args.summary) if args.summary else out_path.with_name("sam3_candidates_nonempty_filtered_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    reject = Counter()
    stats = Counter()
    class_input = Counter()
    class_kept = Counter()
    for row in read_jsonl(in_path):
        stats["input"] += 1
        cls = str(row.get("canonical_prompt", "")).lower()
        class_input[cls] += 1
        reason = rejection_reason(row)
        if reason:
            reject[reason] += 1
            continue
        groups[(row["image"], cls)].append(row)

    kept_all: List[Dict[str, Any]] = []
    duplicate_drop = 0
    for key, items in groups.items():
        kept, dropped = dedupe_group(items, args.iou_threshold)
        duplicate_drop += dropped
        if args.max_per_image_class > 0:
            kept = sorted(kept, key=lambda r: float(r.get("sam3_score", 0.0) or 0.0), reverse=True)[: args.max_per_image_class]
        kept_all.extend(kept)

    kept_all.sort(key=lambda r: (r["image"], str(r.get("canonical_prompt", "")), str(r.get("candidate_id", ""))))
    with out_path.open("w", encoding="utf-8") as f:
        for row in kept_all:
            class_kept[str(row.get("canonical_prompt", "")).lower()] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": int(stats["input"]),
        "kept": len(kept_all),
        "rejected": dict(reject),
        "duplicate_iou_dropped": duplicate_drop,
        "iou_threshold": args.iou_threshold,
        "max_per_image_class": args.max_per_image_class,
        "class_input": dict(class_input),
        "class_kept": dict(class_kept),
        "out_jsonl": str(out_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
