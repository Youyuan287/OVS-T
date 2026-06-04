#!/usr/bin/env python3
"""Build safer existence-calibration JSONL from dataset-v2 manifests."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


CANONICAL = [
    "person", "car", "vehicle", "truck", "road", "tree", "building",
    "pole", "power line", "insulator", "animal",
]
CONFLICT_GROUPS = [
    {"car", "vehicle", "truck"},
    {"person", "pedestrian", "human"},
    {"power line", "wire", "cable"},
    {"pole", "tower", "utility pole", "power pole"},
    {"tree", "vegetation", "plant"},
]
POWER_SENSITIVE = {"power line", "insulator", "pole"}


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


def conflict_set(pos: Set[str]) -> Set[str]:
    out = set(pos)
    for group in CONFLICT_GROUPS:
        if out & group:
            out |= group
    return out


def build(in_jsonl: Path, neg_per_image: int, neg_ratio: float, seed: int) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    random.seed(seed)
    groups = defaultdict(list)
    for row in read_jsonl(in_jsonl):
        groups[row["image"]].append(row)
    out = []
    pos_count = 0
    neg_count = 0
    for image, rows in groups.items():
        pos_prompts = {r["canonical_prompt"] for r in rows}
        avoid = conflict_set(pos_prompts)
        for r in rows:
            out.append({
                "image": r["image"],
                "prompt": r["canonical_prompt"],
                "mask": r["mask"],
                "exists": 1,
                "source": "v2_pos",
                "weight": r.get("weight", 1.0),
                "quality": r.get("quality", "A"),
            })
            pos_count += 1
        candidates = [p for p in CANONICAL if p not in avoid]
        # Avoid over-teaching absence for power classes unless another power class exists in the same image.
        has_power_context = bool(pos_prompts & POWER_SENSITIVE)
        if not has_power_context:
            candidates = [p for p in candidates if p not in POWER_SENSITIVE]
        random.shuffle(candidates)
        max_negs = min(neg_per_image, int(max(1, len(rows) * neg_ratio)))
        for neg in candidates[:max_negs]:
            out.append({
                "image": image,
                "prompt": neg,
                "mask": "",
                "exists": 0,
                "source": "v2_neg_safe",
                "weight": 0.5,
                "quality": "neg",
            })
            neg_count += 1
    random.shuffle(out)
    pc = Counter((r["exists"], r["prompt"]) for r in out).most_common(50)
    stats = {
        "images": len(groups),
        "positive": pos_count,
        "negative": neg_count,
        "total": len(out),
        "prompt_counter": [{"exists": k[0], "prompt": k[1], "count": v} for k, v in pc],
    }
    return out, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_hq", required=True)
    parser.add_argument("--val_hq", required=True)
    parser.add_argument("--out_dir", default="/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_prompt_irgpt_sam3_qwen8b")
    parser.add_argument("--neg_per_image", type=int, default=2)
    parser.add_argument("--neg_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    train, train_stats = build(Path(args.train_hq), args.neg_per_image, args.neg_ratio, args.seed)
    val, val_stats = build(Path(args.val_hq), args.neg_per_image, args.neg_ratio, args.seed + 1)
    write_jsonl(out_dir / "train_exist_calib.jsonl", train)
    write_jsonl(out_dir / "val_exist_calib.jsonl", val)
    summary = {
        "train": train_stats,
        "val": val_stats,
        "train_jsonl": str(out_dir / "train_exist_calib.jsonl"),
        "val_jsonl": str(out_dir / "val_exist_calib.jsonl"),
    }
    (out_dir / "exist_calib_v2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
