#!/usr/bin/env python3
"""Deduplicate and select representative data1 infrared images.

The script is intentionally dependency-light: it uses PIL + numpy only, and
implements exact SHA1 grouping plus BK-tree lookup over 64-bit perceptual hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from PIL import Image, ImageFilter, ImageOps


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            yield path


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def hash_bits(bits: np.ndarray) -> int:
    out = 0
    for bit in bits.astype(bool).reshape(-1):
        out = (out << 1) | int(bit)
    return int(out)


def dhash(gray: Image.Image) -> int:
    small = gray.resize((9, 8), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.int16)
    return hash_bits(arr[:, 1:] > arr[:, :-1])


def phash_lite(gray: Image.Image) -> int:
    small = gray.resize((8, 8), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float32)
    return hash_bits(arr > float(arr.mean()))


def hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


class BKNode:
    def __init__(self, value: int, item: int):
        self.value = value
        self.items = [item]
        self.children: Dict[int, "BKNode"] = {}


class BKTree:
    def __init__(self):
        self.root: BKNode | None = None

    def add(self, value: int, item: int) -> None:
        if self.root is None:
            self.root = BKNode(value, item)
            return
        node = self.root
        while True:
            dist = hamming(value, node.value)
            if dist == 0:
                node.items.append(item)
                return
            child = node.children.get(dist)
            if child is None:
                node.children[dist] = BKNode(value, item)
                return
            node = child

    def query(self, value: int, radius: int) -> List[int]:
        if self.root is None:
            return []
        out: List[int] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            dist = hamming(value, node.value)
            if dist <= radius:
                out.extend(node.items)
            lo, hi = dist - radius, dist + radius
            for edge, child in node.children.items():
                if lo <= edge <= hi:
                    stack.append(child)
        return out


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def image_stats(path: Path) -> Dict[str, Any]:
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        gray = img.convert("L")
        arr = np.asarray(gray, dtype=np.float32)
        hist = gray.histogram()
        probs = np.asarray(hist, dtype=np.float64)
        probs = probs[probs > 0] / max(1.0, probs.sum())
        entropy = float(-(probs * np.log2(probs)).sum())
        edges = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
        sharpness = float(edges.var())
        return {
            "ok": True,
            "width": int(gray.width),
            "height": int(gray.height),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "entropy": entropy,
            "sharpness": sharpness,
            "dhash": dhash(gray),
            "phash": phash_lite(gray),
            "error": "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "width": 0,
            "height": 0,
            "mean": 0.0,
            "std": 0.0,
            "entropy": 0.0,
            "sharpness": 0.0,
            "dhash": 0,
            "phash": 0,
            "error": repr(exc),
        }


def is_bad_quality(meta: Dict[str, Any]) -> bool:
    if not meta["ok"]:
        return True
    if meta["width"] < 16 or meta["height"] < 16:
        return True
    if meta["mean"] <= 2.0 or meta["mean"] >= 253.0:
        return True
    if meta["std"] < 1.0 or meta["entropy"] < 0.25:
        return True
    return False


def quality_score(meta: Dict[str, Any]) -> float:
    pixels = meta["width"] * meta["height"]
    exposure_penalty = abs(meta["mean"] - 128.0) / 128.0
    return (
        np.log1p(pixels) * 0.30
        + min(meta["entropy"], 8.0) * 0.35
        + np.log1p(max(0.0, meta["sharpness"])) * 0.25
        + min(meta["std"], 80.0) / 80.0 * 0.10
        - exposure_penalty * 0.20
    )


def stable_group_id(rep: Path, members: List[Path]) -> str:
    text = str(rep) + "\n" + "\n".join(str(x) for x in sorted(members))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", default="/home/Groups/group2/Working/TJY/sam3_ir_test/data/ir_images/data1")
    parser.add_argument("--out_dir", default="/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_data1_dedup5000_sam3_qwen_current")
    parser.add_argument("--max_images", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--near_threshold", type=int, default=6)
    parser.add_argument("--max_scan", type=int, default=0, help="Optional debug cap before deduplication.")
    args = parser.parse_args()

    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list(iter_images(image_root))
    if args.max_scan > 0:
        images = images[: args.max_scan]

    rows = []
    exact = defaultdict(list)
    quality_reject = Counter()
    for idx, path in enumerate(images):
        sha = sha1_file(path)
        meta = image_stats(path)
        bad = is_bad_quality(meta)
        if bad:
            quality_reject["bad_or_unreadable"] += 1
        row = {"idx": idx, "path": path, "sha1": sha, "bad_quality": bad, **meta}
        rows.append(row)
        exact[sha].append(idx)

    dsu = DSU(len(rows))
    for members in exact.values():
        first = members[0]
        for idx in members[1:]:
            dsu.union(first, idx)

    dh_tree = BKTree()
    ph_tree = BKTree()
    for idx, row in enumerate(rows):
        if row["bad_quality"]:
            continue
        for hit in dh_tree.query(row["dhash"], args.near_threshold):
            if hamming(row["phash"], rows[hit]["phash"]) <= args.near_threshold:
                dsu.union(idx, hit)
        for hit in ph_tree.query(row["phash"], args.near_threshold):
            if hamming(row["dhash"], rows[hit]["dhash"]) <= args.near_threshold:
                dsu.union(idx, hit)
        dh_tree.add(row["dhash"], idx)
        ph_tree.add(row["phash"], idx)

    grouped = defaultdict(list)
    for idx, row in enumerate(rows):
        if not row["bad_quality"]:
            grouped[dsu.find(idx)].append(idx)

    group_records = []
    for members in grouped.values():
        best_idx = max(members, key=lambda i: quality_score(rows[i]))
        rep = rows[best_idx]["path"]
        paths = [rows[i]["path"] for i in members]
        gid = stable_group_id(rep, paths)
        group_records.append({
            "group_id": gid,
            "representative": str(rep),
            "member_count": len(paths),
            "members": [str(p) for p in sorted(paths)],
            "representative_quality_score": float(quality_score(rows[best_idx])),
            "representative_stats": {
                k: rows[best_idx][k]
                for k in ("width", "height", "mean", "std", "entropy", "sharpness")
            },
        })

    rng = random.Random(args.seed)
    group_records.sort(key=lambda r: (-r["representative_quality_score"], r["representative"]))
    top_pool = group_records[: max(args.max_images * 3, args.max_images)]
    rng.shuffle(top_pool)
    selected = sorted(top_pool[: args.max_images], key=lambda r: r["representative"])

    image_list = out_dir / "dedup_image_list.txt"
    groups_path = out_dir / "dedup_groups.jsonl"
    summary_path = out_dir / "dedup_summary.json"
    image_list.write_text("\n".join(r["representative"] for r in selected) + ("\n" if selected else ""), encoding="utf-8")
    with groups_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    exact_duplicates = sum(len(v) - 1 for v in exact.values())
    near_duplicates = sum(max(0, r["member_count"] - 1) for r in group_records)
    summary = {
        "image_root": str(image_root),
        "total_images_scanned": len(images),
        "quality_rejected": int(sum(quality_reject.values())),
        "exact_duplicate_files": int(exact_duplicates),
        "near_duplicate_or_exact_members": int(near_duplicates),
        "dedup_groups_total": len(group_records),
        "selected_images": len(selected),
        "max_images": args.max_images,
        "seed": args.seed,
        "near_threshold": args.near_threshold,
        "dedup_image_list": str(image_list),
        "dedup_groups": str(groups_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
