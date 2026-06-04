#!/usr/bin/env python3
"""Build 2x2 visual panels for Qwen3-VL mask quality critique."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    m = Image.open(path).convert("L")
    if m.size != size:
        m = m.resize(size, Image.Resampling.NEAREST)
    return (np.array(m) > 127).astype(np.uint8)


def overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    img = np.array(image.convert("RGB")).astype(np.float32)
    color = np.zeros_like(img)
    color[..., 0] = 255
    color[..., 1] = 80
    m = mask > 0
    img[m] = img[m] * 0.55 + color[m] * 0.45
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def masked_region(image: Image.Image, mask: np.ndarray) -> Image.Image:
    img = np.array(image.convert("RGB"))
    out = np.zeros_like(img)
    out[mask > 0] = img[mask > 0]
    return Image.fromarray(out)


def crop_from_bbox(image: Image.Image, bbox: Any) -> Image.Image:
    if isinstance(bbox, list) and len(bbox) == 4:
        x1, y1, x2, y2 = [int(round(float(x))) for x in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        if x2 > x1 and y2 > y1:
            return image.crop((x1, y1, x2, y2))
    return image.copy()


def fit(img: Image.Image, size: int = 448) -> Image.Image:
    img = ImageOps.contain(img.convert("RGB"), (size, size))
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def label(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 24), fill=(0, 0, 0))
    draw.text((6, 5), text, fill=(255, 255, 255))
    return img


def build_panel(row: Dict[str, Any], out_path: Path, tile: int) -> None:
    image = Image.open(row["image"]).convert("RGB")
    image = ImageOps.exif_transpose(image)
    mask = load_mask(Path(row["mask"]), image.size)
    crop = crop_from_bbox(image, row.get("bbox"))
    parts = [
        label(fit(image, tile), "original infrared image"),
        label(fit(overlay(image, mask), tile), "mask overlay"),
        label(fit(crop, tile), "bbox/crop zoom"),
        label(fit(masked_region(image, mask), tile), "masked region only"),
    ]
    panel = Image.new("RGB", (tile * 2, tile * 2), (0, 0, 0))
    panel.paste(parts[0], (0, 0))
    panel.paste(parts[1], (tile, 0))
    panel.paste(parts[2], (0, tile))
    panel.paste(parts[3], (tile, tile))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path, quality=92)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out_dir", default="/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_prompt_irgpt_sam3_qwen8b")
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--tile", type=int, default=448)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    panel_dir = out_dir / "qwen_qc_panels"
    tasks_path = out_dir / "qwen_qc_tasks.jsonl"
    stats = {"input": 0, "panels": 0}
    with tasks_path.open("w", encoding="utf-8") as f:
        for row in read_jsonl(Path(args.candidates)):
            if args.max_items and stats["input"] >= args.max_items:
                break
            stats["input"] += 1
            panel_path = panel_dir / row["canonical_prompt"] / f"{row['candidate_id']}.jpg"
            try:
                build_panel(row, panel_path, args.tile)
            except Exception as exc:
                row["panel_error"] = repr(exc)
                continue
            task = {
                "candidate_id": row["candidate_id"],
                "panel": str(panel_path),
                "target_class": row["canonical_prompt"],
                "raw_prompt": row["raw_prompt"],
                "sam3_score": row.get("sam3_score", 0.0),
                "area_ratio": row.get("area_ratio", 0.0),
            }
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
            stats["panels"] += 1
    summary = out_dir / "qwen_qc_panels_summary.json"
    summary.write_text(json.dumps({**stats, "tasks": str(tasks_path)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**stats, "tasks": str(tasks_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
