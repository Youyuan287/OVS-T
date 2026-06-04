#!/usr/bin/env python3
"""Build prompt-bank proposals for dataset-v2 pseudo labeling.

WheatCao/ICCV2025-IRGPT currently publishes dataset/benchmark assets and model
weights, but no official stable inference script. Therefore the default path
does not depend on IRGPT model generation. It creates auditable low-confidence
text proposals from the infrared prompt bank, optionally allowing an external
proposal command when a reliable IRGPT-compatible worker becomes available.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_prompt_bank(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["classes"] if isinstance(data, dict) and "classes" in data else data


def iter_images(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def load_image_list(path: Path) -> List[Path]:
    images = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                images.append(Path(line))
    return images


def filter_classes(classes: List[Dict[str, Any]], include_classes: str) -> List[Dict[str, Any]]:
    if not include_classes:
        return classes
    wanted = {x.strip().lower() for x in include_classes.split(",") if x.strip()}
    out = []
    for item in classes:
        canonical = str(item.get("maps_to", item["canonical"])).lower()
        raw = str(item["canonical"]).lower()
        if canonical in wanted or raw in wanted:
            out.append(item)
    return out


def parse_boxes(text: str) -> List[List[float]]:
    boxes = []
    pattern = r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    for m in re.finditer(pattern, text or ""):
        boxes.append([float(x) for x in m.groups()])
    return boxes


def extract_json(text: str) -> Dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def run_command(template: str, image: Path, prompt_bank: Path, timeout: int) -> str:
    command = template.format(image=str(image), prompt_bank=str(prompt_bank))
    proc = subprocess.run(shlex.split(command), text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        return json.dumps({"error": proc.stderr[-2000:], "stdout": proc.stdout[-2000:]}, ensure_ascii=False)
    return proc.stdout


def fallback_candidates(classes: List[Dict[str, Any]], max_classes: int) -> List[Dict[str, Any]]:
    picked = classes[:max_classes]
    out = []
    for item in picked:
        prompts = item.get("prompts", [item["canonical"]])
        out.append({
            "canonical": item.get("maps_to", item["canonical"]),
            "prompt": prompts[0],
            "bbox": None,
            "confidence": 0.25,
            "is_small_target": bool(item.get("is_small_target_class", False)),
            "reason": "prompt_bank_candidate_without_external_box",
        })
    return out


def normalize_candidates(raw: Dict[str, Any] | None, classes: List[Dict[str, Any]], fallback_count: int) -> tuple[str, List[Dict[str, Any]]]:
    if not raw:
        return "unknown", fallback_candidates(classes, fallback_count)
    scene_type = str(raw.get("scene_type", raw.get("scene", "unknown")))
    candidates = raw.get("candidates", raw.get("objects", []))
    if not isinstance(candidates, list) or not candidates:
        return scene_type, fallback_candidates(classes, fallback_count)
    out = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        canonical = cand.get("canonical") or cand.get("class") or cand.get("label") or cand.get("prompt")
        prompt = cand.get("prompt") or canonical
        if not canonical or not prompt:
            continue
        bbox = cand.get("bbox") or cand.get("box")
        if bbox is None:
            boxes = parse_boxes(str(cand.get("raw_response", cand.get("response", ""))))
            bbox = boxes[0] if boxes else None
        out.append({
            "canonical": str(canonical).strip().lower(),
            "prompt": str(prompt).strip(),
            "bbox": bbox,
            "confidence": float(cand.get("confidence", cand.get("score", 0.5))),
            "is_small_target": bool(cand.get("is_small_target", False)),
            "reason": str(cand.get("reason", "")),
        })
    return scene_type, out or fallback_candidates(classes, fallback_count)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", default="/home/Groups/group2/Working/TJY/sam3_ir_test/data/ir_images")
    parser.add_argument("--prompt_bank", default="data/prompt_bank_ir_v2.json")
    parser.add_argument("--out_jsonl", default="/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_prompt_sam3_qwen8b/prompt_proposals.jsonl")
    parser.add_argument("--summary", default="")
    parser.add_argument("--image_list", default="", help="Optional newline-separated absolute image paths.")
    parser.add_argument("--include_classes", default="", help="Comma-separated canonical classes to emit.")
    parser.add_argument("--max_images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--fallback_classes", type=int, default=12)
    parser.add_argument("--external_proposal_command", default="", help="Optional command template with {image} and {prompt_bank}; stdout should be JSON.")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    random.seed(args.seed)
    prompt_bank_path = Path(args.prompt_bank)
    classes = filter_classes(load_prompt_bank(prompt_bank_path), args.include_classes)
    if args.image_list:
        images = load_image_list(Path(args.image_list))
    else:
        images = list(iter_images(Path(args.image_root)))
    random.shuffle(images)
    if args.max_images > 0:
        images = images[: args.max_images]

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "images": 0,
        "candidates": 0,
        "used_external_proposal_model": bool(args.external_proposal_command),
        "fallback_images": 0,
        "empty_or_error_raw": 0,
    }

    with out_path.open("w", encoding="utf-8") as f:
        for idx, image in enumerate(images):
            raw_text = ""
            raw_json = None
            if args.external_proposal_command:
                raw_text = run_command(args.external_proposal_command, image, prompt_bank_path, args.timeout)
                raw_json = extract_json(raw_text)
                if raw_json is None:
                    stats["empty_or_error_raw"] += 1
            scene_type, candidates = normalize_candidates(raw_json, classes, args.fallback_classes)
            if raw_json is None:
                stats["fallback_images"] += 1
            for cand in candidates:
                row = {
                    "image": str(image),
                    "image_index": idx,
                    "scene_type": scene_type,
                    "canonical": cand["canonical"],
                    "prompt": cand["prompt"],
                    "bbox": cand["bbox"],
                    "confidence": cand["confidence"],
                    "is_small_target": cand["is_small_target"],
                    "reason": cand["reason"],
                    "raw_external_response": raw_text[:4000],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["candidates"] += 1
            stats["images"] += 1

    summary_path = Path(args.summary) if args.summary else out_path.with_name("prompt_proposals_summary.json")
    summary_path.write_text(json.dumps({**stats, "out_jsonl": str(out_path)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**stats, "out_jsonl": str(out_path), "summary": str(summary_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
