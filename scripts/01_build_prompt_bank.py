#!/usr/bin/env python3
"""Build an infrared prompt bank for dataset-v2 pseudo labeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPT_BANK = [
    {
        "canonical": "person",
        "prompts": [
            "person", "pedestrian", "human", "thermal human target",
            "small pedestrian in infrared image", "hidden person near trees",
            "standing person", "walking person", "distant person",
        ],
        "scene_hint": ["security", "road", "urban", "wild"],
        "is_small_target_class": True,
        "structure_type": "compact_object",
    },
    {
        "canonical": "car",
        "prompts": ["car", "automobile", "sedan", "parked car", "thermal car", "small car", "distant car"],
        "scene_hint": ["road", "urban", "security"],
        "is_small_target_class": False,
        "structure_type": "compact_object",
    },
    {
        "canonical": "vehicle",
        "prompts": ["vehicle", "thermal vehicle", "small vehicle", "parked vehicle", "distant vehicle", "other vehicle"],
        "scene_hint": ["road", "urban", "security"],
        "is_small_target_class": False,
        "structure_type": "compact_object",
    },
    {
        "canonical": "truck",
        "prompts": ["truck", "thermal truck", "large vehicle", "cargo truck", "distant truck"],
        "scene_hint": ["road", "urban", "industrial"],
        "is_small_target_class": False,
        "structure_type": "compact_object",
    },
    {
        "canonical": "road",
        "prompts": ["road", "street", "urban road", "thermal road surface", "road region", "lane"],
        "scene_hint": ["road", "urban"],
        "is_small_target_class": False,
        "structure_type": "large_region",
    },
    {
        "canonical": "building",
        "prompts": ["building", "house", "thermal building", "building facade", "distant building", "structure"],
        "scene_hint": ["urban", "security", "industrial"],
        "is_small_target_class": False,
        "structure_type": "large_region",
    },
    {
        "canonical": "tree",
        "prompts": ["tree", "vegetation", "plant", "forest", "bush", "thermal vegetation", "trees near road"],
        "scene_hint": ["road", "urban", "wild"],
        "is_small_target_class": False,
        "structure_type": "large_region",
    },
    {
        "canonical": "pole",
        "prompts": ["pole", "utility pole", "electric pole", "power pole", "transmission tower", "power tower", "tower"],
        "scene_hint": ["power_inspection", "road", "urban"],
        "is_small_target_class": False,
        "structure_type": "thin_structure",
    },
    {
        "canonical": "power line",
        "prompts": [
            "power line", "wire", "overhead wire", "electric wire", "power cable",
            "transmission line", "high voltage line", "overhead power line",
        ],
        "scene_hint": ["power_inspection", "urban"],
        "is_small_target_class": True,
        "structure_type": "thin_structure",
    },
    {
        "canonical": "wire",
        "prompts": ["wire", "thin wire", "overhead wire", "electric wire", "cable"],
        "scene_hint": ["power_inspection", "urban"],
        "is_small_target_class": True,
        "structure_type": "thin_structure",
        "maps_to": "power line",
    },
    {
        "canonical": "insulator",
        "prompts": [
            "insulator", "electrical insulator", "power insulator", "ceramic insulator",
            "composite insulator", "insulator on transmission tower", "small insulator",
            "hot insulator", "overheated insulator",
        ],
        "scene_hint": ["power_inspection"],
        "is_small_target_class": True,
        "structure_type": "small_power_object",
    },
    {
        "canonical": "animal",
        "prompts": ["animal", "thermal animal", "wild animal", "dog", "bear", "small animal", "distant animal"],
        "scene_hint": ["wild", "security", "road"],
        "is_small_target_class": True,
        "structure_type": "compact_object",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/prompt_bank_ir_v2.json")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "ir_prompt_bank_v2",
        "description": "Infrared open-vocabulary prompts for SAM3 pseudo-label generation.",
        "classes": PROMPT_BANK,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "classes": len(PROMPT_BANK), "prompts": sum(len(x["prompts"]) for x in PROMPT_BANK)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
