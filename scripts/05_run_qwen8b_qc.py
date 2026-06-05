#!/usr/bin/env python3
"""Run Qwen3-VL-8B quality critique or a deterministic fallback scorer."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable

from PIL import Image


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


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


def clamp01(x: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return default


def normalize_qc(raw: Dict[str, Any] | None, task: Dict[str, Any], fallback: bool) -> Dict[str, Any]:
    if raw is None:
        raw = {}
    area = clamp01(task.get("area_ratio", 0.0), 0.0)
    sam_score = clamp01(task.get("sam3_score", 0.0), 0.0)
    if fallback:
        semantic = 0.55 if sam_score > 0.0 and area > 0.0 else 0.20
        coverage = 0.55 if 0.0001 <= area <= 0.70 else 0.25
        leakage = 0.20 if area <= 0.70 else 0.70
        context = 0.50
        decision = "review" if semantic >= 0.5 else "drop"
        reason = "rule fallback; Qwen command was not used or failed"
    else:
        semantic = clamp01(raw.get("semantic_match"), 0.5)
        coverage = clamp01(raw.get("mask_coverage"), 0.5)
        leakage = clamp01(raw.get("background_leakage"), 0.3)
        context = clamp01(raw.get("context_consistency"), 0.5)
        decision = str(raw.get("final_decision", "review"))
        reason = str(raw.get("reason", ""))
    return {
        "candidate_id": task["candidate_id"],
        "target_class": task["target_class"],
        "scene_dir": task.get("scene_dir", ""),
        "scene_type": task.get("scene_type", ""),
        "panel": task["panel"],
        "semantic_match": semantic,
        "mask_coverage": coverage,
        "background_leakage": leakage,
        "box_consistency": clamp01(raw.get("box_consistency"), 0.5),
        "context_consistency": context,
        "final_decision": decision,
        "reason": reason,
        "qwen_raw": raw,
        "used_rule_fallback": fallback,
    }


def run_qwen(command_template: str, task: Dict[str, Any], timeout: int) -> tuple[Dict[str, Any] | None, str]:
    prompt = (
        "You are a strict pseudo-label quality critic for infrared segmentation. "
        "Given the panel image, target class and highlighted mask, return JSON only with: "
        "semantic_match, mask_coverage, background_leakage, box_consistency, "
        "context_consistency, final_decision, reason. Scores must be 0-1."
    )
    command = command_template.format(panel=task["panel"], target_class=task["target_class"], prompt=prompt)
    proc = subprocess.run(shlex.split(command), text=True, capture_output=True, timeout=timeout)
    text = proc.stdout if proc.returncode == 0 else proc.stdout + "\n" + proc.stderr
    return extract_json(text), text[-4000:]


def load_local_qwen(model_path: str):
    import importlib.util

    import torch
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    errors = []
    has_accelerate = importlib.util.find_spec("accelerate") is not None
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    for class_name in ("AutoModelForImageTextToText", "Qwen3VLForConditionalGeneration", "AutoModelForVision2Seq"):
        try:
            module = __import__("transformers", fromlist=[class_name])
            cls = getattr(module, class_name)
            kwargs = {"trust_remote_code": True}
            if has_accelerate and torch.cuda.is_available():
                kwargs["device_map"] = "auto"
            try:
                model = cls.from_pretrained(model_path, dtype=dtype, **kwargs)
            except TypeError:
                model = cls.from_pretrained(model_path, torch_dtype=dtype, **kwargs)
            if not has_accelerate and torch.cuda.is_available():
                model = model.to("cuda")
            model.eval()
            return torch, processor, model
        except Exception as exc:
            errors.append(f"{class_name}: {exc}")
    raise RuntimeError("Failed to load local Qwen model. " + " | ".join(errors[-3:]))


def first_device(model):
    try:
        return next(model.parameters()).device
    except Exception:
        return "cpu"


def run_local_qwen(bundle, task: Dict[str, Any], max_new_tokens: int) -> tuple[Dict[str, Any] | None, str]:
    torch, processor, model = bundle
    image = Image.open(task["panel"]).convert("RGB")
    prompt = (
        "You are a strict pseudo-label quality critic for infrared segmentation. "
        f"Target class: {task['target_class']}. "
        "The panel contains original infrared image, mask overlay, crop/zoom, and masked region. "
        "Return JSON only with numeric 0-1 fields semantic_match, mask_coverage, "
        "background_leakage, box_consistency, context_consistency, plus final_decision "
        "(accept, review, or drop) and a short reason."
    )
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    try:
        inputs = inputs.to(first_device(model))
    except Exception:
        pass
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[-1]
    decoded = processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0]
    return extract_json(decoded), decoded[-4000:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out_jsonl", default="/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_prompt_irgpt_sam3_qwen8b/qwen_qc_results.jsonl")
    parser.add_argument("--qwen_command", default="", help="Command template with {panel}, {target_class}, {prompt}; stdout should be JSON.")
    parser.add_argument("--local_qwen_model", default="", help="Load this local Qwen3-VL model once and run all tasks in-process.")
    parser.add_argument("--rule_fallback", action="store_true")
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"tasks": 0, "qwen_ok": 0, "fallback": 0, "parse_failed": 0}
    local_bundle = load_local_qwen(args.local_qwen_model) if args.local_qwen_model else None
    with out_path.open("w", encoding="utf-8") as f:
        for task in read_jsonl(Path(args.tasks)):
            if args.max_items and stats["tasks"] >= args.max_items:
                break
            stats["tasks"] += 1
            raw = None
            raw_text = ""
            fallback = True
            if local_bundle is not None:
                try:
                    raw, raw_text = run_local_qwen(local_bundle, task, args.max_new_tokens)
                    fallback = raw is None
                    if raw is None:
                        stats["parse_failed"] += 1
                    else:
                        stats["qwen_ok"] += 1
                except Exception as exc:
                    raw_text = repr(exc)
                    fallback = True
            elif args.qwen_command:
                try:
                    raw, raw_text = run_qwen(args.qwen_command, task, args.timeout)
                    fallback = raw is None
                    if raw is None:
                        stats["parse_failed"] += 1
                    else:
                        stats["qwen_ok"] += 1
                except Exception as exc:
                    raw_text = repr(exc)
                    fallback = True
            if fallback:
                if not args.rule_fallback:
                    stats["parse_failed"] += 1
                    continue
                stats["fallback"] += 1
            row = normalize_qc(raw, task, fallback)
            row["raw_text_tail"] = raw_text
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if stats["tasks"] % 50 == 0:
                print(json.dumps(stats, ensure_ascii=False), flush=True)

    summary = out_path.with_name("qwen_qc_results_summary.json")
    summary.write_text(json.dumps({**stats, "out_jsonl": str(out_path)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**stats, "out_jsonl": str(out_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
