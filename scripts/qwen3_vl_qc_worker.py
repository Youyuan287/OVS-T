#!/usr/bin/env python3
"""Single-panel Qwen3-VL quality critic worker.

This script is intentionally small because scripts/05_run_qwen8b_qc.py owns the
batch loop, fallback handling, and result normalization. The worker loads the
local Qwen model, critiques one panel, and prints JSON.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image


def extract_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"No JSON object found in model output tail: {text[-500:]}")


def clamp01(value, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


def normalize(raw: dict) -> dict:
    decision = str(raw.get("final_decision", raw.get("decision", "review"))).lower()
    if not re.search(r"^(accept|review|drop)$", decision):
        decision = "review"
    return {
        "semantic_match": clamp01(raw.get("semantic_match"), 0.5),
        "mask_coverage": clamp01(raw.get("mask_coverage"), 0.5),
        "background_leakage": clamp01(raw.get("background_leakage"), 0.3),
        "box_consistency": clamp01(raw.get("box_consistency"), 0.5),
        "context_consistency": clamp01(raw.get("context_consistency"), 0.5),
        "final_decision": decision,
        "reason": str(raw.get("reason", ""))[:500],
    }


def load_model(model_path: str):
    import importlib.util

    import torch
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = None
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
            break
        except Exception as exc:
            errors.append(f"{class_name}: {exc}")
    if model is None:
        raise RuntimeError("Failed to load Qwen3-VL model. " + " | ".join(errors[-3:]))
    model.eval()
    return torch, processor, model


def first_device(model):
    try:
        return next(model.parameters()).device
    except Exception:
        return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/Groups/group2/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--panel", required=True)
    parser.add_argument("--target_class", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    torch, processor, model = load_model(args.model)
    image = Image.open(Path(args.panel)).convert("RGB")
    rubric = args.prompt or "You are a strict pseudo-label quality critic for infrared segmentation."
    user_text = (
        f"{rubric}\n"
        f"Target class: {args.target_class}\n"
        "The panel contains original infrared image, mask overlay, crop/zoom, and masked region. "
        "Return JSON only with numeric 0-1 fields semantic_match, mask_coverage, "
        "background_leakage, box_consistency, context_consistency, plus final_decision "
        "(accept, review, or drop) and a short reason."
    )
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    device = first_device(model)
    try:
        inputs = inputs.to(device)
    except Exception:
        pass
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[-1]
    decoded = processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0]
    print(json.dumps(normalize(extract_json(decoded)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
