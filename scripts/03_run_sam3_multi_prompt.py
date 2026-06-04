#!/usr/bin/env python3
"""Generate SAM3/EfficientSAM3 candidate masks from IRGPT proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from PIL import Image, ImageOps


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_prompt_bank(path: Path) -> Dict[str, List[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    classes = data["classes"] if isinstance(data, dict) and "classes" in data else data
    out = {}
    for item in classes:
        canonical = item.get("maps_to", item["canonical"])
        out.setdefault(canonical, [])
        for p in item.get("prompts", [canonical]):
            if p not in out[canonical]:
                out[canonical].append(p)
    return out


def safe_id(*parts: Any) -> str:
    text = "|".join(str(x) for x in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_ir_as_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = ImageOps.autocontrast(img.convert("L")).convert("RGB")
    return img.convert("RGB")


def clamp_box(box: Any, w: int, h: int, pad_ratio: float = 0.08) -> tuple[int, int, int, int] | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(x) for x in box]
    if x2 <= x1 or y2 <= y1:
        return None
    bw, bh = x2 - x1, y2 - y1
    pad = max(bw, bh) * pad_ratio
    x1 = max(0, int(math.floor(x1 - pad)))
    y1 = max(0, int(math.floor(y1 - pad)))
    x2 = min(w, int(math.ceil(x2 + pad)))
    y2 = min(h, int(math.ceil(y2 + pad)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def mask_metrics(mask: np.ndarray) -> Dict[str, Any]:
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape
    area = int(mask.sum())
    return {
        "area": area,
        "area_ratio": float(area / max(1, h * w)),
        "height": h,
        "width": w,
    }


def paste_crop_mask(crop_mask: np.ndarray, full_size: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    full_w, full_h = full_size
    x1, y1, x2, y2 = box
    out = np.zeros((full_h, full_w), dtype=np.uint8)
    crop_img = Image.fromarray((crop_mask > 0).astype(np.uint8) * 255)
    crop_img = crop_img.resize((x2 - x1, y2 - y1), Image.Resampling.NEAREST)
    out[y1:y2, x1:x2] = (np.array(crop_img) > 0).astype(np.uint8)
    return out


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(path)


def save_overlay(image: Image.Image, mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.array(image.convert("RGB")).astype(np.float32)
    m = mask > 0
    color = np.zeros_like(img)
    color[..., 0] = 255
    color[..., 1] = 80
    alpha = 0.45
    img[m] = img[m] * (1 - alpha) + color[m] * alpha
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path)


def split_submission_weight(raw, tiny_fallback_path: str = ""):
    if not isinstance(raw, (dict, OrderedDict)):
        raise TypeError(f"Unsupported weight type: {type(raw)}")

    if "esam3_model" in raw and "tiny_text" in raw:
        return OrderedDict(raw["esam3_model"]), OrderedDict(raw["tiny_text"])

    keys = [str(k) for k in raw.keys()]
    if any(k.startswith("esam3_model.") for k in keys) and any(k.startswith("tiny_text.") for k in keys):
        esam3_sd = OrderedDict()
        tiny_sd = OrderedDict()
        for k, v in raw.items():
            k = str(k)
            if k.startswith("esam3_model."):
                esam3_sd[k.replace("esam3_model.", "", 1)] = v
            elif k.startswith("tiny_text."):
                tiny_sd[k.replace("tiny_text.", "", 1)] = v
        return esam3_sd, tiny_sd

    if tiny_fallback_path:
        import torch

        tiny_raw = torch.load(tiny_fallback_path, map_location="cpu")
        if isinstance(tiny_raw, dict) and "model" in tiny_raw:
            tiny_raw = tiny_raw["model"]
        return OrderedDict(raw), OrderedDict(tiny_raw)

    raise ValueError(
        "Cannot split checkpoint. Use --submission_ckpt for prefixed combined "
        "weights, or provide --esam3_ckpt and --tiny_ckpt."
    )


def build_model(args):
    import torch
    from torchvision.transforms import v2
    from sam3.model_builder import build_efficientsam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from tiny_text_encoder_esam3 import TinyTextEncoderESAM3

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    model = build_efficientsam3_image_model(
        checkpoint_path=None,
        device=device,
        eval_mode=True,
        enable_segmentation=True,
        backbone_type="efficientvit",
        model_name="b1",
        text_encoder_type="MobileCLIP-S1",
    )
    tiny = TinyTextEncoderESAM3(
        bpe_path=args.bpe_path,
        token_dim=192,
        hidden_dim=256,
        num_layers=3,
        num_heads=6,
    ).to(device)

    if args.submission_ckpt:
        raw = torch.load(args.submission_ckpt, map_location="cpu")
        esam3_sd, tiny_sd = split_submission_weight(raw)
    else:
        raw = torch.load(args.esam3_ckpt, map_location="cpu")
        esam3_sd, tiny_sd = split_submission_weight(raw, args.tiny_ckpt)

    missing, unexpected = model.load_state_dict(esam3_sd, strict=False)
    bad_missing = [x for x in missing if "language_backbone" not in x]
    if bad_missing:
        raise RuntimeError(f"ESAM3 weight mismatch, bad missing examples: {bad_missing[:20]}")
    if unexpected:
        print(f"[Load] ESAM3 unexpected examples: {unexpected[:20]}", flush=True)

    tiny_missing, tiny_unexpected = tiny.load_state_dict(tiny_sd, strict=False)
    if tiny_missing or tiny_unexpected:
        raise RuntimeError(
            f"TinyText mismatch: missing={tiny_missing[:20]}, "
            f"unexpected={tiny_unexpected[:20]}"
        )
    try:
        model.backbone.language_backbone = None
    except Exception:
        pass
    model.backbone.forward_text = tiny.forward_text
    model.eval()
    tiny.eval()
    processor = Sam3Processor(model, resolution=args.resolution, device=device, confidence_threshold=args.threshold)
    return torch, v2, model, processor, device


def predict_mask(torch, v2, model, processor, device: str, image: Image.Image, prompt: str, threshold: float) -> tuple[np.ndarray, float]:
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            state = processor.set_image(image)
            state = processor.set_text_prompt(prompt, state)
    masks = state.get("masks")
    scores = state.get("scores")
    if masks is None:
        return np.zeros((image.height, image.width), dtype=np.uint8), 0.0
    mask_np = masks.detach().cpu().float().numpy()
    mask_np = np.squeeze(mask_np)
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    if mask_np.ndim != 2:
        raise RuntimeError(f"Unexpected SAM3 mask shape after squeeze: {mask_np.shape}")
    score = float(scores[0].detach().cpu().item()) if scores is not None and len(scores) else 0.0
    return (mask_np > threshold).astype(np.uint8), score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--prompt_bank", default="data/prompt_bank_ir_v2.json")
    parser.add_argument("--out_dir", default="/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_prompt_irgpt_sam3_qwen8b")
    parser.add_argument("--submission_ckpt", default="", help="Combined submission checkpoint with esam3_model./tiny_text. prefixes.")
    parser.add_argument("--esam3_ckpt", default="")
    parser.add_argument("--tiny_ckpt", default="")
    parser.add_argument("--bpe_path", default="sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--prompts_per_class", type=int, default=5)
    parser.add_argument("--modes", default="text_only,box_crop_text,crop_text")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    mask_dir = out_dir / "sam3_masks"
    overlay_dir = out_dir / "sam3_overlays"
    out_jsonl = out_dir / "sam3_candidates.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_bank = load_prompt_bank(Path(args.prompt_bank))
    proposals = list(read_jsonl(Path(args.proposals)))
    if args.max_items > 0:
        proposals = proposals[: args.max_items]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    model_bundle = None
    if not args.dry_run:
        if not args.submission_ckpt and (not args.esam3_ckpt or not args.tiny_ckpt):
            raise ValueError(
                "Use --submission_ckpt, or provide --esam3_ckpt and --tiny_ckpt, "
                "unless --dry_run is used"
            )
        sys.path.insert(0, str(Path.cwd()))
        model_bundle = build_model(args)

    stats = {"proposals": len(proposals), "candidates": 0, "dry_run": bool(args.dry_run), "non_empty": 0}
    with out_jsonl.open("w", encoding="utf-8") as f:
        for prop in proposals:
            image_path = Path(prop["image"])
            canonical = str(prop.get("canonical", "")).lower()
            prompts = [prop.get("prompt") or canonical]
            prompts += prompt_bank.get(canonical, [])[: args.prompts_per_class]
            prompts = list(dict.fromkeys([p for p in prompts if p]))
            image = load_ir_as_rgb(image_path)
            full_size = image.size
            box = clamp_box(prop.get("bbox"), image.width, image.height)
            for prompt in prompts:
                for mode in modes:
                    if mode != "text_only" and box is None:
                        continue
                    cid = safe_id(image_path, canonical, prompt, mode, prop.get("bbox"))
                    mask_path = mask_dir / canonical / f"{image_path.stem}_{cid}.png"
                    overlay_path = overlay_dir / canonical / f"{image_path.stem}_{cid}.jpg"
                    if args.dry_run:
                        mask = np.zeros((image.height, image.width), dtype=np.uint8)
                        score = 0.0
                    else:
                        torch, v2, model, processor, device = model_bundle
                        if mode == "text_only":
                            mask, score = predict_mask(torch, v2, model, processor, device, image, prompt, args.threshold)
                        else:
                            x1, y1, x2, y2 = box
                            crop = image.crop((x1, y1, x2, y2))
                            crop_mask, score = predict_mask(torch, v2, model, processor, device, crop, prompt, args.threshold)
                            mask = paste_crop_mask(crop_mask, full_size, box)
                        save_mask(mask, mask_path)
                        save_overlay(image, mask, overlay_path)
                    metrics = mask_metrics(mask)
                    if metrics["area"] > 0:
                        stats["non_empty"] += 1
                    row = {
                        "candidate_id": cid,
                        "image": str(image_path),
                        "canonical_prompt": canonical,
                        "raw_prompt": prompt,
                        "source_mode": mode,
                        "proposal_confidence": prop.get("confidence", 0.0),
                        "bbox": prop.get("bbox"),
                        "sam3_score": score,
                        "mask": str(mask_path),
                        "overlay": str(overlay_path),
                        **metrics,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats["candidates"] += 1

    summary = out_dir / "sam3_candidates_summary.json"
    summary.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**stats, "out_jsonl": str(out_jsonl)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
