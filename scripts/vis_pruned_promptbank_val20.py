import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw
from torchvision.transforms import v2

from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def norm_text(s):
    return " ".join(
        str(s).lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
        .split()
    )


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_ir_as_rgb(path):
    img = Image.open(path)
    if img.mode == "RGB":
        return img.convert("RGB")
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    return img.convert("RGB")


def load_mask(path):
    m = Image.open(path).convert("L")
    return (np.array(m) > 127).astype(np.uint8)


def overlay(image, mask, color=(255, 80, 0), alpha=0.45):
    image = image.convert("RGB")
    img = np.array(image).astype(np.float32)

    if mask.shape[:2] != img.shape[:2]:
        mask = Image.fromarray(mask.astype(np.uint8) * 255).resize(image.size, Image.NEAREST)
        mask = (np.array(mask) > 127).astype(np.uint8)

    c = np.zeros_like(img)
    c[..., 0] = color[0]
    c[..., 1] = color[1]
    c[..., 2] = color[2]

    out = img.copy()
    mb = mask > 0
    out[mb] = out[mb] * (1 - alpha) + c[mb] * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def add_title(img, title, h=34):
    img = img.convert("RGB")
    w, ih = img.size
    canvas = Image.new("RGB", (w, ih + h), (255, 255, 255))
    canvas.paste(img, (0, h))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, w, h], fill=(240, 240, 240))
    d.text((10, 9), title, fill=(0, 0, 0))
    return canvas


def triptych(orig, pseudo, pred, prompt, image_name):
    a = add_title(orig, f"Original | {image_name}")
    b = add_title(pseudo, f"SAM3 Pseudo | {prompt}")
    c = add_title(pred, f"Pruned Student | {prompt}")

    w, h = a.size
    gap = 8
    out = Image.new("RGB", (w * 3 + gap * 2, h), (255, 255, 255))
    out.paste(a, (0, 0))
    out.paste(b, (w + gap, 0))
    out.paste(c, (2 * (w + gap), 0))
    return out


def choose_prompt_bank_item(prompt, prompt_bank):
    prompt_norm = norm_text(prompt)
    items = prompt_bank["items"]
    alias_to_canonical = prompt_bank.get("alias_to_canonical", {})

    # 1. 完全匹配别名
    if prompt_norm in alias_to_canonical:
        canonical = alias_to_canonical[prompt_norm]
        for item in items:
            if item["canonical"] == canonical:
                return item

    # 2. prompt 中包含某个 alias
    for alias, canonical in alias_to_canonical.items():
        if alias and alias in prompt_norm:
            for item in items:
                if item["canonical"] == canonical:
                    return item

    # 3. 某个 alias 包含 prompt
    for alias, canonical in alias_to_canonical.items():
        if prompt_norm and prompt_norm in alias:
            for item in items:
                if item["canonical"] == canonical:
                    return item

    return None


def encode_image_once(model, processor, image_pil, device):
    x = processor.transform(v2.functional.to_image(image_pil).to(device))
    x = x.unsqueeze(0)
    with torch.no_grad():
        return model.backbone.forward_image(x)


def infer_promptbank(model, processor, image_pil, prompt, prompt_bank, device, threshold=0.5):
    item = choose_prompt_bank_item(prompt, prompt_bank)
    if item is None:
        return np.zeros((image_pil.height, image_pil.width), dtype=np.uint8), None

    backbone_out = encode_image_once(model, processor, image_pil, device)

    text_outputs = {}
    for k, v in item["outputs"].items():
        text_outputs[k] = v.to(device)

    backbone_out.update(text_outputs)

    geometric_prompt = model._get_dummy_prompt()

    with torch.no_grad():
        out = model.forward_grounding(
            backbone_out=backbone_out,
            find_input=processor.find_stage,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )

    pred_masks = out.get("pred_masks", None)
    pred_logits = out.get("pred_logits", None)
    presence = out.get("presence_logit_dec", None)

    if pred_masks is None:
        return np.zeros((image_pil.height, image_pil.width), dtype=np.uint8), item

    # pred_masks: [B,Q,H,W]
    masks = pred_masks[0]
    if pred_logits is not None:
        scores = pred_logits.sigmoid()[0].squeeze(-1)
        if presence is not None:
            scores = scores * presence.sigmoid().view(-1)[0]
        keep = scores > threshold
        masks = masks[keep]

    if masks.numel() == 0 or masks.shape[0] == 0:
        return np.zeros((image_pil.height, image_pil.width), dtype=np.uint8), item

    masks = torch.nn.functional.interpolate(
        masks[:, None, :, :],
        size=(image_pil.height, image_pil.width),
        mode="bilinear",
        align_corners=False,
    ).sigmoid()

    union = (masks > 0.5).any(dim=0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return union, item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--pruned_ckpt", required=True)
    parser.add_argument("--prompt_bank", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_save", type=int, default=20)
    parser.add_argument("--max_trials", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    collage_dir = os.path.join(args.out_dir, "collages")
    os.makedirs(collage_dir, exist_ok=True)

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    rows = load_jsonl(args.val_jsonl)
    random.shuffle(rows)

    print(f"[Info] rows={len(rows)}")
    print(f"[Info] device={device}")
    print(f"[Info] pruned_ckpt={args.pruned_ckpt}")
    print(f"[Info] prompt_bank={args.prompt_bank}")

    model = build_efficientsam3_image_model(
        checkpoint_path=None,
        device=device,
        eval_mode=True,
        enable_segmentation=True,
        backbone_type="efficientvit",
        model_name="b1",
        text_encoder_type="MobileCLIP-S1",
    )

    sd = torch.load(args.pruned_ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[Info] loaded pruned ckpt: missing={len(missing)}, unexpected={len(unexpected)}")

    # 不再使用 language_backbone
    try:
        model.backbone.language_backbone = None
        print("[Info] language_backbone set to None")
    except Exception as e:
        print(f"[Warn] failed to remove language_backbone: {e}")

    model.eval()

    prompt_bank = torch.load(args.prompt_bank, map_location="cpu")

    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=args.threshold,
    )

    saved = 0
    tried = 0
    logs = []

    for row in rows:
        if saved >= args.num_save or tried >= args.max_trials:
            break
        tried += 1

        image_path = row["image"]
        mask_path = row["mask"]
        prompt = row["prompt"]

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            continue

        image = load_ir_as_rgb(image_path)
        pseudo_mask = load_mask(mask_path)
        if pseudo_mask.sum() == 0:
            continue

        pred_mask, matched_item = infer_promptbank(
            model=model,
            processor=processor,
            image_pil=image,
            prompt=prompt,
            prompt_bank=prompt_bank,
            device=device,
            threshold=args.threshold,
        )

        if pred_mask.sum() == 0:
            continue

        pseudo_overlay = overlay(image, pseudo_mask, color=(0, 170, 255), alpha=0.45)
        pred_overlay = overlay(image, pred_mask, color=(255, 80, 0), alpha=0.45)

        stem = Path(image_path).stem
        safe_prompt = str(prompt).replace("/", "_").replace(" ", "_")
        out_path = os.path.join(collage_dir, f"{saved:02d}_{stem}__{safe_prompt}.jpg")

        canvas = triptych(image, pseudo_overlay, pred_overlay, prompt, Path(image_path).name)
        canvas.save(out_path, quality=95)

        logs.append({
            "image": image_path,
            "mask": mask_path,
            "prompt": prompt,
            "matched_prompt": None if matched_item is None else matched_item["text"],
            "canonical": None if matched_item is None else matched_item["canonical"],
            "pseudo_nonzero": int(pseudo_mask.sum()),
            "pred_nonzero": int(pred_mask.sum()),
            "out": out_path,
        })

        print(f"[Saved {saved+1}/{args.num_save}] {out_path}")
        saved += 1

    with open(os.path.join(args.out_dir, "selected_samples.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    print(f"[Done] saved={saved}, tried={tried}")
    print(f"[Done] collages={collage_dir}")


if __name__ == "__main__":
    main()
