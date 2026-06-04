import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw

from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# =========================
# 基础工具
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_first(d, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def load_ir_as_rgb(image_path):
    img = Image.open(image_path)
    if img.mode == "RGB":
        return img.convert("RGB")
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    return img.convert("RGB")


def load_mask(mask_path):
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    mask = (mask > 127).astype(np.uint8)
    return mask


def union_masks(masks, image_size):
    # image_size: (W, H)
    if masks is None:
        return np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

    if isinstance(masks, torch.Tensor):
        if masks.shape[0] == 0:
            return np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
        masks = masks.detach().cpu().numpy()

    masks = np.asarray(masks)
    masks = np.squeeze(masks)

    if masks.ndim == 2:
        union = masks > 0
    elif masks.ndim == 3:
        union = np.any(masks > 0, axis=0)
    elif masks.ndim == 4:
        union = np.any(masks[:, 0] > 0, axis=0)
    else:
        raise RuntimeError(f"Unexpected masks shape: {masks.shape}")

    return union.astype(np.uint8)


def overlay_mask_on_image(image_pil, mask, color=(255, 80, 0), alpha=0.45):
    image = image_pil.convert("RGB")
    img_np = np.array(image).astype(np.float32)

    if mask.shape[:2] != img_np.shape[:2]:
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255).resize(image.size, Image.NEAREST)
        mask = (np.array(mask_img) > 127).astype(np.uint8)

    color_np = np.zeros_like(img_np)
    color_np[..., 0] = color[0]
    color_np[..., 1] = color[1]
    color_np[..., 2] = color[2]

    out = img_np.copy()
    mask_bool = mask > 0
    out[mask_bool] = out[mask_bool] * (1 - alpha) + color_np[mask_bool] * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def add_title_bar(img, title, bar_h=32):
    img = img.convert("RGB")
    w, h = img.size
    canvas = Image.new("RGB", (w, h + bar_h), (255, 255, 255))
    canvas.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, w, bar_h], fill=(240, 240, 240))
    draw.text((10, 8), title, fill=(0, 0, 0))
    return canvas


def build_triptych(orig, pseudo_overlay, pred_overlay, prompt, image_name):
    orig_show = add_title_bar(orig, f"Original | {image_name}")
    pseudo_show = add_title_bar(pseudo_overlay, f"SAM3 Pseudo | prompt={prompt}")
    pred_show = add_title_bar(pred_overlay, f"Student Pred | prompt={prompt}")

    w, h = orig_show.size
    gap = 8
    canvas = Image.new("RGB", (w * 3 + gap * 2, h), (255, 255, 255))
    canvas.paste(orig_show, (0, 0))
    canvas.paste(pseudo_show, (w + gap, 0))
    canvas.paste(pred_show, (2 * (w + gap), 0))
    return canvas


def infer_one(processor, image_pil, prompt):
    with torch.no_grad():
        state = processor.set_image(image_pil)
        state = processor.set_text_prompt(prompt=prompt, state=state)

    masks = state.get("masks", None)
    scores = state.get("scores", None)
    union = union_masks(masks, image_pil.size)
    return union, scores


# =========================
# 主函数
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_jsonl", type=str, required=True)
    parser.add_argument("--base_ckpt", type=str, required=True)
    parser.add_argument("--trained_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--num_save", type=int, default=20)
    parser.add_argument("--max_trials", type=int, default=400)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--confidence_threshold", type=float, default=0.05)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ensure_dir(args.out_dir)
    collage_dir = os.path.join(args.out_dir, "collages")
    ensure_dir(collage_dir)

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    print(f"[Info] device={device}")
    print(f"[Info] val_jsonl={args.val_jsonl}")
    print(f"[Info] base_ckpt={args.base_ckpt}")
    print(f"[Info] trained_ckpt={args.trained_ckpt}")
    print(f"[Info] out_dir={args.out_dir}")

    rows = load_jsonl(args.val_jsonl)
    print(f"[Info] total val rows={len(rows)}")

    # 构建模型
    model = build_efficientsam3_image_model(
        checkpoint_path=args.base_ckpt,
        device=device,
        eval_mode=True,
        enable_segmentation=True,
        backbone_type="efficientvit",
        model_name="b1",
        text_encoder_type="MobileCLIP-S1",
    )

    ckpt = torch.load(args.trained_ckpt, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Info] loaded trained ckpt: missing={len(missing)}, unexpected={len(unexpected)}")

    model.eval()
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=args.confidence_threshold,
    )

    candidate_indices = list(range(len(rows)))
    random.shuffle(candidate_indices)

    saved = 0
    tried = 0
    logs = []

    # 尝试更多次，直到凑够 20 张有效样本
    while saved < args.num_save and tried < args.max_trials:
        idx = candidate_indices[tried % len(candidate_indices)]
        tried += 1
        row = rows[idx]

        image_path = get_first(row, ["image", "image_path", "img_path", "img", "source_image"])
        mask_path = get_first(row, ["mask", "mask_path", "pseudo_mask", "pseudo_mask_path"])
        prompt = get_first(row, ["prompt", "text_prompt", "category", "label"])

        if image_path is None or mask_path is None or prompt is None:
            continue
        if not os.path.exists(image_path):
            continue
        if not os.path.exists(mask_path):
            continue

        # 读取原图与伪标签
        try:
            image_pil = load_ir_as_rgb(image_path)
            pseudo_mask = load_mask(mask_path)
        except Exception as e:
            print(f"[Skip] load fail: {e}")
            continue

        # 过滤空白伪标签
        if pseudo_mask.sum() == 0:
            continue

        # 学生模型推理
        try:
            pred_mask, scores = infer_one(processor, image_pil, prompt)
        except Exception as e:
            print(f"[Skip] infer fail: {e}")
            continue

        # 过滤空白预测
        if pred_mask.sum() == 0:
            continue

        # 生成可视化
        pseudo_overlay = overlay_mask_on_image(image_pil, pseudo_mask, color=(0, 170, 255), alpha=0.45)
        pred_overlay = overlay_mask_on_image(image_pil, pred_mask, color=(255, 80, 0), alpha=0.45)

        image_name = Path(image_path).name
        safe_prompt = str(prompt).replace("/", "_").replace(" ", "_")
        stem = Path(image_path).stem

        collage = build_triptych(
            orig=image_pil,
            pseudo_overlay=pseudo_overlay,
            pred_overlay=pred_overlay,
            prompt=prompt,
            image_name=image_name,
        )

        out_path = os.path.join(collage_dir, f"{saved:02d}_{stem}__{safe_prompt}.jpg")
        collage.save(out_path, quality=95)

        log_item = {
            "save_id": saved,
            "row_index": idx,
            "image_path": image_path,
            "mask_path": mask_path,
            "prompt": prompt,
            "pred_nonzero": int(pred_mask.sum()),
            "pseudo_nonzero": int(pseudo_mask.sum()),
            "scores": scores.detach().cpu().tolist() if isinstance(scores, torch.Tensor) else None,
            "out_path": out_path,
        }
        logs.append(log_item)

        print(f"[Saved {saved+1}/{args.num_save}] {out_path}")
        saved += 1

    # 保存日志
    log_json = os.path.join(args.out_dir, "selected_samples.json")
    with open(log_json, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    print(f"[Done] saved={saved}, tried={tried}")
    print(f"[Done] collages at: {collage_dir}")
    print(f"[Done] logs at: {log_json}")


if __name__ == "__main__":
    main()