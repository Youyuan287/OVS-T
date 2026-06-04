import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def load_ir_as_rgb(image_path: str) -> Image.Image:
    img = Image.open(image_path)

    # 兼容灰度红外、16-bit红外、伪彩色红外
    if img.mode == "RGB":
        return img.convert("RGB")

    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.convert("RGB")
    return img


def to_numpy_mask(masks):
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().float().numpy()

    masks = np.asarray(masks)
    masks = np.squeeze(masks)

    # 多个 mask 时默认取第一个
    if masks.ndim == 3:
        masks = masks[0]

    if masks.ndim != 2:
        raise RuntimeError(f"Unexpected mask shape after squeeze: {masks.shape}")

    if masks.max() > 1:
        masks = masks / (masks.max() + 1e-6)

    return masks.astype(np.float32)


def save_overlay(image: Image.Image, mask: np.ndarray, out_path: str, threshold: float = 0.5):
    image = image.convert("RGB")
    mask_bin = mask > threshold

    image_np = np.array(image).astype(np.float32)
    color = np.zeros_like(image_np)
    color[..., 0] = 255
    color[..., 1] = 80
    color[..., 2] = 0

    alpha = 0.45
    overlay = image_np.copy()
    overlay[mask_bin] = overlay[mask_bin] * (1 - alpha) + color[mask_bin] * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    Image.fromarray(overlay).save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input IR image")
    parser.add_argument("--prompt", required=True, help="Text prompt, e.g. person / car / vehicle")
    parser.add_argument("--ckpt", required=True, help="Path to EfficientSAM3 checkpoint")
    parser.add_argument("--out_dir", default="outputs/esam3_test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    print(f"[INFO] device: {device}")
    print(f"[INFO] checkpoint: {args.ckpt}")
    print(f"[INFO] image: {args.image}")
    print(f"[INFO] prompt: {args.prompt}")

    image = load_ir_as_rgb(args.image)

    model = build_efficientsam3_image_model(
        checkpoint_path=args.ckpt,
        device=device,
        eval_mode=True,
        enable_segmentation=True,
        backbone_type="efficientvit",
        model_name="b1",
        text_encoder_type="MobileCLIP-S1",
    )

    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=args.threshold,
    )

    with torch.no_grad():
        state = processor.set_image(image)
        state = processor.set_text_prompt(prompt=args.prompt, state=state)

    print("[INFO] output keys:", list(state.keys()))

    masks = state.get("masks", None)
    scores = state.get("scores", None)

    if masks is None:
        raise RuntimeError("No masks found in output state.")

    if isinstance(masks, torch.Tensor):
        print("[INFO] masks shape:", tuple(masks.shape))
    else:
        print("[INFO] masks type:", type(masks))

    if scores is not None:
        print("[INFO] scores:", scores)

    mask = to_numpy_mask(masks)

    stem = Path(args.image).stem
    safe_prompt = args.prompt.replace(" ", "_").replace("/", "_")

    mask_path = os.path.join(args.out_dir, f"{stem}_{safe_prompt}_mask.png")
    overlay_path = os.path.join(args.out_dir, f"{stem}_{safe_prompt}_overlay.png")

    Image.fromarray(((mask > args.threshold).astype(np.uint8) * 255)).save(mask_path)
    save_overlay(image, mask, overlay_path, threshold=args.threshold)

    print(f"[DONE] mask saved to: {mask_path}")
    print(f"[DONE] overlay saved to: {overlay_path}")


if __name__ == "__main__":
    main()
