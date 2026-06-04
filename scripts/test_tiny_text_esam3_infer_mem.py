import argparse
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from tiny_text_encoder_esam3 import TinyTextEncoderESAM3


def load_ir_as_rgb(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = ImageOps.autocontrast(img.convert("L")).convert("RGB")
    else:
        img = img.convert("RGB")
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--esam3_ckpt", required=True)
    parser.add_argument("--tiny_ckpt", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="car")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    print(f"[Info] device={device}", flush=True)

    model = build_efficientsam3_image_model(
        checkpoint_path=None,
        device=device,
        eval_mode=True,
        enable_segmentation=True,
        backbone_type="efficientvit",
        model_name="b1",
        text_encoder_type="MobileCLIP-S1",
    )

    sd = torch.load(args.esam3_ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[Info] ESAM3 loaded: missing={len(missing)}, unexpected={len(unexpected)}", flush=True)

    tiny = TinyTextEncoderESAM3(
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        token_dim=192,
        hidden_dim=256,
        num_layers=3,
        num_heads=6,
    ).to(device)

    tiny_ckpt = torch.load(args.tiny_ckpt, map_location=device)
    tiny.load_state_dict(tiny_ckpt["model"], strict=True)
    tiny.eval()

    # 替换文本分支：保留在线编码能力，不再用 prompt bank
    try:
        model.backbone.language_backbone = None
    except Exception:
        pass
    model.backbone.forward_text = tiny.forward_text

    model.eval()

    processor = Sam3Processor(
        model,
        resolution=args.resolution,
        device=device,
        confidence_threshold=args.threshold,
    )

    img = load_ir_as_rgb(args.image)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            # 1. 一张图只跑一次 image encoder
            state = processor.set_image(img)

            # 2. 文本由 TinyTextEncoder 实时编码
            text_outputs = model.backbone.forward_text([args.prompt], device=device)
            state["backbone_out"].update(text_outputs)

            if "geometric_prompt" not in state:
                state["geometric_prompt"] = model._get_dummy_prompt()

            # 3. 直接调用 grounding，不走 processor.set_text_prompt 的全量后处理
            out = model.forward_grounding(
                backbone_out=state["backbone_out"],
                find_input=processor.find_stage,
                geometric_prompt=state["geometric_prompt"],
                find_target=None,
            )

            pred_masks = out["pred_masks"]       # [B, Q, h, w]
            pred_logits = out["pred_logits"]     # [B, Q, 1]
            presence = out.get("presence_logit_dec", None)

            scores = pred_logits.sigmoid()
            if presence is not None:
                scores = scores * presence.sigmoid().unsqueeze(1)
            scores = scores.squeeze(-1)[0]       # [Q]

            # 4. 只保留 threshold 以上的 top-k mask
            keep = scores > args.threshold
            keep_idx = torch.nonzero(keep, as_tuple=False).view(-1)

            if keep_idx.numel() > args.topk:
                top_scores, top_pos = scores[keep_idx].topk(args.topk)
                keep_idx = keep_idx[top_pos]

            print("[Info] raw candidates:", int(scores.numel()), flush=True)
            print("[Info] kept candidates:", int(keep_idx.numel()), flush=True)

            img_w, img_h = img.size

            if keep_idx.numel() == 0:
                union = torch.zeros((img_h, img_w), dtype=torch.bool, device=device)
            else:
                selected = pred_masks[0, keep_idx]  # [K,h,w]
                selected = F.interpolate(
                    selected[:, None, :, :],
                    size=(img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid()
                union = (selected > 0.5).any(dim=0).squeeze(0)

            mask_sum = int(union.sum().item())

            # 主动释放大中间变量
            del out, pred_masks, pred_logits, scores
            if device == "cuda":
                torch.cuda.empty_cache()

    print("[Done]", flush=True)
    print("mask sum:", mask_sum, flush=True)

    if device == "cuda":
        print("allocated MiB:", torch.cuda.memory_allocated() / 1024 / 1024, flush=True)
        print("reserved MiB:", torch.cuda.memory_reserved() / 1024 / 1024, flush=True)
        print("peak allocated MiB:", torch.cuda.max_memory_allocated() / 1024 / 1024, flush=True)


if __name__ == "__main__":
    main()
