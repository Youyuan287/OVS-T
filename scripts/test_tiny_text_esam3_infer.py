import argparse
import torch
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
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--resolution", type=int, default=1008)
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

    print(f"[Info] TinyText loaded from: {args.tiny_ckpt}", flush=True)

    # 删除原 language_backbone，改用 tiny text encoder 在线编码任意 prompt
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
            state = processor.set_image(img)
            state = processor.set_text_prompt(args.prompt, state)

    print("[Done]", flush=True)
    print("output keys:", list(state.keys()), flush=True)

    if "masks" in state:
        print("masks shape:", tuple(state["masks"].shape), flush=True)
        print("mask sum:", int(state["masks"].sum().item()), flush=True)
    else:
        print("masks: None", flush=True)

    if "scores" in state:
        print("scores shape:", tuple(state["scores"].shape), flush=True)
        print("scores first:", state["scores"][:10], flush=True)

    if device == "cuda":
        print("allocated MiB:", torch.cuda.memory_allocated() / 1024 / 1024, flush=True)
        print("reserved MiB:", torch.cuda.memory_reserved() / 1024 / 1024, flush=True)
        print("peak allocated MiB:", torch.cuda.max_memory_allocated() / 1024 / 1024, flush=True)


if __name__ == "__main__":
    main()
