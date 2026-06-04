import os
import argparse
from pathlib import Path

import torch
from sam3.model_builder import build_efficientsam3_image_model


def unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--full_ckpt", required=True)
    parser.add_argument("--pruned_ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    print(f"[Info] device={device}")
    print(f"[Info] base_ckpt={args.base_ckpt}")
    print(f"[Info] full_ckpt={args.full_ckpt}")
    print(f"[Info] pruned_ckpt={args.pruned_ckpt}")

    # 1. 构建完整模型，包含 language_backbone
    model = build_efficientsam3_image_model(
        checkpoint_path=args.base_ckpt,
        device=device,
        eval_mode=True,
        enable_segmentation=True,
        backbone_type="efficientvit",
        model_name="b1",
        text_encoder_type="MobileCLIP-S1",
    )

    # 2. 加载之前完整 ESAM3 训练权重，保留 language_backbone
    full_ckpt = torch.load(args.full_ckpt, map_location=device)
    full_sd = unwrap_state_dict(full_ckpt)
    missing, unexpected = model.load_state_dict(full_sd, strict=False)
    print(f"[Info] load full ckpt: missing={len(missing)}, unexpected={len(unexpected)}")

    # 3. 叠加 pruned 训练出来的非 language 权重
    pruned_ckpt = torch.load(args.pruned_ckpt, map_location=device)
    pruned_sd = unwrap_state_dict(pruned_ckpt)

    cur_sd = model.state_dict()
    loaded = 0
    skipped = 0

    for k, v in pruned_sd.items():
        if k.startswith("backbone.language_backbone."):
            skipped += 1
            continue
        if k in cur_sd and cur_sd[k].shape == v.shape:
            cur_sd[k] = v.to(cur_sd[k].device).type_as(cur_sd[k])
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(cur_sd, strict=False)
    print(f"[Info] overlay pruned non-language weights: loaded={loaded}, skipped={skipped}")

    # 4. 保存完整 open 模型 state_dict
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    save_sd = {}
    for k, v in model.state_dict().items():
        if torch.is_tensor(v):
            save_sd[k] = v.detach().float().cpu() if torch.is_floating_point(v) else v.detach().cpu()

    torch.save(save_sd, out)

    params = sum(v.numel() for v in save_sd.values() if torch.is_tensor(v))
    size_mb = os.path.getsize(out) / 1024 / 1024

    print("[Done]")
    print(f"params: {params:,}")
    print(f"file size: {size_mb:.2f} MiB")
    print(f"saved to: {out}")


if __name__ == "__main__":
    main()
