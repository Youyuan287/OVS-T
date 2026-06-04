import os
import json
import argparse
from pathlib import Path

import torch

from sam3.model_builder import build_efficientsam3_image_model


PROMPT_BANK = {
    "car": [
        "car", "vehicle", "automobile", "sedan", "SUV", "van", "taxi",
        "small car", "white car", "black car"
    ],
    "truck": [
        "truck", "lorry", "pickup truck", "heavy truck"
    ],
    "bus": [
        "bus", "coach"
    ],
    "person": [
        "person", "human", "pedestrian", "man", "woman", "people",
        "a person", "standing person", "walking person"
    ],
    "road": [
        "road", "street", "roadway", "lane", "path", "ground road"
    ],
    "tree": [
        "tree", "trees", "vegetation", "plant", "forest", "bush"
    ],
    "building": [
        "building", "house", "wall", "construction", "roof"
    ],
    "pole": [
        "pole", "utility pole", "electric pole", "power pole",
        "thin pole", "vertical pole"
    ],
    "insulator": [
        "insulator", "power insulator", "electric insulator",
        "transmission insulator"
    ],
    "animal": [
        "animal", "dog", "cat", "bird", "livestock"
    ],
    "wire": [
        "wire", "cable", "power line", "electric wire", "transmission line"
    ],
    "tower": [
        "tower", "power tower", "transmission tower", "pylon"
    ],
    "background": [
        "background"
    ],
}


def norm_text(s: str) -> str:
    return " ".join(
        s.lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
        .split()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", required=True)
    parser.add_argument("--trained_ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] device={device}")
    print(f"[Info] base_ckpt={args.base_ckpt}")
    print(f"[Info] trained_ckpt={args.trained_ckpt}")

    # 先构建完整模型，用完整 language_backbone 提取 prompt bank
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
    trained_sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(trained_sd, strict=False)
    print(f"[Info] loaded trained ckpt: missing={len(missing)}, unexpected={len(unexpected)}")

    model.eval()

    # 1. 导出 prompt bank：提前把常用 prompt 编码成 language_features 等特征
    bank_items = []
    alias_to_canonical = {}

    with torch.no_grad():
        for canonical, aliases in PROMPT_BANK.items():
            for text in aliases:
                text_norm = norm_text(text)
                alias_to_canonical[text_norm] = canonical

                text_outputs = model.backbone.forward_text([text], device=device)

                bank_items.append({
                    "canonical": canonical,
                    "text": text,
                    "text_norm": text_norm,
                    "outputs": {
                        k: v.detach().cpu()
                        for k, v in text_outputs.items()
                        if torch.is_tensor(v)
                    },
                })

    prompt_bank = {
        "items": bank_items,
        "alias_to_canonical": alias_to_canonical,
        "canonical_to_aliases": PROMPT_BANK,
    }

    prompt_bank_path = model_dir / "prompt_bank.pt"
    torch.save(prompt_bank, prompt_bank_path)

    prompt_alias_path = model_dir / "prompt_alias.json"
    with open(prompt_alias_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "alias_to_canonical": alias_to_canonical,
                "canonical_to_aliases": PROMPT_BANK,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 2. 导出去掉 language_backbone 的纯 FP32 state_dict
    # 注意：这里去掉的是 state_dict 中的 language_backbone 参数。
    # 后续推理时不能再调用 model.backbone.forward_text，而要从 prompt_bank 读预计算特征。
    pruned_sd = {}
    kept_params = 0
    removed_params = 0

    for k, v in trained_sd.items():
        if not torch.is_tensor(v):
            continue

        if k.startswith("backbone.language_backbone."):
            removed_params += v.numel()
            continue

        if torch.is_floating_point(v):
            pruned_sd[k] = v.detach().float().cpu()
        else:
            pruned_sd[k] = v.detach().cpu()

        kept_params += v.numel()

    sam3_path = model_dir / "sam3.pt"
    torch.save(pruned_sd, sam3_path)

    sam3_size = os.path.getsize(sam3_path) / 1024 / 1024
    bank_size = os.path.getsize(prompt_bank_path) / 1024 / 1024
    alias_size = os.path.getsize(prompt_alias_path) / 1024 / 1024

    print("[Done]")
    print(f"kept params: {kept_params:,}")
    print(f"removed language params: {removed_params:,}")
    print(f"model/sam3.pt size: {sam3_size:.2f} MiB")
    print(f"model/prompt_bank.pt size: {bank_size:.2f} MiB")
    print(f"model/prompt_alias.json size: {alias_size:.4f} MiB")
    print(f"saved to: {model_dir}")


if __name__ == "__main__":
    main()
