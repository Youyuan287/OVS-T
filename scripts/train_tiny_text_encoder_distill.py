import os
import json
import time
import random
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sam3.model_builder import build_efficientsam3_image_model
from tiny_text_encoder_esam3 import TinyTextEncoderESAM3


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_prompt_pool():
    classes = [
        "car", "vehicle", "automobile", "sedan", "suv", "van", "truck", "bus",
        "person", "pedestrian", "human", "worker", "man", "woman",
        "road", "street", "road surface", "lane", "highway",
        "tree", "vegetation", "bush", "forest", "plant",
        "pole", "electric pole", "utility pole", "power pole",
        "insulator", "power insulator", "electrical insulator",
        "tower", "transmission tower", "electric tower", "pylon",
        "wire", "power line", "transmission line", "cable", "conductor",
        "building", "house", "wall", "roof",
        "animal", "dog", "cat", "livestock",
        "background object", "obstacle", "thermal target", "infrared target",
    ]

    templates = [
        "{}",
        "a {}",
        "the {}",
        "segment the {}",
        "find the {}",
        "the {} in the image",
        "a {} in thermal image",
        "a {} in infrared image",
        "an infrared scene containing {}",
        "a thermal scene containing {}",
        "small {}",
        "large {}",
        "distant {}",
        "nearby {}",
        "{} target",
        "{} object",
        "{} region",
    ]

    prompts = []
    for c in classes:
        for t in templates:
            prompts.append(t.format(c))

    extra = [
        "a pedestrian walking on the road",
        "a car parked beside the road",
        "a utility pole beside the street",
        "power lines above the road",
        "transmission tower in infrared image",
        "insulators on the power tower",
        "a vehicle in night vision",
        "an animal in thermal camera",
        "trees near the power line",
        "buildings in the background",
    ]
    prompts.extend(extra)

    # 去重但保持顺序
    seen = set()
    out = []
    for p in prompts:
        q = " ".join(p.lower().split())
        if q not in seen:
            seen.add(q)
            out.append(p)

    return out


class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = list(prompts)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def collate_fn(batch):
    return list(batch)


def masked_mse(student, teacher, mask):
    # student/teacher: [L, B, C]
    # mask: [B, L], True means padding
    valid = (~mask).transpose(0, 1).unsqueeze(-1).to(student.dtype)  # [L,B,1]
    diff = (student - teacher) ** 2
    denom = valid.sum() * student.shape[-1] + 1e-6
    return (diff * valid).sum() / denom


def masked_cos_loss(student, teacher, mask):
    valid = (~mask).transpose(0, 1)  # [L,B]
    if valid.sum().item() == 0:
        return student.sum() * 0.0

    s = student[valid]
    t = teacher[valid]
    s = F.normalize(s, dim=-1)
    t = F.normalize(t, dim=-1)
    return 1.0 - (s * t).sum(dim=-1).mean()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--bpe_path", default="sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--token_dim", type=int, default=192)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    print(f"[Info] device={device}", flush=True)

    teacher = build_efficientsam3_image_model(
        checkpoint_path=None,
        device=device,
        eval_mode=True,
        enable_segmentation=True,
        backbone_type="efficientvit",
        model_name="b1",
        text_encoder_type="MobileCLIP-S1",
    )
    sd = torch.load(args.teacher_ckpt, map_location=device)
    missing, unexpected = teacher.load_state_dict(sd, strict=False)
    print(f"[Info] teacher loaded: missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = TinyTextEncoderESAM3(
        bpe_path=args.bpe_path,
        token_dim=args.token_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)

    print(f"[Info] student params={count_params(student):,}", flush=True)

    prompts = build_prompt_pool()
    print(f"[Info] num prompts={len(prompts)}", flush=True)

    dataset = PromptDataset(prompts)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_loss = 1e9
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        student.train()

        total_loss = 0.0
        total_feat = 0.0
        total_embed = 0.0
        total_cos_f = 0.0
        total_cos_e = 0.0
        n = 0

        for batch_prompts in loader:
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                tout = teacher.backbone.forward_text(batch_prompts, device=device)

            sout = student.forward_text(batch_prompts, device=device)

            mask = tout["language_mask"]

            loss_feat = masked_mse(sout["language_features"], tout["language_features"], mask)
            loss_embed = masked_mse(sout["language_embeds"], tout["language_embeds"], mask)
            cos_f = masked_cos_loss(sout["language_features"], tout["language_features"], mask)
            cos_e = masked_cos_loss(sout["language_embeds"], tout["language_embeds"], mask)

            loss = loss_feat + 0.5 * loss_embed + 0.1 * cos_f + 0.05 * cos_e

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_feat += float(loss_feat.item())
            total_embed += float(loss_embed.item())
            total_cos_f += float(cos_f.item())
            total_cos_e += float(cos_e.item())
            n += 1

        avg = {
            "epoch": epoch,
            "loss": total_loss / max(1, n),
            "loss_feat": total_feat / max(1, n),
            "loss_embed": total_embed / max(1, n),
            "cos_feat": total_cos_f / max(1, n),
            "cos_embed": total_cos_e / max(1, n),
            "time_sec": round(time.time() - t0, 2),
        }
        history.append(avg)

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"loss={avg['loss']:.6f}, feat={avg['loss_feat']:.6f}, "
            f"embed={avg['loss_embed']:.6f}, cos_f={avg['cos_feat']:.6f}, "
            f"cos_e={avg['cos_embed']:.6f}, time={avg['time_sec']:.1f}s",
            flush=True,
        )

        torch.save(
            {
                "model": student.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "loss": avg["loss"],
            },
            ckpt_dir / "last.pth",
        )

        if avg["loss"] < best_loss:
            best_loss = avg["loss"]
            torch.save(
                {
                    "model": student.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "loss": avg["loss"],
                },
                ckpt_dir / "best.pth",
            )

        with open(out_dir / "log_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    student.save_config(out_dir / "tiny_text_config.json")
    print(f"[Done] best_loss={best_loss:.6f}", flush=True)


if __name__ == "__main__":
    main()
