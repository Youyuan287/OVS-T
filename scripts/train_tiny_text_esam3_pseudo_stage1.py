import os
import json
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from tiny_text_encoder_esam3 import TinyTextEncoderESAM3


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick(row, keys, default=None):
    for k in keys:
        if k in row:
            return row[k]
    return default


def load_ir_as_rgb(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = ImageOps.autocontrast(img.convert("L")).convert("RGB")
    else:
        img = img.convert("RGB")
    return img


def load_mask(path):
    m = Image.open(path).convert("L")
    arr = np.array(m)
    arr = (arr > 0).astype(np.float32)
    return torch.from_numpy(arr)[None, None]  # [1,1,H,W]


def resize_target(mask, size, device):
    mask = mask.to(device)
    mask = F.interpolate(mask, size=size, mode="nearest")
    return mask[0, 0]


def bce_dice_loss(pred_prob, target, eps=1e-6):
    # BCE on probabilities is unsafe under autocast.
    # Force loss computation to FP32 and disable autocast.
    device_type = "cuda" if pred_prob.is_cuda else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=False):
        pred_prob = pred_prob.float().clamp(eps, 1.0 - eps)
        target = target.float()

        bce = F.binary_cross_entropy(pred_prob, target)

        inter = (pred_prob * target).sum()
        union = pred_prob.sum() + target.sum()
        dice = 1.0 - (2.0 * inter + eps) / (union + eps)

        loss = bce + dice

    return loss, bce.detach(), dice.detach()


def hard_iou(pred_prob, target, eps=1e-6):
    pred = (pred_prob > 0.5).float()
    inter = (pred * target).sum()
    union = ((pred + target) > 0).float().sum()
    return ((inter + eps) / (union + eps)).item()


def build_groups(rows, max_images=0):
    groups = defaultdict(list)
    for r in rows:
        image = pick(r, ["image", "image_path", "img", "img_path"])
        mask = pick(r, ["mask", "mask_path", "pseudo_mask", "mask_file"])
        prompt = pick(r, ["prompt", "text_prompt", "text", "category", "class_name"])

        if image is None or mask is None or prompt is None:
            continue

        groups[image].append({
            "image": image,
            "mask": mask,
            "prompt": str(prompt),
        })

    items = list(groups.items())
    random.shuffle(items)

    if max_images and max_images > 0:
        items = items[:max_images]

    return items


def set_trainable_stage1(
    model,
    tiny,
    train_decoder=False,
    tiny_train_mode="none",
    train_vision_tail=False,
    vision_tail_n=60,
):
    # freeze all ESAM3 first
    for p in model.parameters():
        p.requires_grad = False

    # freeze TinyText by default
    for p in tiny.parameters():
        p.requires_grad = False

    # TinyText train mode
    # none: fully frozen
    # proj: only output projections and norm
    # all : all TinyText parameters
    if tiny_train_mode == "all":
        for p in tiny.parameters():
            p.requires_grad = True
    elif tiny_train_mode == "proj":
        for name, p in tiny.named_parameters():
            lname = name.lower()
            if (
                "to_language_features" in lname
                or "to_language_embeds" in lname
                or "norm" in lname
            ):
                p.requires_grad = True
    elif tiny_train_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown tiny_train_mode: {tiny_train_mode}")

    # ESAM3 decoder / head modules
    train_keywords = [
        "segmentation_head",
        "dot_prod_scoring",
    ]

    if train_decoder:
        train_keywords += [
            "transformer",
            "geometry_encoder",
        ]

    for name, p in model.named_parameters():
        lname = name.lower()

        if any(k in lname for k in train_keywords):
            p.requires_grad = True

        # original language backbone always frozen / unused
        if "language_backbone" in lname:
            p.requires_grad = False

    # Train only the last N parameter tensors of vision_backbone
    if train_vision_tail:
        vision_params = [
            (name, p)
            for name, p in model.named_parameters()
            if "backbone.vision_backbone" in name
        ]

        if vision_tail_n <= 0 or vision_tail_n > len(vision_params):
            vision_tail_n = len(vision_params)

        for name, p in vision_params[-vision_tail_n:]:
            p.requires_grad = True

        print(
            f"[Info] train vision tail: last {vision_tail_n}/{len(vision_params)} vision params",
            flush=True,
        )
    else:
        for name, p in model.named_parameters():
            if "backbone.vision_backbone" in name:
                p.requires_grad = False

def count_trainable(model, tiny):
    total = 0
    trainable = 0
    for p in list(model.parameters()) + list(tiny.parameters()):
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return total, trainable


def forward_one_prompt(
    model,
    tiny,
    processor,
    backbone_out_image,
    prompt,
    target_mask,
    threshold,
    topk,
    device,
):
    text_outputs = tiny.forward_text([prompt], device=device)

    backbone_out = dict(backbone_out_image)
    backbone_out.update(text_outputs)

    geometric_prompt = model._get_dummy_prompt()

    out = model.forward_grounding(
        backbone_out=backbone_out,
        find_input=processor.find_stage,
        geometric_prompt=geometric_prompt,
        find_target=None,
    )

    pred_masks = out["pred_masks"]        # [1,Q,h,w], logits
    pred_logits = out["pred_logits"]      # [1,Q,1]
    presence = out.get("presence_logit_dec", None)

    scores = pred_logits.sigmoid()
    if presence is not None:
        scores = scores * presence.sigmoid().unsqueeze(1)
    scores = scores.squeeze(-1)[0]        # [Q]

    keep_idx = torch.nonzero(scores > threshold, as_tuple=False).view(-1)

    if keep_idx.numel() == 0:
        # fallback: at least use top-1, otherwise no gradient
        keep_idx = scores.topk(1).indices

    if keep_idx.numel() > topk:
        _, pos = scores[keep_idx].topk(topk)
        keep_idx = keep_idx[pos]

    selected_logits = pred_masks[0, keep_idx]  # [K,h,w]

    target = resize_target(target_mask, selected_logits.shape[-2:], device=device)

    weights = torch.softmax(scores[keep_idx].float(), dim=0).to(selected_logits.dtype)
    prob_each = selected_logits.sigmoid()
    soft_mask = (weights[:, None, None] * prob_each).sum(dim=0)

    loss, bce, dice = bce_dice_loss(soft_mask.float(), target.float())
    iou = hard_iou(soft_mask.detach().float(), target.float())

    return loss, float(bce.item()), float(dice.item()), iou, int(keep_idx.numel())


def run_epoch(
    model,
    tiny,
    processor,
    groups,
    optimizer,
    train,
    threshold,
    topk,
    device,
    log_interval=100,
):
    # 注意：model 必须保持 eval，避免 forward_grounding 进入 matching 分支
    model.eval()
    tiny.train() if train else tiny.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_prompts = 0
    total_images = 0
    total_kept = 0
    skipped = 0

    for idx, (image_path, rows) in enumerate(groups, 1):
        try:
            img = load_ir_as_rgb(image_path)
        except Exception:
            skipped += len(rows)
            continue

        vision_trainable = train and any(
            ("backbone.vision_backbone" in name and p.requires_grad)
            for name, p in model.named_parameters()
        )

        try:
            image_tensor = processor.transform(v2.functional.to_image(img).to(device)).unsqueeze(0)
            if vision_trainable:
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    backbone_out_image = model.backbone.forward_image(image_tensor)
            else:
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                        backbone_out_image = model.backbone.forward_image(image_tensor)
        except Exception:
            skipped += len(rows)
            continue

        total_images += 1

        if train and vision_trainable:
            optimizer.zero_grad(set_to_none=True)

        valid_prompt_in_image = 0

        for r in rows:
            try:
                target_mask = load_mask(r["mask"])
            except Exception:
                skipped += 1
                continue

            if train and not vision_trainable:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                loss, bce, dice, iou, kept = forward_one_prompt(
                    model=model,
                    tiny=tiny,
                    processor=processor,
                    backbone_out_image=backbone_out_image,
                    prompt=r["prompt"],
                    target_mask=target_mask,
                    threshold=threshold,
                    topk=topk,
                    device=device,
                )

            if train:
                loss.backward(retain_graph=vision_trainable)
                valid_prompt_in_image += 1

                if not vision_trainable:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in tiny.parameters() if p.requires_grad] +
                        [p for p in model.parameters() if p.requires_grad],
                        1.0,
                    )
                    optimizer.step()

            total_loss += float(loss.item())
            total_bce += bce
            total_dice += dice
            total_iou += iou
            total_kept += kept
            total_prompts += 1

        if train and vision_trainable and valid_prompt_in_image > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in tiny.parameters() if p.requires_grad] +
                [p for p in model.parameters() if p.requires_grad],
                1.0,
            )
            optimizer.step()

        del backbone_out_image, image_tensor
        if device == "cuda":
            torch.cuda.empty_cache()

        if train and idx % log_interval == 0:
            print(
                f"  [image {idx}/{len(groups)}] "
                f"loss={total_loss/max(1,total_prompts):.4f}, "
                f"iou={total_iou/max(1,total_prompts):.4f}, "
                f"prompts={total_prompts}, skipped={skipped}",
                flush=True,
            )

    return {
        "loss": total_loss / max(1, total_prompts),
        "bce": total_bce / max(1, total_prompts),
        "dice": total_dice / max(1, total_prompts),
        "iou": total_iou / max(1, total_prompts),
        "images": total_images,
        "prompts": total_prompts,
        "avg_kept": total_kept / max(1, total_prompts),
        "skipped": skipped,
    }


def save_ckpt(path, model, tiny, optimizer, epoch, best_iou, best_loss, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "esam3_model": model.state_dict(),
            "tiny_text": tiny.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "best_iou": best_iou,
            "best_loss": best_loss,
            "args": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--esam3_ckpt", required=True)
    parser.add_argument("--tiny_ckpt", required=True)
    parser.add_argument("--save_dir", required=True)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_train_images", type=int, default=0)
    parser.add_argument("--max_val_images", type=int, default=500)
    parser.add_argument("--lr_tiny", type=float, default=1e-5)
    parser.add_argument("--lr_head", type=float, default=1e-5)
    parser.add_argument("--lr_decoder", type=float, default=3e-6)
    parser.add_argument("--lr_vision", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--train_decoder", action="store_true")
    parser.add_argument("--train_vision_tail", action="store_true")
    parser.add_argument("--vision_tail_n", type=int, default=60)
    parser.add_argument("--tiny_train_mode", type=str, default="none", choices=["none", "proj", "all"])
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--resume_model_only", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"[Info] TinyText loaded: {args.tiny_ckpt}", flush=True)

    try:
        model.backbone.language_backbone = None
    except Exception:
        pass
    model.backbone.forward_text = tiny.forward_text

    set_trainable_stage1(model, tiny, train_decoder=args.train_decoder, tiny_train_mode=args.tiny_train_mode, train_vision_tail=args.train_vision_tail, vision_tail_n=args.vision_tail_n)
    total_params, trainable_params = count_trainable(model, tiny)
    print(f"[Info] total params approx={total_params:,}", flush=True)
    print(f"[Info] trainable params={trainable_params:,}", flush=True)

    processor = Sam3Processor(
        model,
        resolution=args.resolution,
        device=device,
        confidence_threshold=args.threshold,
    )

    train_rows = load_jsonl(args.train_jsonl)
    val_rows = load_jsonl(args.val_jsonl)

    train_groups = build_groups(train_rows, max_images=args.max_train_images)
    val_groups = build_groups(val_rows, max_images=args.max_val_images)

    print(f"[Info] train images={len(train_groups)}", flush=True)
    print(f"[Info] val images={len(val_groups)}", flush=True)

    tiny_params = [p for p in tiny.parameters() if p.requires_grad]

    vision_params = []
    decoder_params = []
    head_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        lname = name.lower()
        if "backbone.vision_backbone" in lname:
            vision_params.append(p)
        elif "transformer" in lname or "geometry_encoder" in lname:
            decoder_params.append(p)
        elif "segmentation_head" in lname or "dot_prod_scoring" in lname:
            head_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if len(tiny_params) > 0:
        param_groups.append({"params": tiny_params, "lr": args.lr_tiny})
    if len(decoder_params) > 0:
        param_groups.append({"params": decoder_params, "lr": args.lr_decoder})
    if len(head_params) > 0:
        param_groups.append({"params": head_params, "lr": args.lr_head})
    if len(vision_params) > 0:
        param_groups.append({"params": vision_params, "lr": args.lr_vision})
    if len(other_params) > 0:
        param_groups.append({"params": other_params, "lr": args.lr_head})

    print(
        f"[Info] optimizer groups: tiny={len(tiny_params)}, decoder={len(decoder_params)}, "
        f"head={len(head_params)}, vision={len(vision_params)}, other={len(other_params)}",
        flush=True,
    )

    if len(param_groups) == 0:
        raise RuntimeError("No trainable parameters. Please check train settings.")

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
    )

    best_iou = 0.0
    best_loss = 1e9
    history = []
    start_epoch = 1

    if args.resume:
        print(f"[Info] resume from: {args.resume}", flush=True)
        resume_ckpt = torch.load(args.resume, map_location=device)

        if "esam3_model" in resume_ckpt:
            missing, unexpected = model.load_state_dict(resume_ckpt["esam3_model"], strict=False)
            print(f"[Info] resume ESAM3: missing={len(missing)}, unexpected={len(unexpected)}", flush=True)

        if "tiny_text" in resume_ckpt:
            tiny.load_state_dict(resume_ckpt["tiny_text"], strict=True)
            print("[Info] resume TinyText ok", flush=True)

        if resume_ckpt.get("optimizer") is not None and not args.resume_model_only:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            print("[Info] resume optimizer ok", flush=True)
        elif args.resume_model_only:
            print("[Info] resume model only; optimizer is re-initialized", flush=True)

        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        best_iou = float(resume_ckpt.get("best_iou", 0.0))
        best_loss = float(resume_ckpt.get("best_loss", 1e9))

        log_path = save_dir / "log_history.json"
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                history = json.load(f)

        print(f"[Info] start_epoch={start_epoch}, best_iou={best_iou:.4f}, best_loss={best_loss:.4f}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_log = run_epoch(
            model=model,
            tiny=tiny,
            processor=processor,
            groups=train_groups,
            optimizer=optimizer,
            train=True,
            threshold=args.threshold,
            topk=args.topk,
            device=device,
        )

        with torch.no_grad():
            val_log = run_epoch(
                model=model,
                tiny=tiny,
                processor=processor,
                groups=val_groups,
                optimizer=None,
                train=False,
                threshold=args.threshold,
                topk=args.topk,
                device=device,
            )

        dt = time.time() - t0

        record = {
            "epoch": epoch,
            "time_sec": round(dt, 2),
            "train": train_log,
            "val": val_log,
        }
        history.append(record)

        print(
            f"[Epoch {epoch}/{args.epochs}] {dt:.1f}s | "
            f"train_loss={train_log['loss']:.4f}, train_iou={train_log['iou']:.4f}, "
            f"val_loss={val_log['loss']:.4f}, val_iou={val_log['iou']:.4f}, "
            f"val_kept={val_log['avg_kept']:.1f}",
            flush=True,
        )

        save_ckpt(
            ckpt_dir / "last.pth",
            model,
            tiny,
            optimizer,
            epoch,
            best_iou,
            best_loss,
            args,
        )

        if val_log["iou"] > best_iou:
            best_iou = val_log["iou"]
            save_ckpt(
                ckpt_dir / "best_by_iou.pth",
                model,
                tiny,
                optimizer,
                epoch,
                best_iou,
                best_loss,
                args,
            )
            print(f"[Best-IoU] epoch={epoch}, best_iou={best_iou:.4f}", flush=True)

        if val_log["loss"] < best_loss:
            best_loss = val_log["loss"]
            save_ckpt(
                ckpt_dir / "best_by_loss.pth",
                model,
                tiny,
                optimizer,
                epoch,
                best_iou,
                best_loss,
                args,
            )
            print(f"[Best-Loss] epoch={epoch}, best_loss={best_loss:.4f}", flush=True)

        with open(save_dir / "log_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"[Done] best_iou={best_iou:.4f}, best_loss={best_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
