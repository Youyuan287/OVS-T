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
        if k in row and row[k] is not None:
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
    arr = (np.array(m) > 0).astype(np.float32)
    return torch.from_numpy(arr)[None, None]


def resize_target(mask, size, device):
    mask = mask.to(device)
    mask = F.interpolate(mask, size=size, mode="nearest")
    return mask[0, 0]


def bce_dice_loss_logits(logits, target, eps=1e-6):
    with torch.amp.autocast("cuda" if logits.is_cuda else "cpu", enabled=False):
        logits = logits.float()
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = logits.sigmoid()
        inter = (prob * target).sum()
        union = prob.sum() + target.sum()
        dice = 1.0 - (2.0 * inter + eps) / (union + eps)
        return bce + dice


def hard_iou_from_logits(logits, target, eps=1e-6):
    pred = (logits.sigmoid() > 0.5).float()
    inter = (pred * target).sum()
    union = ((pred + target) > 0).float().sum()
    return float(((inter + eps) / (union + eps)).detach().cpu().item())


def build_groups(rows, max_images=0):
    groups = defaultdict(list)

    for r in rows:
        image = pick(r, ["image", "image_path", "img", "img_path"])
        prompt = pick(r, ["prompt", "text_prompt", "text", "category", "class_name"])
        mask = pick(r, ["mask", "mask_path", "pseudo_mask", "mask_file"], "")
        exists = int(r.get("exists", 1))

        if image is None or prompt is None:
            continue

        groups[image].append({
            "image": image,
            "prompt": str(prompt),
            "mask": mask if mask is not None else "",
            "exists": exists,
            "source": r.get("source", ""),
        })

    items = list(groups.items())
    random.shuffle(items)

    if max_images and max_images > 0:
        items = items[:max_images]

    return items


def set_trainable_stage4(model, tiny, tiny_train_mode="proj", train_mask_head=True):
    # 先全部冻结
    for p in model.parameters():
        p.requires_grad = False

    for p in tiny.parameters():
        p.requires_grad = False

    # 轻量文本编码器只训练接口层，避免破坏开放文本能力
    if tiny_train_mode == "proj":
        for name, p in tiny.named_parameters():
            lname = name.lower()
            if (
                "to_language_features" in lname
                or "to_language_embeds" in lname
                or "norm" in lname
            ):
                p.requires_grad = True

    elif tiny_train_mode == "all":
        for p in tiny.parameters():
            p.requires_grad = True

    elif tiny_train_mode == "none":
        pass

    else:
        raise ValueError(f"unknown tiny_train_mode={tiny_train_mode}")

    # Stage4 主要校准候选打分头、存在性相关分支，以及少量分割头
    train_keywords = [
        "dot_prod_scoring",
        "segmentation_head",
        "presence",
        "score",
        "objectness",
        "obj_score",
    ]

    for name, p in model.named_parameters():
        lname = name.lower()

        if any(k in lname for k in train_keywords):
            p.requires_grad = True

        # 原始大文本编码器不用
        if "language_backbone" in lname:
            p.requires_grad = False

        # Stage4 不动视觉主干，避免破坏 Stage3 红外适配
        if "backbone.vision_backbone" in lname:
            p.requires_grad = False

    if not train_mask_head:
        for name, p in model.named_parameters():
            if "segmentation_head" in name.lower():
                p.requires_grad = False


def count_trainable(model, tiny):
    total = 0
    trainable = 0

    for p in list(model.parameters()) + list(tiny.parameters()):
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    return total, trainable


def best_candidate_by_iou(pred_masks, target):
    """
    pred_masks: [1,Q,h,w] logits
    target: [h,w]
    """
    with torch.no_grad():
        probs = pred_masks[0].float().sigmoid()
        t = target.float()[None]

        inter = (probs * t).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + t.sum(dim=(1, 2)) - inter
        ious = (inter + 1e-6) / (union + 1e-6)

        best = int(torch.argmax(ious).item())

    return best


def forward_prompt(model, tiny, processor, backbone_out_image, row, device, args):
    prompt = row["prompt"]
    exists = int(row["exists"])

    text_outputs = tiny.forward_text([prompt], device=device)

    backbone_out = dict(backbone_out_image)
    backbone_out.update(text_outputs)

    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
        out = model.forward_grounding(
            backbone_out=backbone_out,
            find_input=processor.find_stage,
            geometric_prompt=model._get_dummy_prompt(),
            find_target=None,
        )

    pred_masks = out["pred_masks"]
    pred_logits = out["pred_logits"]
    presence = out.get("presence_logit_dec", None)

    cand_logits = pred_logits[0, :, 0].float()

    loss = cand_logits.new_tensor(0.0)

    log = {
        "exists": exists,
        "loss": 0.0,
        "mask_loss": 0.0,
        "iou": 0.0,
        "score_pos": 0.0,
        "score_neg": 0.0,
        "presence": -1.0,
    }

    # 存在性分支：有就压高，没有就压低
    if presence is not None:
        presence_logit = presence.float().mean().view(1)
        target_presence = torch.ones_like(presence_logit) if exists else torch.zeros_like(presence_logit)

        presence_loss = F.binary_cross_entropy_with_logits(presence_logit, target_presence)
        loss = loss + args.w_presence * presence_loss

        log["presence"] = float(presence_logit.sigmoid().detach().cpu().item())

    if exists == 1:
        if not row.get("mask"):
            return None

        target_mask = load_mask(row["mask"])
        target = resize_target(target_mask, pred_masks.shape[-2:], device=device)

        best_idx = best_candidate_by_iou(pred_masks, target)

        # 正样本：保持 mask 能力
        mask_logits = pred_masks[0, best_idx].float()
        mask_loss = bce_dice_loss_logits(mask_logits, target)
        loss = loss + args.w_mask * mask_loss

        # 正样本：最佳候选分数应该高
        best_logit = cand_logits[best_idx].view(1)
        score_loss = F.binary_cross_entropy_with_logits(best_logit, torch.ones_like(best_logit))
        loss = loss + args.w_score * score_loss

        log["mask_loss"] = float(mask_loss.detach().cpu().item())
        log["iou"] = hard_iou_from_logits(mask_logits.detach(), target.detach())
        log["score_pos"] = float(best_logit.sigmoid().detach().cpu().item())

    else:
        # 负样本：不训练 mask，只压低最容易误检的前几个候选分数
        k = min(args.neg_topk, cand_logits.numel())
        hard_logits = cand_logits.topk(k).values
        score_loss = F.binary_cross_entropy_with_logits(hard_logits, torch.zeros_like(hard_logits))
        loss = loss + args.w_neg_score * score_loss

        log["score_neg"] = float(hard_logits.sigmoid().mean().detach().cpu().item())

    log["loss"] = float(loss.detach().cpu().item())

    return loss, log


def run_epoch(model, tiny, processor, groups, optimizer, train, device, args):
    model.eval()

    if train:
        tiny.train()
    else:
        tiny.eval()

    total_loss = 0.0
    total = 0

    pos_n = 0
    neg_n = 0

    pos_iou = 0.0
    pos_score = 0.0
    neg_score = 0.0

    presence_pos = 0.0
    presence_neg = 0.0

    skipped = 0

    for idx, (image_path, rows) in enumerate(groups, 1):
        try:
            img = load_ir_as_rgb(image_path)
        except Exception:
            skipped += len(rows)
            continue

        try:
            image_tensor = processor.transform(v2.functional.to_image(img).to(device)).unsqueeze(0)

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    backbone_out_image = model.backbone.forward_image(image_tensor)

        except Exception:
            skipped += len(rows)
            continue

        for r in rows:
            if train:
                optimizer.zero_grad(set_to_none=True)

            try:
                if train:
                    result = forward_prompt(model, tiny, processor, backbone_out_image, r, device, args)
                else:
                    with torch.no_grad():
                        result = forward_prompt(model, tiny, processor, backbone_out_image, r, device, args)

            except Exception:
                skipped += 1
                continue

            if result is None:
                skipped += 1
                continue

            loss, log = result

            if train:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    [p for p in tiny.parameters() if p.requires_grad] +
                    [p for p in model.parameters() if p.requires_grad],
                    1.0,
                )

                optimizer.step()

            total += 1
            total_loss += log["loss"]

            if log["exists"] == 1:
                pos_n += 1
                pos_iou += log["iou"]
                pos_score += log["score_pos"]

                if log["presence"] >= 0:
                    presence_pos += log["presence"]

            else:
                neg_n += 1
                neg_score += log["score_neg"]

                if log["presence"] >= 0:
                    presence_neg += log["presence"]

        del backbone_out_image, image_tensor

        if device == "cuda":
            torch.cuda.empty_cache()

        if train and idx % args.log_interval == 0:
            print(
                f"  [image {idx}/{len(groups)}] "
                f"loss={total_loss/max(1,total):.4f}, "
                f"pos_iou={pos_iou/max(1,pos_n):.4f}, "
                f"pos_score={pos_score/max(1,pos_n):.4f}, "
                f"neg_score={neg_score/max(1,neg_n):.4f}, "
                f"pos={pos_n}, neg={neg_n}, skipped={skipped}",
                flush=True,
            )

    return {
        "loss": total_loss / max(1, total),
        "pos_iou": pos_iou / max(1, pos_n),
        "pos_score": pos_score / max(1, pos_n),
        "neg_score": neg_score / max(1, neg_n),
        "presence_pos": presence_pos / max(1, pos_n),
        "presence_neg": presence_neg / max(1, neg_n),
        "total": total,
        "pos": pos_n,
        "neg": neg_n,
        "skipped": skipped,
    }


def save_ckpt(path, model, tiny, optimizer, epoch, best_loss, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "esam3_model": model.state_dict(),
            "tiny_text": tiny.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "best_loss": best_loss,
            "args": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--base_esam3_ckpt", default="outputs/full_open_esam3_from_pruned_stage2/model/sam3.pt")
    parser.add_argument("--tiny_ckpt", default="outputs/tiny_text_encoder_distill_e300/checkpoints/best.pth")
    parser.add_argument("--save_dir", required=True)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_train_images", type=int, default=0)
    parser.add_argument("--max_val_images", type=int, default=500)
    parser.add_argument("--resolution", type=int, default=768)

    parser.add_argument("--lr_head", type=float, default=5e-6)
    parser.add_argument("--lr_tiny", type=float, default=5e-7)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--w_mask", type=float, default=0.5)
    parser.add_argument("--w_score", type=float, default=0.5)
    parser.add_argument("--w_neg_score", type=float, default=0.5)
    parser.add_argument("--w_presence", type=float, default=0.1)
    parser.add_argument("--neg_topk", type=int, default=5)

    parser.add_argument("--tiny_train_mode", default="proj", choices=["none", "proj", "all"])
    parser.add_argument("--train_mask_head", action="store_true")

    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log_interval", type=int, default=50)

    args = parser.parse_args()
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
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

    base_sd = torch.load(args.base_esam3_ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    print(f"[Info] base ESAM3 loaded: missing={len(missing)}, unexpected={len(unexpected)}", flush=True)

    tiny = TinyTextEncoderESAM3(
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        token_dim=192,
        hidden_dim=256,
        num_layers=3,
        num_heads=6,
    ).to(device)

    tiny_ckpt = torch.load(args.tiny_ckpt, map_location=device)
    if isinstance(tiny_ckpt, dict) and "model" in tiny_ckpt:
        tiny.load_state_dict(tiny_ckpt["model"], strict=True)
    else:
        tiny.load_state_dict(tiny_ckpt, strict=True)

    resume_ckpt = torch.load(args.resume, map_location=device)

    missing, unexpected = model.load_state_dict(resume_ckpt["esam3_model"], strict=False)
    print(f"[Info] resume ESAM3: missing={len(missing)}, unexpected={len(unexpected)}", flush=True)

    tiny.load_state_dict(resume_ckpt["tiny_text"], strict=True)
    print(f"[Info] resume TinyText ok: {args.resume}", flush=True)

    try:
        model.backbone.language_backbone = None
    except Exception:
        pass

    model.backbone.forward_text = tiny.forward_text

    set_trainable_stage4(
        model=model,
        tiny=tiny,
        tiny_train_mode=args.tiny_train_mode,
        train_mask_head=args.train_mask_head,
    )

    total_params, trainable_params = count_trainable(model, tiny)
    print(f"[Info] total params={total_params:,}", flush=True)
    print(f"[Info] trainable params={trainable_params:,}", flush=True)

    processor = Sam3Processor(
        model,
        resolution=args.resolution,
        device=device,
        confidence_threshold=0.0,
    )

    train_rows = load_jsonl(args.train_jsonl)
    val_rows = load_jsonl(args.val_jsonl)

    train_groups = build_groups(train_rows, max_images=args.max_train_images)
    val_groups = build_groups(val_rows, max_images=args.max_val_images)

    print(f"[Info] train images={len(train_groups)}", flush=True)
    print(f"[Info] val images={len(val_groups)}", flush=True)

    tiny_params = [p for p in tiny.parameters() if p.requires_grad]
    head_params = [p for p in model.parameters() if p.requires_grad]

    param_groups = []
    if tiny_params:
        param_groups.append({"params": tiny_params, "lr": args.lr_tiny})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr_head})

    if not param_groups:
        raise RuntimeError("No trainable parameters.")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    best_loss = 1e9
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_log = run_epoch(
            model=model,
            tiny=tiny,
            processor=processor,
            groups=train_groups,
            optimizer=optimizer,
            train=True,
            device=device,
            args=args,
        )

        val_log = run_epoch(
            model=model,
            tiny=tiny,
            processor=processor,
            groups=val_groups,
            optimizer=None,
            train=False,
            device=device,
            args=args,
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
            f"train_loss={train_log['loss']:.4f}, "
            f"train_pos_iou={train_log['pos_iou']:.4f}, "
            f"train_pos_score={train_log['pos_score']:.4f}, "
            f"train_neg_score={train_log['neg_score']:.4f}, "
            f"val_loss={val_log['loss']:.4f}, "
            f"val_pos_iou={val_log['pos_iou']:.4f}, "
            f"val_pos_score={val_log['pos_score']:.4f}, "
            f"val_neg_score={val_log['neg_score']:.4f}, "
            f"val_presence_pos={val_log['presence_pos']:.4f}, "
            f"val_presence_neg={val_log['presence_neg']:.4f}",
            flush=True,
        )

        save_ckpt(
            ckpt_dir / "last.pth",
            model,
            tiny,
            optimizer,
            epoch,
            best_loss,
            args,
        )

        if val_log["loss"] < best_loss:
            best_loss = val_log["loss"]
            save_ckpt(
                ckpt_dir / "best_by_loss.pth",
                model,
                tiny,
                optimizer,
                epoch,
                best_loss,
                args,
            )
            print(f"[Best] epoch={epoch}, best_loss={best_loss:.4f}", flush=True)

        with open(save_dir / "log_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"[Done] best_loss={best_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
