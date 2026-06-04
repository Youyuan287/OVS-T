import os
import json
import time
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def set_seed(seed=33):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def load_mask_tensor(path, size_hw=None):
    m = Image.open(path).convert("L")
    arr = (np.array(m) > 127).astype(np.float32)
    t = torch.from_numpy(arr)[None, None, ...]  # 1,1,H,W
    if size_hw is not None:
        t = F.interpolate(t, size=size_hw, mode="nearest")
    return t


class ESAM3PseudoDataset(Dataset):
    def __init__(self, jsonl_path, max_samples=0):
        self.rows = load_jsonl(jsonl_path)
        if max_samples and max_samples > 0:
            self.rows = self.rows[:max_samples]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return r


def dice_loss_with_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    prob = prob.flatten(1)
    target = target.flatten(1)
    inter = (prob * target).sum(dim=1)
    denom = prob.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


def mask_iou_from_logits(logits, target, eps=1e-6):
    with torch.no_grad():
        pred = (torch.sigmoid(logits) > 0.5).float()
        pred = pred.flatten(1)
        target = target.flatten(1)
        inter = (pred * target).sum(dim=1)
        union = ((pred + target) > 0).float().sum(dim=1)
        return ((inter + eps) / (union + eps)).mean().item()


def select_best_logit(masks_logits, target):
    """
    masks_logits: [N,1,H,W]
    target: [1,1,H,W]
    """
    if masks_logits.shape[0] == 1:
        return masks_logits[0:1]

    with torch.no_grad():
        probs = torch.sigmoid(masks_logits)
        preds = (probs > 0.5).float()
        target_n = target.expand_as(preds)
        inter = (preds * target_n).flatten(1).sum(dim=1)
        union = ((preds + target_n) > 0).float().flatten(1).sum(dim=1)
        iou = (inter + 1e-6) / (union + 1e-6)
        best_idx = int(torch.argmax(iou).item())

    return masks_logits[best_idx:best_idx + 1]


def set_stage1_trainable(model):
    """
    第一版保守策略：
    - 冻结 text 相关参数，尽量保留开放词汇能力
    - 其他参数允许训练
    """
    for name, p in model.named_parameters():
        lname = name.lower()
        if "text" in lname or "clip" in lname or "token" in lname:
            p.requires_grad = False
        else:
            p.requires_grad = True


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_one_epoch(model, processor, loader, optimizer, device, train=True, use_amp=False, grad_clip_norm=1.0):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_iou = 0.0
    used = 0
    skipped = 0

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    for rows in loader:
        # batch_size 固定为 1，因此 rows 是长度为1的 list
        row = rows[0]

        image = load_ir_as_rgb(row["image"])
        prompt = row["prompt"]
        weight = float(row.get("weight", 1.0))

        if train:
            optimizer.zero_grad(set_to_none=True)

        amp_ctx = torch.amp.autocast(device_type=device_type, enabled=(use_amp and device_type == "cuda"))

        with torch.set_grad_enabled(train):
            with amp_ctx:
                state = processor.set_image(image)
                state = processor.set_text_prompt(prompt=prompt, state=state)

                masks_logits = state.get("masks_logits", None)

                if masks_logits is None or masks_logits.shape[0] == 0:
                    skipped += 1
                    continue

                if masks_logits.ndim == 3:
                    masks_logits = masks_logits[:, None, :, :]

                target = load_mask_tensor(row["mask"], size_hw=masks_logits.shape[-2:]).to(device)

                selected_logit = select_best_logit(masks_logits, target)

                bce = F.binary_cross_entropy_with_logits(selected_logit, target)
                dice = dice_loss_with_logits(selected_logit, target)
                loss = weight * (bce + dice)

        if train:
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_clip_norm)
            optimizer.step()

        total_loss += float(loss.item())
        total_bce += float(bce.item())
        total_dice += float(dice.item())
        total_iou += mask_iou_from_logits(selected_logit.detach(), target.detach())
        used += 1

    return {
        "loss": total_loss / max(1, used),
        "bce": total_bce / max(1, used),
        "dice": total_dice / max(1, used),
        "iou": total_iou / max(1, used),
        "used": used,
        "skipped": skipped,
    }


def save_ckpt(path, model, optimizer, epoch, best_iou, args):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "best_iou": best_iou,
        "args": vars(args),
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_train", type=int, default=100)
    parser.add_argument("--max_val", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence_threshold", type=float, default=-1.0)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_dir = Path(args.save_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    print(f"[Info] device={device}")
    print(f"[Info] ckpt={args.ckpt}")

    model = build_efficientsam3_image_model(
        checkpoint_path=args.ckpt,
        device=device,
        eval_mode=False,
        enable_segmentation=True,
        backbone_type="efficientvit",
        model_name="b1",
        text_encoder_type="MobileCLIP-S1",
    )

    set_stage1_trainable(model)
    model.to(device)

    total, trainable = count_parameters(model)
    print(f"[Info] total params={total}")
    print(f"[Info] trainable params={trainable}")

    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=args.confidence_threshold,
    )

    train_set = ESAM3PseudoDataset(args.train_jsonl, max_samples=args.max_train)
    val_set = ESAM3PseudoDataset(args.val_jsonl, max_samples=args.max_val)

    # processor 当前按单图单prompt调用，batch_size先固定1
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda x: x)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_iou = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_log = run_one_epoch(
            model, processor, train_loader, optimizer, device,
            train=True, use_amp=args.use_amp
        )

        with torch.no_grad():
            val_log = run_one_epoch(
                model, processor, val_loader, optimizer, device,
                train=False, use_amp=False
            )

        dt = time.time() - t0

        msg = (
            f"[Epoch {epoch}/{args.epochs}] {dt:.1f}s | "
            f"train_loss={train_log['loss']:.4f}, train_iou={train_log['iou']:.4f}, "
            f"used={train_log['used']}, skipped={train_log['skipped']} | "
            f"val_loss={val_log['loss']:.4f}, val_iou={val_log['iou']:.4f}, "
            f"used={val_log['used']}, skipped={val_log['skipped']}"
        )
        print(msg)

        history.append({
            "epoch": epoch,
            "train": train_log,
            "val": val_log,
            "time": dt,
        })

        with open(Path(args.save_dir) / "log_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        save_ckpt(ckpt_dir / "last.pth", model, optimizer, epoch, best_iou, args)

        if val_log["iou"] > best_iou:
            best_iou = val_log["iou"]
            save_ckpt(ckpt_dir / "best_by_iou.pth", model, optimizer, epoch, best_iou, args)
            print(f"[Best] epoch={epoch}, best_iou={best_iou:.4f}")

    print(f"[Done] best_iou={best_iou:.4f}")


if __name__ == "__main__":
    main()
