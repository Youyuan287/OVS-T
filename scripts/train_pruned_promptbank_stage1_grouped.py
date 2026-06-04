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
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def set_seed(seed=33):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_text(s):
    return " ".join(str(s).lower().replace("_", " ").replace("-", " ").replace("/", " ").split())


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


def load_mask_tensor(path, size_hw, device):
    m = Image.open(path).convert("L")
    arr = (np.array(m) > 127).astype(np.float32)
    t = torch.from_numpy(arr)[None, None, ...].to(device)
    t = F.interpolate(t, size=size_hw, mode="nearest")
    return t


class GroupedPseudoDataset(Dataset):
    def __init__(self, jsonl_path, max_images=0):
        rows = load_jsonl(jsonl_path)
        groups = defaultdict(list)
        for r in rows:
            groups[r["image"]].append(r)

        self.items = [{"image": image_path, "rows": rs} for image_path, rs in groups.items()]
        if max_images and max_images > 0:
            self.items = self.items[:max_images]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def choose_prompt_bank_item(prompt, prompt_bank):
    prompt_norm = norm_text(prompt)
    items = prompt_bank["items"]
    alias_to_canonical = prompt_bank.get("alias_to_canonical", {})

    # 1. 优先完全匹配文本本身，例如 prompt=vehicle 就用 text=vehicle
    for item in items:
        if norm_text(item.get("text", "")) == prompt_norm:
            return item

    # 2. 完全匹配 alias
    if prompt_norm in alias_to_canonical:
        canonical = alias_to_canonical[prompt_norm]
        for item in items:
            if item.get("canonical") == canonical and norm_text(item.get("text", "")) == canonical:
                return item
        for item in items:
            if item.get("canonical") == canonical:
                return item

    # 3. prompt 中包含 alias
    for alias, canonical in alias_to_canonical.items():
        if alias and alias in prompt_norm:
            for item in items:
                if item.get("canonical") == canonical and norm_text(item.get("text", "")) == canonical:
                    return item
            for item in items:
                if item.get("canonical") == canonical:
                    return item

    return None


def dice_loss_with_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    prob = prob.flatten(1)
    target = target.flatten(1)
    inter = (prob * target).sum(dim=1)
    denom = prob.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def mask_iou_from_logits(logits, target, eps=1e-6):
    with torch.no_grad():
        pred = (torch.sigmoid(logits) > 0.5).float()
        pred = pred.flatten(1)
        target = target.flatten(1)
        inter = (pred * target).sum(dim=1)
        union = ((pred + target) > 0).float().sum(dim=1)
        return ((inter + eps) / (union + eps)).mean().item()


def normalize_pred_masks(pred_masks):
    if pred_masks.ndim == 4:
        pred_masks = pred_masks[0]
    elif pred_masks.ndim == 3:
        pass
    else:
        raise RuntimeError(f"Unexpected pred_masks shape: {tuple(pred_masks.shape)}")
    return pred_masks[:, None, :, :]


def select_best_logit(masks_logits, target):
    if masks_logits.shape[0] == 1:
        return masks_logits[0:1]

    with torch.no_grad():
        probs = torch.sigmoid(masks_logits)
        preds = (probs > 0.5).float()
        target_q = target.expand_as(preds)
        inter = (preds * target_q).flatten(1).sum(dim=1)
        union = ((preds + target_q) > 0).float().flatten(1).sum(dim=1)
        iou = (inter + 1e-6) / (union + 1e-6)
        best_idx = int(torch.argmax(iou).item())

    return masks_logits[best_idx:best_idx + 1]



def set_stage1_trainable(model):
    """
    Fast Pruned Stage1:
    - 冻结 vision_backbone / transformer / geometry_encoder；
    - 只训练 segmentation_head、dot_prod_scoring 以及明显的 mask/presence/head 小模块；
    - 目标是先快速适配 prompt-bank 版本，不再打开 31M 参数。
    """
    train_keywords = [
        "segmentation_head",
        "dot_prod_scoring",
        "presence",
        "mask_head",
        "bbox_head",
    ]

    freeze_keywords = [
        "backbone",
        "language",
        "text",
        "clip",
        "token",
        "transformer",
        "geometry_encoder",
    ]

    for name, p in model.named_parameters():
        lname = name.lower()
        p.requires_grad = False

        if any(k in lname for k in train_keywords):
            p.requires_grad = True

        if any(k in lname for k in freeze_keywords):
            p.requires_grad = False


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def encode_image_once(model, processor, image_pil, device):
    image_tensor = processor.transform(v2.functional.to_image(image_pil).to(device))
    image_batch = image_tensor.unsqueeze(0)

    # Stage1 冻结视觉 backbone
    with torch.no_grad():
        backbone_out = model.backbone.forward_image(image_batch)

    return backbone_out


def forward_one_prompt(model, processor, backbone_out_image, prompt_item, device):
    backbone_out = dict(backbone_out_image)

    for k, v in prompt_item["outputs"].items():
        backbone_out[k] = v

    geometric_prompt = model._get_dummy_prompt()

    outputs = model.forward_grounding(
        backbone_out=backbone_out,
        find_input=processor.find_stage,
        geometric_prompt=geometric_prompt,
        find_target=None,
    )
    return outputs


def run_one_epoch(model, processor, prompt_bank, loader, optimizer, scaler, device, train=True, use_amp=False):
    model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    total_iou = 0.0
    used_images = 0
    used_prompts = 0
    skipped_prompts = 0

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    for batch in loader:
        item = batch[0]
        image_path = item["image"]
        rows = item["rows"]

        image = load_ir_as_rgb(image_path)

        if train:
            optimizer.zero_grad(set_to_none=True)

        group_losses = []
        group_bces = []
        group_dices = []
        group_ious = []

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device_type, enabled=(use_amp and device_type == "cuda")):
                backbone_out_image = encode_image_once(model, processor, image, device)

                for r in rows:
                    prompt = r["prompt"]
                    weight = float(r.get("weight", 1.0))

                    prompt_item = choose_prompt_bank_item(prompt, prompt_bank)
                    if prompt_item is None:
                        skipped_prompts += 1
                        continue

                    outputs = forward_one_prompt(
                        model=model,
                        processor=processor,
                        backbone_out_image=backbone_out_image,
                        prompt_item=prompt_item,
                        device=device,
                    )

                    pred_masks = outputs.get("pred_masks", None)
                    if pred_masks is None:
                        skipped_prompts += 1
                        continue

                    masks_logits = normalize_pred_masks(pred_masks)
                    if masks_logits.shape[0] == 0:
                        skipped_prompts += 1
                        continue

                    target = load_mask_tensor(r["mask"], size_hw=masks_logits.shape[-2:], device=device)
                    selected_logit = select_best_logit(masks_logits, target)

                    bce = F.binary_cross_entropy_with_logits(selected_logit, target)
                    dice = dice_loss_with_logits(selected_logit, target)
                    loss = weight * (bce + dice)

                    group_losses.append(loss)
                    group_bces.append(bce.detach())
                    group_dices.append(dice.detach())
                    group_ious.append(mask_iou_from_logits(selected_logit.detach(), target.detach()))
                    used_prompts += 1

                if len(group_losses) == 0:
                    continue

                group_loss = torch.stack(group_losses).mean()

        if train:
            if scaler is not None and use_amp and device_type == "cuda":
                scaler.scale(group_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                group_loss.backward()
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                optimizer.step()

        used_images += 1
        total_loss += float(group_loss.item())
        total_bce += float(torch.stack(group_bces).mean().item()) if group_bces else 0.0
        total_dice += float(torch.stack(group_dices).mean().item()) if group_dices else 0.0
        total_iou += float(np.mean(group_ious)) if group_ious else 0.0

    return {
        "loss": total_loss / max(1, used_images),
        "bce": total_bce / max(1, used_images),
        "dice": total_dice / max(1, used_images),
        "iou": total_iou / max(1, used_images),
        "used_images": used_images,
        "used_prompts": used_prompts,
        "skipped_prompts": skipped_prompts,
    }


def clean_state_dict_for_save(model):
    sd = {}
    for k, v in model.state_dict().items():
        if k.startswith("backbone.language_backbone."):
            continue
        if torch.is_tensor(v):
            sd[k] = v.detach().cpu()
    return sd


def save_ckpt(path, model, optimizer, epoch, best_iou, args):
    torch.save({
        "model": clean_state_dict_for_save(model),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "best_iou": best_iou,
        "args": vars(args),
    }, path)


def save_topk(ckpt_dir, model, optimizer, epoch, best_iou, args, cur_iou, cur_loss, topk_iou, topk_loss, keep_topk=5):
    save_ckpt(ckpt_dir / "last.pth", model, optimizer, epoch, best_iou, args)

    if len(topk_iou) < keep_topk or cur_iou > min(x["score"] for x in topk_iou):
        p = ckpt_dir / f"top_iou_epoch_{epoch:03d}_iou_{cur_iou:.4f}_loss_{cur_loss:.4f}.pth"
        save_ckpt(p, model, optimizer, epoch, best_iou, args)
        topk_iou.append({"path": str(p), "score": cur_iou, "epoch": epoch})
        topk_iou.sort(key=lambda x: x["score"], reverse=True)
        while len(topk_iou) > keep_topk:
            rm = topk_iou.pop(-1)
            try:
                Path(rm["path"]).unlink()
            except FileNotFoundError:
                pass

    if len(topk_loss) < keep_topk or cur_loss < max(x["score"] for x in topk_loss):
        p = ckpt_dir / f"top_loss_epoch_{epoch:03d}_loss_{cur_loss:.4f}_iou_{cur_iou:.4f}.pth"
        save_ckpt(p, model, optimizer, epoch, best_iou, args)
        topk_loss.append({"path": str(p), "score": cur_loss, "epoch": epoch})
        topk_loss.sort(key=lambda x: x["score"])
        while len(topk_loss) > keep_topk:
            rm = topk_loss.pop(-1)
            try:
                Path(rm["path"]).unlink()
            except FileNotFoundError:
                pass

    with open(ckpt_dir / "topk_records.json", "w", encoding="utf-8") as f:
        json.dump({"topk_iou": topk_iou, "topk_loss": topk_loss}, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--pruned_ckpt", required=True)
    parser.add_argument("--prompt_bank", required=True)
    parser.add_argument("--save_dir", required=True)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max_train_images", type=int, default=0)
    parser.add_argument("--max_val_images", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_dir = Path(args.save_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    print(f"[Info] device={device}", flush=True)
    print(f"[Info] pruned_ckpt={args.pruned_ckpt}", flush=True)
    print(f"[Info] prompt_bank={args.prompt_bank}", flush=True)

    model = build_efficientsam3_image_model(
        checkpoint_path=None,
        device=device,
        eval_mode=True,
        enable_segmentation=True,
        backbone_type="efficientvit",
        model_name="b1",
        text_encoder_type="MobileCLIP-S1",
    )

    sd = torch.load(args.pruned_ckpt, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[Info] load pruned: missing={len(missing)}, unexpected={len(unexpected)}", flush=True)

    # 删除 language_backbone，避免误用
    try:
        model.backbone.language_backbone = None
        print("[Info] language_backbone removed", flush=True)
    except Exception as e:
        print(f"[Warn] remove language_backbone failed: {e}", flush=True)

    set_stage1_trainable(model)
    model.to(device)

    total, trainable = count_parameters(model)
    print(f"[Info] total params={total}", flush=True)
    print(f"[Info] trainable params={trainable}", flush=True)

    prompt_bank = torch.load(args.prompt_bank, map_location="cpu")
    # 提前把 prompt bank tensor 放到 GPU，避免每个 prompt 重复 .to(device)
    for item in prompt_bank["items"]:
        for k, v in item["outputs"].items():
            if torch.is_tensor(v):
                item["outputs"][k] = v.to(device)

    processor = Sam3Processor(model, device=device, confidence_threshold=-1.0)

    train_set = GroupedPseudoDataset(args.train_jsonl, max_images=args.max_train_images)
    val_set = GroupedPseudoDataset(args.val_jsonl, max_images=args.max_val_images)

    print(f"[Info] train images={len(train_set)}", flush=True)
    print(f"[Info] val images={len(val_set)}", flush=True)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda x: x)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(args.use_amp and device == "cuda"))

    best_iou = 0.0
    best_loss = float("inf")
    topk_iou = []
    topk_loss = []
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_log = run_one_epoch(model, processor, prompt_bank, train_loader, optimizer, scaler, device, train=True, use_amp=args.use_amp)

        with torch.no_grad():
            val_log = run_one_epoch(model, processor, prompt_bank, val_loader, optimizer, None, device, train=False, use_amp=False)

        dt = time.time() - t0
        cur_iou = float(val_log["iou"])
        cur_loss = float(val_log["loss"])

        if cur_iou > best_iou:
            best_iou = cur_iou
            save_ckpt(ckpt_dir / "best_by_iou.pth", model, optimizer, epoch, best_iou, args)

        if cur_loss < best_loss:
            best_loss = cur_loss
            save_ckpt(ckpt_dir / "best_by_loss.pth", model, optimizer, epoch, best_iou, args)

        save_topk(ckpt_dir, model, optimizer, epoch, best_iou, args, cur_iou, cur_loss, topk_iou, topk_loss)

        log = {
            "epoch": epoch,
            "time_sec": dt,
            "train": train_log,
            "val": val_log,
            "best_iou": best_iou,
            "best_loss": best_loss,
        }
        history.append(log)

        with open(Path(args.save_dir) / "log_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        print(
            f"[Epoch {epoch}/{args.epochs}] {dt:.1f}s | "
            f"train_loss={train_log['loss']:.4f}, train_iou={train_log['iou']:.4f}, "
            f"val_loss={val_log['loss']:.4f}, val_iou={val_log['iou']:.4f}, "
            f"best_iou={best_iou:.4f}, best_loss={best_loss:.4f}, "
            f"train_prompts={train_log['used_prompts']}, val_prompts={val_log['used_prompts']}, "
            f"skipped={train_log['skipped_prompts'] + val_log['skipped_prompts']}",
            flush=True
        )

    print(f"[Done] best_iou={best_iou:.4f}, best_loss={best_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
