import os
import sys
import json
from pathlib import Path

CODE_ROOT_FOR_IMPORT = Path(os.environ.get("RAYTRON_CODE", Path(__file__).resolve().parent))
sys.path.insert(0, str(CODE_ROOT_FOR_IMPORT))
sys.path.insert(0, str(CODE_ROOT_FOR_IMPORT / "sam3"))
from collections import defaultdict, OrderedDict
from contextlib import nullcontext

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from tiny_text_encoder_esam3 import TinyTextEncoderESAM3


# =========================
# Fixed official paths
# =========================
CODE_ROOT = Path(os.environ.get("RAYTRON_CODE", Path(__file__).resolve().parent))
TEST_ROOT = Path(os.environ.get("RAYTRON_TEST", "/raytron/test"))

TASK_PATH = TEST_ROOT / "test_tasks.json"
OUT_PATH = TEST_ROOT / "predictions.json"

WEIGHT_PATH = CODE_ROOT / "model" / "sam3.pt"
TINY_FALLBACK_PATH = CODE_ROOT / "model" / "tiny_text.pt"
BPE_PATH = CODE_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"


# =========================
# Runtime hyperparameters
# =========================
RESOLUTION = int(os.environ.get("RESOLUTION", "768"))

# Stage4 calibrated checkpoint used this gate.
# If too many black masks in local validation, try 0.45 or 0.35.
EXIST_THRESHOLD = float(os.environ.get("EXIST_THRESHOLD", "0.55"))

MASK_THRESHOLD = float(os.environ.get("MASK_THRESHOLD", "0.5"))

# top1 is safer for false positives. train_like unions candidates above MASK_SCORE_THRESHOLD.
SELECT_MODE = os.environ.get("SELECT_MODE", "top1")  # top1 / train_like
MASK_SCORE_THRESHOLD = float(os.environ.get("MASK_SCORE_THRESHOLD", "0.5"))
TOPK = int(os.environ.get("TOPK", "20"))

# Prompt normalization is useful if the model was trained mainly on canonical English category words.
USE_PROMPT_NORMALIZE = os.environ.get("PROMPT_NORMALIZE", "1") == "1"


try:
    from pycocotools import mask as mask_utils
    HAS_PYCOCO = True
except Exception:
    HAS_PYCOCO = False


def amp_context(device: str):
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def encode_rle(mask: np.ndarray):
    """
    Encode binary mask to COCO-style RLE.
    mask: H x W, uint8, values 0/1.
    """
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError(f"mask must be HxW, got shape={mask.shape}")

    if HAS_PYCOCO:
        rle = mask_utils.encode(np.asfortranarray(mask))
        counts = rle["counts"]
        if isinstance(counts, bytes):
            counts = counts.decode("utf-8")
        return {
            "size": [int(rle["size"][0]), int(rle["size"][1])],
            "counts": counts,
        }

    # Fallback uncompressed RLE, Fortran order, starts with count of zeros.
    h, w = mask.shape
    pixels = mask.flatten(order="F")
    counts = []
    prev = 0
    run_len = 0

    for p in pixels:
        p = int(p)
        if p == prev:
            run_len += 1
        else:
            counts.append(run_len)
            run_len = 1
            prev = p

    counts.append(run_len)
    return {"size": [int(h), int(w)], "counts": counts}


def get_first(d, keys, default=None):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def normalize_tasks(raw):
    """
    Official format is a JSON array, but this function also tolerates several wrapper forms.
    """
    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        for k in ["tasks", "annotations", "data", "test_tasks", "items"]:
            if k in raw and isinstance(raw[k], list):
                return raw[k]

        vals = list(raw.values())
        if vals and all(isinstance(v, dict) for v in vals):
            return vals

    raise ValueError("test_tasks.json must be a list or contain a task list")


def resolve_image_path(image_rel):
    p = Path(str(image_rel))
    if p.is_absolute():
        return p
    return TEST_ROOT / p


def load_ir_as_rgb(path: Path):
    """
    Infrared images can be grayscale. Convert to RGB because the visual backbone expects 3 channels.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)

    if img.mode == "RGB":
        return img.convert("RGB")

    # For IR images, autocontrast helps stabilize intensity range.
    return ImageOps.autocontrast(img.convert("L")).convert("RGB")


ALIAS_MAP = {
    # English aliases
    "human": "person",
    "people": "person",
    "pedestrian": "person",
    "man": "person",
    "woman": "person",

    "automobile": "car",
    "sedan": "car",
    "suv": "car",
    "van": "vehicle",

    "wire": "power line",
    "cable": "power line",
    "power cable": "power line",

    "utility pole": "pole",
    "electric pole": "pole",
    "power pole": "pole",
    "telegraph pole": "pole",
    "transmission tower": "pole",
    "power tower": "pole",
    "tower": "pole",

    "electrical insulator": "insulator",
    "power insulator": "insulator",

    "vegetation": "tree",
    "plant": "tree",

    "bear": "animal",
    "dog": "animal",

    # Chinese aliases
    "人": "person",
    "行人": "person",
    "人员": "person",
    "男人": "person",
    "女人": "person",

    "汽车": "car",
    "轿车": "car",
    "小汽车": "car",
    "车辆": "vehicle",
    "车": "vehicle",
    "卡车": "truck",
    "货车": "truck",
    "公交车": "vehicle",
    "客车": "vehicle",

    "绝缘子": "insulator",
    "变压器": "transformer",

    "电线": "power line",
    "导线": "power line",
    "输电线": "power line",
    "电缆": "power line",

    "电杆": "pole",
    "杆塔": "pole",
    "铁塔": "pole",
    "输电塔": "pole",
    "电力杆塔": "pole",

    "树": "tree",
    "树木": "tree",
    "植被": "tree",

    "动物": "animal",
    "熊": "animal",
    "狗": "animal",
}


PHRASE_RULES = [
    ("transmission tower", "pole"),
    ("power tower", "pole"),
    ("utility pole", "pole"),
    ("electric pole", "pole"),
    ("power pole", "pole"),
    ("tower", "pole"),
    ("pole", "pole"),

    ("insulator", "insulator"),
    ("transformer", "transformer"),

    ("power line", "power line"),
    ("wire", "power line"),
    ("cable", "power line"),

    ("truck", "truck"),
    ("bus", "vehicle"),
    ("van", "vehicle"),
    ("vehicle", "vehicle"),
    ("automobile", "car"),
    ("car", "car"),

    ("person", "person"),
    ("human", "person"),
    ("pedestrian", "person"),

    ("road", "road"),
    ("street", "road"),

    ("building", "building"),
    ("tree", "tree"),
    ("vegetation", "tree"),

    ("animal", "animal"),
    ("bear", "animal"),
    ("dog", "animal"),

    ("绝缘子", "insulator"),
    ("变压器", "transformer"),
    ("输电线", "power line"),
    ("电线", "power line"),
    ("导线", "power line"),
    ("电缆", "power line"),
    ("杆塔", "pole"),
    ("电杆", "pole"),
    ("铁塔", "pole"),
    ("人", "person"),
    ("行人", "person"),
    ("车辆", "vehicle"),
    ("汽车", "car"),
    ("卡车", "truck"),
    ("货车", "truck"),
    ("树", "tree"),
    ("植被", "tree"),
    ("动物", "animal"),
]


def normalize_prompt(prompt):
    if not USE_PROMPT_NORMALIZE:
        return str(prompt).strip()

    p = str(prompt).strip().lower()
    p = p.replace("_", " ").replace("-", " ")
    p = " ".join(p.split())

    if p in ALIAS_MAP:
        return ALIAS_MAP[p]

    for key, canonical in PHRASE_RULES:
        if key in p:
            return canonical

    return p


def split_submission_weight(raw):
    """
    Supported formats:
    1) Recommended submission format:
       {"esam3_model.xxx": tensor, "tiny_text.xxx": tensor}

    2) Raw training checkpoint format:
       {"esam3_model": OrderedDict, "tiny_text": OrderedDict, "optimizer": ...}

    3) Legacy separate format:
       sam3.pt only contains esam3_model state_dict, and model/tiny_text.pt exists.
    """
    if not isinstance(raw, (dict, OrderedDict)):
        raise TypeError(f"Unsupported weight type: {type(raw)}")

    # Raw training checkpoint style.
    if "esam3_model" in raw and "tiny_text" in raw:
        esam3_sd = raw["esam3_model"]
        tiny_sd = raw["tiny_text"]
        return OrderedDict(esam3_sd), OrderedDict(tiny_sd)

    keys = list(raw.keys())

    # Recommended pure tensor submission style with prefixes.
    if any(str(k).startswith("esam3_model.") for k in keys) and any(str(k).startswith("tiny_text.") for k in keys):
        esam3_sd = OrderedDict()
        tiny_sd = OrderedDict()

        for k, v in raw.items():
            k = str(k)
            if k.startswith("esam3_model."):
                esam3_sd[k.replace("esam3_model.", "", 1)] = v
            elif k.startswith("tiny_text."):
                tiny_sd[k.replace("tiny_text.", "", 1)] = v

        return esam3_sd, tiny_sd

    # Legacy: sam3.pt is only image model, tiny_text.pt is separate.
    if TINY_FALLBACK_PATH.exists():
        tiny_raw = torch.load(TINY_FALLBACK_PATH, map_location="cpu")
        if isinstance(tiny_raw, dict) and "model" in tiny_raw:
            tiny_raw = tiny_raw["model"]
        return OrderedDict(raw), OrderedDict(tiny_raw)

    raise ValueError(
        "Cannot split model/sam3.pt. Expected prefixed keys "
        "'esam3_model.xxx' and 'tiny_text.xxx', or raw checkpoint containing "
        "'esam3_model' and 'tiny_text'."
    )


def build_model(device: str):
    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"model weight not found: {WEIGHT_PATH}")

    if not BPE_PATH.exists():
        raise FileNotFoundError(f"BPE vocab not found: {BPE_PATH}")

    model = build_efficientsam3_image_model(
        checkpoint_path=None,
        device=device,
        eval_mode=True,
        enable_segmentation=True,
        backbone_type="efficientvit",
        model_name="b1",
        text_encoder_type="MobileCLIP-S1",
    )

    tiny = TinyTextEncoderESAM3(
        bpe_path=str(BPE_PATH),
        token_dim=192,
        hidden_dim=256,
        num_layers=3,
        num_heads=6,
    ).to(device)

    raw = torch.load(WEIGHT_PATH, map_location="cpu")
    esam3_sd, tiny_sd = split_submission_weight(raw)

    missing, unexpected = model.load_state_dict(esam3_sd, strict=False)
    bad_missing = [x for x in missing if "language_backbone" not in x]

    print(
        f"[Load] ESAM3 missing={len(missing)}, unexpected={len(unexpected)}, "
        f"bad_missing={len(bad_missing)}",
        flush=True,
    )

    if bad_missing:
        print("[Load] bad missing examples:", bad_missing[:30], flush=True)
        raise RuntimeError("ESAM3 weight mismatch: missing non-language-backbone keys")

    # Unexpected keys are usually not acceptable for a clean submission.
    # But some training wrappers may save harmless extras, so print them instead of failing.
    if unexpected:
        print("[Load] unexpected examples:", unexpected[:30], flush=True)

    tiny_missing, tiny_unexpected = tiny.load_state_dict(tiny_sd, strict=False)
    print(
        f"[Load] TinyText missing={len(tiny_missing)}, unexpected={len(tiny_unexpected)}",
        flush=True,
    )

    if tiny_missing or tiny_unexpected:
        print("[Load] TinyText missing examples:", tiny_missing[:30], flush=True)
        print("[Load] TinyText unexpected examples:", tiny_unexpected[:30], flush=True)
        raise RuntimeError("TinyText weight mismatch")

    # Replace the heavy language backbone with the distilled tiny text encoder.
    try:
        model.backbone.language_backbone = None
    except Exception:
        pass

    model.backbone.forward_text = tiny.forward_text

    model.eval()
    tiny.eval()

    return model, tiny


def select_keep_indices(score_raw: torch.Tensor):
    if score_raw.numel() == 0:
        return None

    if SELECT_MODE == "train_like":
        keep_idx = torch.nonzero(score_raw > MASK_SCORE_THRESHOLD, as_tuple=False).view(-1)

        if keep_idx.numel() == 0:
            keep_idx = score_raw.topk(1).indices

        if keep_idx.numel() > TOPK:
            _, pos = score_raw[keep_idx].topk(TOPK)
            keep_idx = keep_idx[pos]

        return keep_idx

    # Default: use the highest scoring candidate only.
    return score_raw.topk(1).indices


def empty_mask(image_size):
    img_w, img_h = image_size
    return np.zeros((img_h, img_w), dtype=np.uint8)


def predict_one_prompt(
    model,
    tiny,
    processor,
    image_features,
    prompt: str,
    image_size,
    device: str,
):
    img_w, img_h = image_size

    with torch.inference_mode():
        text_outputs = tiny.forward_text([prompt], device=device)

        # One image feature, different text features.
        backbone_out = dict(image_features)
        backbone_out.update(text_outputs)

        with amp_context(device):
            out = model.forward_grounding(
                backbone_out=backbone_out,
                find_input=processor.find_stage,
                geometric_prompt=model._get_dummy_prompt(),
                find_target=None,
            )

        pred_masks = out["pred_masks"]
        pred_logits = out["pred_logits"]
        presence = out.get("presence_logit_dec", None)

        score_raw = pred_logits.sigmoid().squeeze(-1)[0].float()

        if score_raw.numel() == 0:
            return empty_mask(image_size), {
                "exists": False,
                "reason": "no_candidate",
                "max_score": 0.0,
                "presence": 0.0,
                "exist_score": 0.0,
                "area_ratio": 0.0,
            }

        max_score = float(score_raw.max().detach().cpu().item())

        if presence is not None:
            presence_score = float(presence.sigmoid().mean().detach().cpu().item())
        else:
            presence_score = 1.0

        exist_score = max_score * presence_score

        if exist_score < EXIST_THRESHOLD:
            return empty_mask(image_size), {
                "exists": False,
                "reason": "low_exist_score",
                "max_score": max_score,
                "presence": presence_score,
                "exist_score": exist_score,
                "area_ratio": 0.0,
            }

        keep_idx = select_keep_indices(score_raw)
        if keep_idx is None or keep_idx.numel() == 0:
            return empty_mask(image_size), {
                "exists": False,
                "reason": "no_keep",
                "max_score": max_score,
                "presence": presence_score,
                "exist_score": exist_score,
                "area_ratio": 0.0,
            }

        selected_masks = pred_masks[0, keep_idx]

        selected_masks = F.interpolate(
            selected_masks[:, None, :, :],
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        bin_masks = selected_masks > MASK_THRESHOLD

        # Semantic union: one ann_id only outputs one merged mask.
        union = bin_masks.any(dim=0).squeeze(0)

        mask = union.detach().cpu().numpy().astype(np.uint8)
        area_ratio = float(mask.sum() / max(1, img_h * img_w))

        return mask, {
            "exists": bool(mask.sum() > 0),
            "reason": "ok",
            "max_score": max_score,
            "presence": presence_score,
            "exist_score": exist_score,
            "area_ratio": area_ratio,
        }


def read_tasks():
    if not TASK_PATH.exists():
        raise FileNotFoundError(f"test_tasks.json not found: {TASK_PATH}")

    with open(TASK_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    tasks = normalize_tasks(raw)

    if not isinstance(tasks, list):
        raise ValueError("normalized tasks must be a list")

    return tasks


def group_tasks(tasks):
    grouped = defaultdict(list)

    for idx, task in enumerate(tasks):
        ann_id = get_first(task, ["ann_id", "id", "task_id", "annotation_id"])
        image_rel = get_first(task, ["image_path", "image", "img_path", "img", "file_name", "filename", "path"])
        prompt_raw = get_first(task, ["text_prompt", "prompt", "text", "phrase", "category", "class_name", "label"], "")

        if ann_id is None:
            raise ValueError(f"bad task: missing ann_id, task={task}")
        if image_rel is None:
            raise ValueError(f"bad task: missing image_path, task={task}")

        grouped[str(image_rel)].append({
            "order": idx,
            "ann_id": ann_id,
            "prompt_raw": prompt_raw,
            "prompt": normalize_prompt(prompt_raw),
        })

    return grouped


def run():
    torch.set_grad_enabled(False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tasks = read_tasks()
    grouped = group_tasks(tasks)

    print(
        f"[Start] device={device}, tasks={len(tasks)}, images={len(grouped)}, "
        f"resolution={RESOLUTION}, exist_thr={EXIST_THRESHOLD}, mask_thr={MASK_THRESHOLD}, "
        f"select_mode={SELECT_MODE}, prompt_normalize={USE_PROMPT_NORMALIZE}",
        flush=True,
    )

    model, tiny = build_model(device)

    processor = Sam3Processor(
        model,
        resolution=RESOLUTION,
        device=device,
        confidence_threshold=0.0,
    )

    predictions_by_order = [None] * len(tasks)

    stats = {
        "num_tasks": len(tasks),
        "num_images": len(grouped),
        "num_positive": 0,
        "num_black": 0,
        "num_low_exist_score": 0,
        "max_score_sum": 0.0,
        "presence_sum": 0.0,
        "exist_score_sum": 0.0,
        "area_ratio_sum": 0.0,
    }

    for img_idx, (image_rel, items) in enumerate(grouped.items(), start=1):
        image_path = resolve_image_path(image_rel)
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")

        image_pil = load_ir_as_rgb(image_path)
        image_size = image_pil.size

        image_tensor = processor.transform(
            v2.functional.to_image(image_pil).to(device)
        ).unsqueeze(0)

        # Key optimization: one image only runs visual encoder once.
        with torch.inference_mode():
            with amp_context(device):
                image_features = model.backbone.forward_image(image_tensor)

        for item in items:
            mask, info = predict_one_prompt(
                model=model,
                tiny=tiny,
                processor=processor,
                image_features=image_features,
                prompt=item["prompt"],
                image_size=image_size,
                device=device,
            )

            stats["max_score_sum"] += float(info["max_score"])
            stats["presence_sum"] += float(info["presence"])
            stats["exist_score_sum"] += float(info["exist_score"])
            stats["area_ratio_sum"] += float(info["area_ratio"])

            if info["exists"] and mask.sum() > 0:
                stats["num_positive"] += 1
            else:
                stats["num_black"] += 1
                if info["reason"] == "low_exist_score":
                    stats["num_low_exist_score"] += 1

            predictions_by_order[item["order"]] = {
                "ann_id": item["ann_id"],
                "rle": encode_rle(mask),
            }

        del image_features, image_tensor

        if device == "cuda":
            torch.cuda.empty_cache()

        if img_idx % 50 == 0 or img_idx == len(grouped):
            print(f"[Progress] {img_idx}/{len(grouped)} images", flush=True)

    missing = [i for i, x in enumerate(predictions_by_order) if x is None]
    if missing:
        raise RuntimeError(f"missing predictions for task indices: {missing[:20]}")

    denom = max(1, len(tasks))
    stats["avg_max_score"] = stats["max_score_sum"] / denom
    stats["avg_presence"] = stats["presence_sum"] / denom
    stats["avg_exist_score"] = stats["exist_score_sum"] / denom
    stats["avg_area_ratio"] = stats["area_ratio_sum"] / denom

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"predictions": predictions_by_order}, f, ensure_ascii=False)

    print(f"[Stats] {stats}", flush=True)
    print(f"[Done] saved predictions to {OUT_PATH}, num={len(predictions_by_order)}", flush=True)


if __name__ == "__main__":
    run()
