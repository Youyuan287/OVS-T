#!/usr/bin/env python3
"""HTTP server for interactive EfficientSAM3 inference.

The server intentionally uses only the Python standard library for HTTP so it
can run in the existing esam3_312 environment without installing web packages.
It keeps the SAM3 model loaded and writes all interactive outputs under
outputs/interactive_web_runs/<run_id>/.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_panel(image_path: Path, mask_path: Path, row: Dict[str, Any], out_path: Path, tile: int = 448) -> None:
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    mask_img = Image.open(mask_path).convert("L")
    if mask_img.size != image.size:
        mask_img = mask_img.resize(image.size, Image.Resampling.NEAREST)
    mask = (np.array(mask_img) > 127).astype(np.uint8)

    def fit(img: Image.Image) -> Image.Image:
        img = ImageOps.contain(img.convert("RGB"), (tile, tile))
        canvas = Image.new("RGB", (tile, tile), (0, 0, 0))
        canvas.paste(img, ((tile - img.width) // 2, (tile - img.height) // 2))
        return canvas

    def label(img: Image.Image, text: str) -> Image.Image:
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, img.width, 24), fill=(0, 0, 0))
        draw.text((6, 5), text, fill=(255, 255, 255))
        return img

    img_np = np.array(image.convert("RGB")).astype(np.float32)
    color = np.zeros_like(img_np)
    color[..., 0] = 255
    color[..., 1] = 80
    m = mask > 0
    overlay = img_np.copy()
    overlay[m] = overlay[m] * 0.55 + color[m] * 0.45
    overlay_img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    masked = np.zeros_like(np.array(image.convert("RGB")))
    masked[m] = np.array(image.convert("RGB"))[m]
    masked_img = Image.fromarray(masked)

    parts = [
        label(fit(image), "original infrared image"),
        label(fit(overlay_img), "mask overlay"),
        label(fit(image.copy()), f"prompt: {row.get('raw_prompt', '')}"[:80]),
        label(fit(masked_img), "masked region only"),
    ]
    panel = Image.new("RGB", (tile * 2, tile * 2), (0, 0, 0))
    panel.paste(parts[0], (0, 0))
    panel.paste(parts[1], (tile, 0))
    panel.paste(parts[2], (0, tile))
    panel.paste(parts[3], (tile, tile))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path, quality=92)


class Sam3Server:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo = Path(args.repo).resolve()
        sys.path.insert(0, str(self.repo))
        self.mod = load_module(self.repo / "scripts" / "03_run_sam3_multi_prompt.py", "sam3_multi_prompt")
        build_args = SimpleNamespace(
            device=args.device,
            resolution=args.resolution,
            threshold=args.threshold,
            submission_ckpt=args.submission_ckpt,
            esam3_ckpt="",
            tiny_ckpt="",
            bpe_path=args.bpe_path,
        )
        self.model_bundle = self.mod.build_model(build_args)
        self.out_root = Path(args.out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)

    def segment(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        image_path = Path(payload["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(str(image_path))
        prompts = payload.get("prompts") or [payload.get("prompt", "object")]
        prompts = [str(p).strip() for p in prompts if str(p).strip()]
        if not prompts:
            raise ValueError("No prompts provided")
        run_id = str(payload.get("run_id") or self.mod.safe_id(image_path, prompts))
        threshold = float(payload.get("threshold", self.args.threshold))
        scene_dir = str(payload.get("scene_dir") or image_path.parent.name)
        scene_type = str(payload.get("scene_type") or "interactive")
        modes = [m.strip() for m in str(payload.get("modes") or "text_only").split(",") if m.strip()]

        run_dir = self.out_root / run_id
        mask_dir = run_dir / "sam3_masks"
        overlay_dir = run_dir / "sam3_overlays"
        panel_dir = run_dir / "qwen_qc_panels"
        image = self.mod.load_ir_as_rgb(image_path)
        candidates: List[Dict[str, Any]] = []
        torch, v2, model, processor, device = self.model_bundle

        for prompt in prompts:
            canonical = str(payload.get("canonical") or prompt).lower()
            for mode in modes:
                if mode != "text_only":
                    continue
                cid = self.mod.safe_id(image_path, canonical, prompt, mode, threshold, run_id)
                mask_path = mask_dir / canonical / f"{image_path.stem}_{cid}.png"
                overlay_path = overlay_dir / canonical / f"{image_path.stem}_{cid}.jpg"
                panel_path = panel_dir / canonical / f"{cid}.jpg"
                mask, score = self.mod.predict_mask(torch, v2, model, processor, device, image, prompt, threshold)
                self.mod.save_mask(mask, mask_path)
                self.mod.save_overlay(image, mask, overlay_path)
                metrics = self.mod.mask_metrics(mask)
                row = {
                    "candidate_id": cid,
                    "image": str(image_path),
                    "scene_dir": scene_dir,
                    "scene_type": scene_type,
                    "canonical_prompt": canonical,
                    "raw_prompt": prompt,
                    "source_mode": mode,
                    "sam3_score": score,
                    "mask": str(mask_path),
                    "overlay": str(overlay_path),
                    "panel": str(panel_path),
                    **metrics,
                }
                build_panel(image_path, mask_path, row, panel_path)
                candidates.append(row)

        result = {"ok": True, "run_id": run_id, "out_dir": str(run_dir), "candidates": candidates}
        with (run_dir / "sam3_interactive_result.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result


def make_handler(server: Sam3Server):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: Dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200, {"ok": True, "service": "sam3", "device": server.args.device})
            else:
                self._json(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path == "/segment":
                    self._json(200, server.segment(payload))
                else:
                    self._json(404, {"ok": False, "error": "not found"})
            except Exception as exc:
                self._json(500, {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()[-4000:]})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[sam3] {self.address_string()} {fmt % args}", flush=True)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/home/Groups/group2/Working/tyy/project/efficientsam3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--submission_ckpt", default="submit_epoch4_best/model/sam3.pt")
    parser.add_argument("--bpe_path", default="sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    parser.add_argument("--out_root", default="/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/interactive_web_runs")
    args = parser.parse_args()
    args.submission_ckpt = str((Path(args.repo) / args.submission_ckpt).resolve())
    args.bpe_path = str((Path(args.repo) / args.bpe_path).resolve())
    server = Sam3Server(args)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(server))
    print(f"SAM3 server listening on {args.host}:{args.port}", flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
