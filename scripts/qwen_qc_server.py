#!/usr/bin/env python3
"""HTTP server for interactive Qwen3-VL mask quality critique."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QwenServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo = Path(args.repo).resolve()
        sys.path.insert(0, str(self.repo))
        self.mod = load_module(self.repo / "scripts" / "05_run_qwen8b_qc.py", "qwen_qc")
        self.bundle = self.mod.load_local_qwen(args.model)

    def qc(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        panel = Path(payload["panel_path"])
        if not panel.exists():
            raise FileNotFoundError(str(panel))
        task = {
            "candidate_id": str(payload.get("candidate_id") or panel.stem),
            "panel": str(panel),
            "target_class": str(payload["target_class"]),
            "scene_dir": str(payload.get("scene_dir", "")),
            "scene_type": str(payload.get("scene_type", "")),
            "raw_prompt": str(payload.get("raw_prompt", "")),
            "sam3_score": float(payload.get("sam3_score", 0.0) or 0.0),
            "area_ratio": float(payload.get("area_ratio", 0.0) or 0.0),
        }
        raw, raw_text = self.mod.run_local_qwen(self.bundle, task, int(payload.get("max_new_tokens", self.args.max_new_tokens)))
        if raw is None:
            raise RuntimeError(f"Qwen output was not parseable: {raw_text[-500:]}")
        row = self.mod.normalize_qc(raw, task, False)
        row["raw_text_tail"] = raw_text
        return {"ok": True, "qc": row}


def make_handler(server: QwenServer):
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
                self._json(200, {"ok": True, "service": "qwen", "model": server.args.model})
            else:
                self._json(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path == "/qc":
                    self._json(200, server.qc(payload))
                else:
                    self._json(404, {"ok": False, "error": "not found"})
            except Exception as exc:
                self._json(500, {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()[-4000:]})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[qwen] {self.address_string()} {fmt % args}", flush=True)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/home/Groups/group2/Working/tyy/project/efficientsam3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--model", default="/home/Groups/group2/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()
    server = QwenServer(args)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(server))
    print(f"Qwen server listening on {args.host}:{args.port}", flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
