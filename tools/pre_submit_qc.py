#!/usr/bin/env python3
"""提交前推理质量门禁。

这个脚本不预测线上分数，只检查提交包是否存在明显格式或推理失败风险。
它可以直接调用 submit_epoch4_best/inference.py，也可以在 --skip_inference
模式下只检查已有 predictions.json。
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from pycocotools import mask as mask_utils
except Exception:  # pragma: no cover
    mask_utils = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EfficientSAM3 提交前 QC 门禁")
    parser.add_argument("--code_root", default="submit_epoch4_best", help="提交代码目录")
    parser.add_argument("--test_root", default=None, help="测试任务目录，目录中应包含 test_tasks.json")
    parser.add_argument("--tasks_json", default=None, help="官方 test_tasks.json 或自定义 smoke tasks 路径")
    parser.add_argument("--predictions", default=None, help="已有 predictions.json；配合 --skip_inference 使用")
    parser.add_argument("--out_dir", default="experiments/pre_submit_qc", help="QC 输出目录")
    parser.add_argument("--skip_inference", action="store_true", help="跳过推理，只检查已有 predictions.json")
    parser.add_argument("--env", action="append", default=[], help="额外环境变量，格式 KEY=VALUE，可重复")
    parser.add_argument("--large_area_ratio", type=float, default=0.70, help="超大 mask 面积比例阈值")
    parser.add_argument("--fail_on_warning", action="store_true", help="有警告也返回非零退出码")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_tasks(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        for key in ("tasks", "annotations", "data"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError("test_tasks.json 必须是列表，或包含 tasks/annotations/data 列表字段")

    tasks: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"第 {idx} 条任务不是 JSON object")
        ann_id = item.get("ann_id", item.get("id", idx))
        image_path = item.get("image_path") or item.get("file_name") or item.get("image")
        prompt = item.get("text") or item.get("prompt") or item.get("phrase") or item.get("category_name") or "object"
        tasks.append({"ann_id": str(ann_id), "image_path": image_path, "prompt": prompt, "raw": item})
    return tasks


def prepare_test_root(args: argparse.Namespace, run_dir: Path) -> Tuple[Path, Path, List[Dict[str, Any]]]:
    if args.tasks_json:
        tasks_path = Path(args.tasks_json).expanduser().resolve()
        raw = read_json(tasks_path)
        tasks = normalize_tasks(raw)
        test_root = Path(args.test_root).expanduser().resolve() if args.test_root else tasks_path.parent
        work_test_root = run_dir / "test_root"
        work_test_root.mkdir(parents=True, exist_ok=True)
        (work_test_root / "test_tasks.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        return work_test_root, work_test_root / "test_tasks.json", tasks

    if not args.test_root:
        raise ValueError("必须提供 --test_root 或 --tasks_json")
    test_root = Path(args.test_root).expanduser().resolve()
    tasks_path = test_root / "test_tasks.json"
    raw = read_json(tasks_path)
    return test_root, tasks_path, normalize_tasks(raw)


def parse_env(items: Iterable[str]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"环境变量格式错误: {item}，应为 KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"环境变量名为空: {item}")
        env[key] = value
    return env


def run_inference(code_root: Path, test_root: Path, run_dir: Path, extra_env: Dict[str, str]) -> Path:
    env = os.environ.copy()
    env.update(extra_env)
    env["RAYTRON_CODE"] = str(code_root.resolve())
    env["RAYTRON_TEST"] = str(test_root.resolve())

    cmd = [sys.executable, str((code_root / "inference.py").resolve())]
    proc = subprocess.run(cmd, env=env, cwd=str(code_root.parent.resolve()), text=True, capture_output=True)
    (run_dir / "inference_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (run_dir / "inference_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"推理脚本返回非零退出码 {proc.returncode}，详见 {run_dir}")

    pred_path = test_root / "predictions.json"
    if not pred_path.exists():
        alt_path = code_root / "predictions.json"
        if alt_path.exists():
            pred_path = alt_path
        else:
            raise FileNotFoundError("推理完成但未找到 predictions.json")
    return pred_path


def decode_rle(rle: Dict[str, Any]) -> Tuple[Any, str | None]:
    if mask_utils is not None:
        try:
            return mask_utils.decode(rle), None
        except Exception as exc:
            return None, f"pycocotools 解码失败: {exc}"

    counts = rle.get("counts")
    size = rle.get("size")
    if np is None or not isinstance(counts, list) or not isinstance(size, list) or len(size) != 2:
        return None, "缺少 pycocotools，且 RLE 不是未压缩 list counts，无法本地解码"
    h, w = int(size[0]), int(size[1])
    flat = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    value = 0
    for run in counts:
        run = int(run)
        if value == 1 and run > 0:
            flat[idx : idx + run] = 1
        idx += run
        value = 1 - value
    if idx != h * w:
        return None, f"未压缩 RLE 长度不匹配: {idx} != {h*w}"
    return flat.reshape((h, w), order="F"), None


def image_size_for_task(task: Dict[str, Any], test_root: Path) -> Tuple[int | None, int | None, str | None]:
    raw = task.get("raw", {})
    for h_key, w_key in (("height", "width"), ("h", "w")):
        if h_key in raw and w_key in raw:
            return int(raw[h_key]), int(raw[w_key]), None
    image_path = task.get("image_path")
    if not image_path:
        return None, None, "任务缺少 image_path，也没有 height/width"
    p = Path(str(image_path))
    if not p.is_absolute():
        p = test_root / p
    if Image is None:
        return None, None, "缺少 PIL，无法读取图片尺寸"
    if not p.exists():
        return None, None, f"图片不存在: {p}"
    with Image.open(p) as img:
        w, h = img.size
    return h, w, None


def extract_inference_stats(run_dir: Path) -> Dict[str, Any] | None:
    stdout_path = run_dir / "inference_stdout.log"
    if not stdout_path.exists():
        return None
    for line in stdout_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("[Stats]"):
            text = line[len("[Stats]") :].strip()
            try:
                return ast.literal_eval(text)
            except Exception:
                return {"raw": text}
    return None


def validate_predictions(
    pred_path: Path,
    tasks: List[Dict[str, Any]],
    test_root: Path,
    large_area_ratio: float,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    raw_preds = read_json(pred_path)
    if isinstance(raw_preds, dict):
        for key in ("predictions", "annotations", "results"):
            if key in raw_preds and isinstance(raw_preds[key], list):
                raw_preds = raw_preds[key]
                break
    if not isinstance(raw_preds, list):
        raise ValueError("predictions.json 必须是列表，或包含 predictions/annotations/results 列表字段")

    expected_ids = [t["ann_id"] for t in tasks]
    expected_counter = Counter(expected_ids)
    pred_ids = [str(p.get("ann_id", p.get("id", ""))) for p in raw_preds if isinstance(p, dict)]
    pred_counter = Counter(pred_ids)

    errors: List[str] = []
    warnings: List[str] = []
    missing = sorted([ann_id for ann_id in expected_counter if pred_counter.get(ann_id, 0) == 0])
    extra = sorted([ann_id for ann_id in pred_counter if expected_counter.get(ann_id, 0) == 0])
    duplicate = sorted([ann_id for ann_id, count in pred_counter.items() if count > expected_counter.get(ann_id, 0)])
    if missing:
        errors.append(f"缺少 ann_id: {missing[:20]}{'...' if len(missing) > 20 else ''}")
    if extra:
        errors.append(f"存在额外 ann_id: {extra[:20]}{'...' if len(extra) > 20 else ''}")
    if duplicate:
        errors.append(f"ann_id 重复: {duplicate[:20]}{'...' if len(duplicate) > 20 else ''}")

    task_by_id = {t["ann_id"]: t for t in tasks}
    area_ratios: List[float] = []
    empty_count = 0
    decode_failed = 0
    size_mismatch = 0
    large_count = 0
    checked = 0

    for idx, pred in enumerate(raw_preds):
        if not isinstance(pred, dict):
            errors.append(f"第 {idx} 条 prediction 不是 JSON object")
            continue
        ann_id = str(pred.get("ann_id", pred.get("id", "")))
        seg = pred.get("segmentation") or pred.get("rle") or pred.get("mask")
        if not isinstance(seg, dict) or "counts" not in seg or "size" not in seg:
            errors.append(f"ann_id={ann_id} 缺少有效 RLE segmentation")
            continue
        mask, decode_error = decode_rle(seg)
        if decode_error:
            decode_failed += 1
            warnings.append(f"ann_id={ann_id} {decode_error}")
            continue
        checked += 1
        h, w = int(mask.shape[0]), int(mask.shape[1])
        task = task_by_id.get(ann_id)
        if task is not None:
            img_h, img_w, img_error = image_size_for_task(task, test_root)
            if img_error:
                warnings.append(f"ann_id={ann_id} {img_error}")
            elif img_h is not None and img_w is not None and (h != img_h or w != img_w):
                size_mismatch += 1
                errors.append(f"ann_id={ann_id} mask 尺寸 {h}x{w} != 图像尺寸 {img_h}x{img_w}")
        area = float(mask.sum())
        ratio = area / float(max(1, h * w))
        area_ratios.append(ratio)
        if area <= 0:
            empty_count += 1
        if ratio >= large_area_ratio:
            large_count += 1

    if decode_failed == len(raw_preds) and raw_preds:
        warnings.append("所有 RLE 都未能解码，建议在提交前安装 pycocotools 后复查 mask 尺寸和面积")

    stats = {
        "任务数量": len(tasks),
        "预测数量": len(raw_preds),
        "唯一 ann_id 数": len(pred_counter),
        "缺失 ann_id 数": len(missing),
        "额外 ann_id 数": len(extra),
        "重复 ann_id 数": len(duplicate),
        "已解码 mask 数": checked,
        "RLE 解码失败数": decode_failed,
        "mask 尺寸不匹配数": size_mismatch,
        "空 mask 数": empty_count,
        "空 mask 比例": round(empty_count / max(1, len(raw_preds)), 6),
        "平均面积比例": round(sum(area_ratios) / max(1, len(area_ratios)), 6),
        "最大面积比例": round(max(area_ratios) if area_ratios else 0.0, 6),
        "超大 mask 数": large_count,
        "超大 mask 阈值": large_area_ratio,
    }
    return stats, errors, warnings


def write_reports(run_dir: Path, summary: Dict[str, Any]) -> None:
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 提交前 QC 摘要",
        "",
        f"- 时间：{summary['时间']}",
        f"- 结论：{summary['结论']}",
        f"- 推理脚本：{summary['推理脚本']}",
        f"- 测试任务：{summary['测试任务']}",
        f"- 预测文件：{summary['预测文件']}",
        "",
        "## 关键统计",
        "",
    ]
    for key, value in summary["统计"].items():
        lines.append(f"- {key}：{value}")
    if summary.get("推理统计") is not None:
        lines.extend(["", "## 推理脚本统计", "", "```json", json.dumps(summary["推理统计"], ensure_ascii=False, indent=2), "```"])
    if summary["错误"]:
        lines.extend(["", "## 错误", ""])
        lines.extend([f"- {item}" for item in summary["错误"]])
    if summary["警告"]:
        lines.extend(["", "## 警告", ""])
        lines.extend([f"- {item}" for item in summary["警告"][:50]])
        if len(summary["警告"]) > 50:
            lines.append(f"- 其余 {len(summary['警告']) - 50} 条见 summary.json")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    code_root = Path(args.code_root).expanduser().resolve()
    test_root, tasks_path, tasks = prepare_test_root(args, run_dir)
    extra_env = parse_env(args.env)

    if args.skip_inference:
        pred_path = Path(args.predictions).expanduser().resolve() if args.predictions else test_root / "predictions.json"
        if not pred_path.exists():
            raise FileNotFoundError(f"未找到 predictions.json: {pred_path}")
    else:
        pred_path = run_inference(code_root, test_root, run_dir, extra_env)

    stats, errors, warnings = validate_predictions(pred_path, tasks, test_root, args.large_area_ratio)
    infer_stats = extract_inference_stats(run_dir)
    conclusion = "通过" if not errors and (not warnings or not args.fail_on_warning) else "不通过"
    summary = {
        "时间": timestamp,
        "结论": conclusion,
        "推理脚本": str(code_root / "inference.py"),
        "测试任务": str(tasks_path),
        "预测文件": str(pred_path),
        "额外环境变量": extra_env,
        "统计": stats,
        "推理统计": infer_stats,
        "错误": errors,
        "警告": warnings,
        "报告目录": str(run_dir),
    }
    write_reports(run_dir, summary)

    print(json.dumps({"结论": conclusion, "报告目录": str(run_dir), "统计": stats, "错误数": len(errors), "警告数": len(warnings)}, ensure_ascii=False, indent=2))
    if errors or (warnings and args.fail_on_warning):
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[QC错误] {exc}", file=sys.stderr)
        raise SystemExit(1)
