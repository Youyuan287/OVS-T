#!/usr/bin/env bash
set -euo pipefail

PROJECT=${PROJECT:-/home/Groups/group2/Working/tyy/project/efficientsam3}
IMAGE_ROOT=${IMAGE_ROOT:-/home/Groups/group2/Working/TJY/sam3_ir_test/data/ir_images/data1}
OUT=${OUT:-/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_data1_dedup5000_sam3_qwen_current}
PY=${PY:-/home/Groups/group2/Working/seg/miniconda3/envs/esam3_312/bin/python}
QWEN_PY=${QWEN_PY:-/home/Groups/group2/Working/seg/miniconda3/envs/thgs/bin/python}
QWEN_MODEL=${QWEN_MODEL:-/home/Groups/group2/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct}
MAX_IMAGES=${MAX_IMAGES:-5000}
SEED=${SEED:-33}
MAX_SCAN=${MAX_SCAN:-0}
SAM3_MAX_ITEMS=${SAM3_MAX_ITEMS:-0}
QWEN_MAX_ITEMS=${QWEN_MAX_ITEMS:-0}

if [ -e "$OUT" ] && [ "${ALLOW_EXISTING_OUT:-0}" != "1" ]; then
  echo "Output directory already exists: $OUT" >&2
  echo "Set OUT to a new path, or set ALLOW_EXISTING_OUT=1 for an intentional resume." >&2
  exit 2
fi

mkdir -p "$OUT"
cd "$PROJECT"

echo "[1/6] Deduplicate and select images"
"$PY" scripts/10_dedupe_select_data1.py \
  --image_root "$IMAGE_ROOT" \
  --out_dir "$OUT" \
  --max_images "$MAX_IMAGES" \
  --seed "$SEED" \
  ${MAX_SCAN:+--max_scan "$MAX_SCAN"}

echo "[2/6] Build prompt proposals"
"$PY" scripts/02_build_prompt_proposals.py \
  --prompt_bank data/prompt_bank_ir_v2.json \
  --image_list "$OUT/dedup_image_list.txt" \
  --out_jsonl "$OUT/prompt_proposals.jsonl" \
  --max_images "$MAX_IMAGES" \
  --fallback_classes 12 \
  --default_scene_type urban_scene \
  --summary "$OUT/prompt_proposals_summary.json"

echo "[3/6] Run SAM3 text-only proposals"
"$PY" scripts/03_run_sam3_multi_prompt.py \
  --proposals "$OUT/prompt_proposals.jsonl" \
  --prompt_bank data/prompt_bank_ir_v2.json \
  --out_dir "$OUT" \
  --submission_ckpt submit_epoch4_best/model/sam3.pt \
  --max_items "$SAM3_MAX_ITEMS" \
  --prompts_per_class 4 \
  --modes text_only \
  --threshold 0.35 \
  --resolution 768 \
  --device cuda

echo "[4/6] Filter non-empty SAM3 candidates before Qwen"
"$PY" scripts/11_filter_sam3_nonempty_candidates.py \
  --candidates "$OUT/sam3_candidates.jsonl" \
  --out_jsonl "$OUT/sam3_candidates_nonempty_filtered.jsonl" \
  --summary "$OUT/sam3_candidates_nonempty_filtered_summary.json" \
  --iou_threshold 0.90

echo "[5/6] Build Qwen panels and run Qwen QC"
"$PY" scripts/04_build_qwen_qc_panels.py \
  --candidates "$OUT/sam3_candidates_nonempty_filtered.jsonl" \
  --out_dir "$OUT"

"$QWEN_PY" scripts/05_run_qwen8b_qc.py \
  --tasks "$OUT/qwen_qc_tasks.jsonl" \
  --out_jsonl "$OUT/qwen_qc_results.jsonl" \
  --local_qwen_model "$QWEN_MODEL" \
  --max_items "$QWEN_MAX_ITEMS"

echo "[6/6] Merge manifest with dedup-group split"
"$PY" scripts/06_merge_filter_manifest.py \
  --candidates "$OUT/sam3_candidates_nonempty_filtered.jsonl" \
  --qwen "$OUT/qwen_qc_results.jsonl" \
  --out_dir "$OUT" \
  --max_per_class 3000 \
  --target_total 20000 \
  --val_ratio 0.1 \
  --seed "$SEED" \
  --dedup_groups "$OUT/dedup_groups.jsonl"

echo "Done: $OUT"
