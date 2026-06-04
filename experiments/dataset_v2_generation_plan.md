# Dataset V2 生成计划

目标：在不覆盖当前 61 分基线数据的前提下，重新制作一版红外高置信伪监督数据集。

输出目录：

```text
/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_prompt_sam3_qwen8b
```


## IRGPT 状态修正

- WheatCao/ICCV2025-IRGPT 当前主要发布 IR-TD 数据、json_sft 模板和测试工具，缺少官方稳定推理入口。
- 第一版 Dataset V2 不再把 IRGPT 作为实际 proposal generator。
- IRGPT/IR-TD 只用于参考红外任务模板、类别表达和长短语设计。
- 真实伪标签生成主路径为：扩展 prompt bank -> SAM3 多提示候选 mask -> Qwen3-VL-8B 质检 -> 类别自适应 QC。

## 流程

1. 构建红外扩展词表：

```bash
conda run -n esam3_312 python scripts/01_build_prompt_bank.py \
  --out data/prompt_bank_ir_v2.json
```

2. 生成 IRGPT proposal。若 IRGPT 入口暂未稳定，可先不传 `--irgpt_command`，脚本会生成低置信 fallback proposal，用于打通 SAM3/Qwen/manifest 流程。

```bash
conda run -n esam3_312 python scripts/02_build_prompt_proposals.py \
  --prompt_bank data/prompt_bank_ir_v2.json \
  --max_images 500
```

3. 生成 SAM3 候选 mask。先用 `--dry_run` 检查任务展开数量；正式运行时填写权重路径。

```bash
conda run -n esam3_312 python scripts/03_run_sam3_multi_prompt.py \
  --proposals /home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_prompt_sam3_qwen8b/irgpt_proposals.jsonl \
  --prompt_bank data/prompt_bank_ir_v2.json \
  --dry_run
```

4. 生成 Qwen3-VL-8B 四宫格质检面板：

```bash
conda run -n esam3_312 python scripts/04_build_qwen_qc_panels.py \
  --candidates /home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_prompt_sam3_qwen8b/sam3_candidates.jsonl
```

5. 运行 Qwen3-VL-8B 质检。第一版推荐接入真实 `--qwen_command`；若只联调流程，可用 `--rule_fallback`。

```bash
conda run -n esam3_312 python scripts/05_run_qwen8b_qc.py \
  --tasks /home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_prompt_sam3_qwen8b/qwen_qc_tasks.jsonl \
  --rule_fallback
```

6. 融合 SAM3、Qwen 和类别规则，生成高置信 manifest：

```bash
conda run -n esam3_312 python scripts/06_merge_filter_manifest.py \
  --candidates /home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_prompt_sam3_qwen8b/sam3_candidates.jsonl \
  --qwen /home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_prompt_sam3_qwen8b/qwen_qc_results.jsonl
```

7. 生成存在性校准数据：

```bash
conda run -n esam3_312 python scripts/07_make_exist_calib_v2.py \
  --train_hq /home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_prompt_sam3_qwen8b/train_hq.jsonl \
  --val_hq /home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_prompt_sam3_qwen8b/val_hq.jsonl
```

## Pilot 门禁

- 先跑 200-500 张图，不直接全量。
- 人工抽查至少 100 张 overlay 和 Qwen 面板。
- 重点检查 `person`、`vehicle`、`power line`、`pole`、`insulator`。
- 低置信样本只丢弃或进入 review，不作为背景负样本。
- `train_hq` 和 `val_hq` 必须按 image 分组切分，无同图泄漏。

## 决策

- 若 pilot 中 Qwen 能明显过滤错误 mask，再扩到 5k-10k 样本。
- 若 Qwen JSON 解析失败率高，先修 Qwen worker，不进入训练。
- 若电力类仍接近 0，优先扩 prompt bank、增加多尺度 SAM3 生成策略和类别规则，不继续训练。
