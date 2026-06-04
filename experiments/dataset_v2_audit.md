# Dataset V2 审计记录

## 当前状态

- 状态：脚本骨架已建立，等待 pilot 运行。
- Python 环境：`esam3_312`。
- 输出目录：`/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_prompt_sam3_qwen8b`。

## IRGPT 推理状态

- 官方 WheatCao/ICCV2025-IRGPT 仓库暂无稳定推理脚本。
- 本版审计重点从 IRGPT 生成质量转为 prompt proposal 覆盖率、SAM3 候选 mask 质量和 Qwen 质检有效性。

## 需要记录的统计

| 阶段 | 指标 | 结果 | 结论 |
|---|---|---:|---|
| Prompt bank | canonical 类别数 | 待填 | 待填 |
| Prompt bank | prompt 总数 | 待填 | 待填 |
| Prompt proposal | 图像数 | 待填 | 待填 |
| Prompt proposal | candidate 数 | 待填 | 待填 |
| 外部 proposal worker | 空输出/乱码率 | 不启用时填 0 | 默认不依赖 IRGPT 推理 |
| SAM3 candidates | 候选 mask 数 | 待填 | 待填 |
| SAM3 candidates | 非空 mask 比例 | 待填 | 待填 |
| Qwen QC | JSON 解析成功率 | 待填 | 待填 |
| Qwen QC | fallback 比例 | 待填 | 待填 |
| Manifest | `train_hq` 样本数 | 待填 | 待填 |
| Manifest | `val_hq` 样本数 | 待填 | 待填 |
| Manifest | 电力类样本数 | 待填 | 待填 |

## 人工抽检重点

- 小目标是否被保留：`person`、`animal`、`insulator`。
- 细长结构是否被误删：`power line`、`wire`、`pole`。
- 大区域是否吞掉前景目标：`road`、`building`、`tree`。
- Qwen 是否能识别明显类别不匹配或背景泄漏。
- 中低置信样本是否被错误当作负样本。

## Pilot 结论

待运行后填写。
