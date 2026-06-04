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
## 2026-06-04 SAM3-only 小规模 Pilot

- 输出目录：/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_prompt_sam3_qwen8b_pilot_current
- 使用环境：/home/Groups/group2/Working/seg/miniconda3/envs/esam3_312/bin/python
- 使用权重：submit_epoch4_best/model/sam3.pt
- Proposal：3 张图，36 条 prompt proposal。
- SAM3 生成：处理前 18 条 proposal，每类前 4 个 prompt，仅 	ext_only 模式。
- 候选 mask：72 个，其中 28 个非空。
- Qwen 面板：72 个，任务文件 qwen_qc_tasks.jsonl 已生成。

| 类别 | 候选数 | 非空数 | 非空平均面积比例 | 观察 |
|---|---:|---:|---:|---|
| road | 8 | 8 | 0.297274 | 大区域召回稳定，需要 Qwen 检查是否吞前景。 |
| vehicle | 8 | 6 | 0.013710 | 车辆类可作为第一批有效类别。 |
| car | 8 | 5 | 0.011912 | 与 vehicle 有重叠，后续需要同义冲突合并。 |
| truck | 8 | 3 | 0.010699 | 有召回，但需确认是否真实 truck。 |
| building | 8 | 4 | 0.217121 | 大面积类别可出 mask。 |
| person | 8 | 1 | 0.000351 | 小目标召回偏低，需要更定向采样和低阈值消融。 |
| pole | 4 | 1 | 0.003235 | 有少量召回，需扩电力/杆塔场景样本。 |
| power line | 8 | 0 | 0 | 当前随机图或 prompt 下无召回，需电力专项采样。 |
| insulator | 4 | 0 | 0 | 当前随机图无召回，不能据此否定词表。 |
| animal | 4 | 0 | 0 | 当前随机图无召回。 |
| tree | 4 | 0 | 0 | 当前随机图无召回，需看图像场景。 |

结论：扩展词表 + SAM3 text-only 链路可运行，车辆/道路/建筑类已有非空候选；电力类和小目标需要改为场景定向采样，不能只靠随机 3 张图判断。下一步建议抽取包含杆塔/输电线/巡检场景的图像子集，再跑 power line/pole/insulator 专项 pilot。

