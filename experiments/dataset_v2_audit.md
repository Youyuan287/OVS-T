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


## 目标采样处理策略

针对随机 SAM3-only pilot 中电力类和小目标大量为空的问题，下一步不再随机抽图，而是从旧伪标签 pseudo_lora_b1_step300/masks_union 中挖掘曾经出现 pole/insulator/person/vehicle/car 的图像作为专项 pilot 种子。

已新增 scripts/08_select_targeted_pilot_images.py，当前 dry-run 结果：

- 候选图像：5049 张。
- 选中图像：30 张。
- 选中类别计数：pole=30、ehicle=29、car=27、person=26、insulator=4。
- 输出图片列表：/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_targeted_sampling/targeted_images.txt。

下一步使用该 image list 生成 pole/power line/insulator/person/vehicle/car 专项 prompt proposal，并用更低候选阈值跑 SAM3 text-only pilot。

## 2026-06-05 data3 电力场景 SAM3 + Qwen3-VL Pilot

- 输出目录：`/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_data3_power_pilot_current`
- 场景目录：`data3`
- 场景标注：`power/electric_scene`
- 图像来源：`/home/Groups/group2/Working/TJY/sam3_ir_test/data/ir_images/data3`
- 图像数：`29`，覆盖 data3 全部图像。
- 采样：旧伪标签命中优先，命中图像 `2` 张，随机补齐 `27` 张。
- SAM3 环境：`/home/Groups/group2/Working/seg/miniconda3/envs/esam3_312/bin/python`
- Qwen 环境：`/home/Groups/group2/Working/seg/miniconda3/envs/thgs/bin/python`
- Qwen 模型：`/home/Groups/group2/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct`
- SAM3 权重：`submit_epoch4_best/model/sam3.pt`
- SAM3 设置：`threshold=0.35`，`resolution=768`，`modes=text_only`，`prompts_per_class=4`

| 阶段 | 指标 | 结果 | 结论 |
|---|---:|---:|---|
| Scene sampling | data3 图像数 | 29 | 第一轮电力场景全覆盖。 |
| Scene sampling | 旧伪标签命中图像 | 2 | data3 中旧伪标签可用种子很少。 |
| Prompt proposal | 图像数 | 29 | 全部来自 `data3`。 |
| Prompt proposal | candidate 数 | 261 | 9 类 fallback proposal。 |
| SAM3 candidates | 候选 mask 数 | 1044 | 每条 proposal 约 4 个 text prompt。 |
| SAM3 candidates | 非空 mask 数 | 108 | 非空率约 10.34%。 |
| Qwen QC | task 数 | 1044 | 全量 panel 质检。 |
| Qwen QC | 成功输出 | 1043 | 真实 Qwen 成功，无 rule fallback。 |
| Qwen QC | fallback 数 | 0 | 已使用本地 Qwen3-VL-8B。 |
| Manifest | kept_total | 1 | 当前硬过滤后仅保留 1 条。 |
| Manifest | train_hq / val_hq | 1 / 0 | 小样本拆分已避免单样本落入 val。 |
| Manifest | 电力类样本数 | 0 | `pole/power line/insulator` 暂无高置信入选。 |

### SAM3 非空统计

| 类别 | proposal 数 | candidate 数 | 非空数 | 非空平均面积比例 |
|---|---:|---:|---:|---:|
| building | 29 | 116 | 16 | 0.206927 |
| car | 29 | 116 | 13 | 0.074255 |
| insulator | 29 | 116 | 0 | 0 |
| person | 29 | 116 | 8 | 0.108903 |
| pole | 29 | 116 | 19 | 0.041129 |
| power line | 58 | 232 | 2 | 0.034711 |
| road | 29 | 116 | 41 | 0.274438 |
| vehicle | 29 | 116 | 9 | 0.087040 |

### Qwen3-VL 决策统计

| 类别 | QC 数 | accept | review | drop | 平均 semantic_match |
|---|---:|---:|---:|---:|---:|
| building | 116 | 22 | 7 | 87 | 0.2422 |
| car | 116 | 0 | 0 | 116 | 0.0000 |
| insulator | 116 | 104 | 8 | 4 | 0.9552 |
| person | 116 | 0 | 1 | 115 | 0.0069 |
| pole | 115 | 49 | 29 | 37 | 0.6365 |
| power line | 232 | 64 | 66 | 102 | 0.5241 |
| road | 116 | 0 | 0 | 116 | 0.0000 |
| vehicle | 116 | 0 | 0 | 116 | 0.0000 |

### Overlay 示例

- `power line`: `/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_data3_power_pilot_current/sam3_overlays/power line/20230817162952_inf_2480_2888fb397c61.jpg`
- `pole`: `/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_data3_power_pilot_current/sam3_overlays/pole/20230817163629_inf_2820_185434f56c2c.jpg`
- `insulator`: 无非空 SAM3 mask 示例。

### 结论

- data3 电力场景小数据集链路已经跑通：场景采样 -> prompt proposal -> SAM3 mask -> Qwen3-VL panel QC -> manifest。
- 当前瓶颈主要是 SAM3 text-only 对电力细目标候选不足：`insulator` 全空，`power line` 仅 2 个非空，`pole` 有 19 个非空但未通过最终高置信过滤。
- Qwen3-VL 对电力类别语义并不完全失败，`insulator/pole/power line` 有较多 accept/review；但空 mask 和低最终融合分阻止其进入 `train_hq`。
- 下一步不建议直接训练。建议优先尝试电力类专用 proposal：更强图像定位提示、box/point proposal、降低 SAM3 候选阈值消融，或从 data3 之外扩展更多电力场景样本。
