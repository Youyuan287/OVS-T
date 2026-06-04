# 变更与提交记录

本文件用于记录每次代码、脚本、文档或实验日志改动。后续所有提交前必须先更新本文件或对应实验文档，确保 GitHub commit 可以回溯到清晰的中文说明。

## 记录规则

- 每次提交记录：日期、commit、提交信息、改动目的、主要文件、验证情况、后续动作。
- 数据、权重、提交包不进入 Git，只记录服务器路径、大小、生成脚本和对应 commit。
- 线上提交版本还必须同步更新 `experiments/submission_log.md`，并按 `submit-YYYYMMDD-S1` 到 `submit-YYYYMMDD-S4` 打 tag。
- 若只是文档修正，也需要记录原因，避免后续误解技术路线。

## 2026-06-04

### 待提交 - 
ecord-sam3-only-pilot-results

- 目的：记录扩展词表 SAM3-only 小规模 pilot 的实际输出结果。
- 主要内容：
  - 使用 submit_epoch4_best/model/sam3.pt 对 3 张图、18 条 proposal、72 个 text-only 候选进行 mask 生成。
  - 生成 72 个 Qwen 四宫格质检面板。
  - 在 experiments/dataset_v2_audit.md 中记录类别非空数和初步结论。
- 验证：
  - SAM3 pilot 成功完成，28/72 候选非空。
  - Qwen 面板生成成功，任务文件路径：/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_prompt_sam3_qwen8b_pilot_current/qwen_qc_tasks.jsonl。
- 后续：针对电力类和小目标改为场景定向采样。


### 待提交 - handle-empty-sam3-pilot-candidates

- 目的：处理 SAM3 多提示 pilot 中部分 prompt 返回空候选维度的问题。
- 主要改动：
  - 更新 scripts/03_run_sam3_multi_prompt.py，当 SAM3 mask tensor 任一维度为 0 或 squeeze 后为空时，输出空 mask 并继续处理后续 prompt。
- 验证：
  - py_compile 通过。
- 后续：继续重跑 SAM3-only pilot。


### 待提交 - ix-sam3-pilot-mask-shape

- 目的：修复 SAM3-only pilot 中候选 mask 保存失败的问题。
- 主要改动：
  - 更新 scripts/03_run_sam3_multi_prompt.py，对 SAM3 返回的 [1,1,H,W] 或类似维度进行 squeeze。
  - 若 squeeze 后仍不是二维 mask，则显式报错，避免写出错误伪标签。
- 验证：
  - py_compile 通过。
  - --help 通过。
- 后续：重跑 3 张图小规模 SAM3-only pilot。


### 待提交 - support-submission-ckpt-for-sam3-pilot

- 目的：让 Dataset V2 SAM3-only pilot 可以直接使用当前提交包组合权重 submit_epoch4_best/model/sam3.pt。
- 主要改动：
  - 更新 scripts/03_run_sam3_multi_prompt.py，新增 --submission_ckpt 参数。
  - 支持 esam3_model. 和 	iny_text. 前缀的组合权重拆分。
  - 保留原有 --esam3_ckpt + --tiny_ckpt 分离权重模式。
- 验证：
  - py_compile 通过。
  - --help 通过。
- 后续：使用扩展词表先跑小规模 SAM3 mask 生成和 Qwen 面板，不进入训练。


### `7758b5f` - `adjust-dataset-v2-for-no-irgpt-inference`

- 目的：修正 Dataset V2 路线，不再依赖 `WheatCao/ICCV2025-IRGPT` 官方仓库直接推理。
- 主要改动：
  - 新增 `scripts/02_build_prompt_proposals.py`，作为 prompt proposal 主入口。
  - 将 `scripts/02_run_irgpt_proposals.py` 改为兼容 wrapper。
  - 更新 `experiments/dataset_v2_generation_plan.md`，明确 IRGPT 官方缺少稳定推理脚本。
  - 更新 `experiments/dataset_v2_audit.md`，审计重点改为 prompt proposal、SAM3 候选和 Qwen 质检。
- 验证：
  - `py_compile` 通过。
  - `--help` 通过。
  - 小规模 dry-run 成功生成 `prompt_proposals.jsonl`。
- 后续：Dataset V2 第一版主路径改为扩展 prompt bank -> SAM3 多提示候选 mask -> Qwen3-VL-8B 质检 -> 类别自适应 QC。

### `f326e42` - `add-dataset-v2-pseudo-label-pipeline`

- 目的：建立 Dataset V2 伪标签生产流水线骨架。
- 主要改动：
  - 新增红外扩展词表 `data/prompt_bank_ir_v2.json`。
  - 新增 7 个数据流水线脚本，覆盖 prompt/proposal、SAM3 候选、Qwen 面板、Qwen 质检、manifest 融合和存在性校准。
  - 新增 `experiments/dataset_v2_generation_plan.md` 和 `experiments/dataset_v2_audit.md`。
- 验证：
  - 全部脚本 `py_compile` 通过。
  - 全部脚本 `--help` 通过。
  - 极小 dry-run 链路跑通。
- 后续：接入真实 SAM3 权重路径和 Qwen3-VL-8B worker 后跑 200-500 张 pilot。

### `aca2ecf` - `implement-pre-submit-qc-and-inference-controls`

- 目的：为每日 4 次线上提交建立本地格式门禁，并增加低风险推理侧消融开关。
- 主要改动：
  - 新增 `tools/pre_submit_qc.py`，检查 `ann_id`、RLE、mask 尺寸、空 mask 比例和面积分布。
  - 在 `submit_epoch4_best/inference.py` 中增加存在性阈值、候选选择、prompt 归一化、面积保护、孔洞填充和碎片过滤等环境变量开关。
  - 更新中文 `daily_plan` 和 `submission_log`。
- 验证：
  - 推理脚本和 QC 脚本 `py_compile` 通过。
  - QC smoke 测试通过。
  - 大文件检查通过。
- 后续：线上提交前必须先跑 QC，并记录对应 commit/tag/环境变量。

### `febe98a` - `update-chinese-submission-logs`

- 目的：将实验日志和提交纪律改为中文记录，便于队伍协作。
- 主要改动：
  - 更新 `experiments/daily_plan.md`。
  - 更新 `experiments/submission_log.md`。
- 验证：文档更新后已推送。
- 后续：每次线上提交出分后立即补充线上分数和结论。

### `53e1291` - `init-competition-version-management`

- 目的：建立服务器项目 GitHub 版本管理基础。
- 主要改动：
  - 初始化/整理仓库版本管理。
  - 增加 `.gitignore`，避免权重、数据集、压缩包和输出目录进入 Git。
  - 建立比赛文档和实验记录入口。
- 验证：首次推送到 `origin/main` 成功。
- 后续：所有代码和日志改动均需 commit + push。
