# 变更与提交记录

本文件用于记录每次代码、脚本、文档或实验日志改动。后续所有提交前必须先更新本文件或对应实验文档，确保 GitHub commit 可以回溯到清晰的中文说明。

## 记录规则

- 每次提交记录：日期、commit、提交信息、改动目的、主要文件、验证情况、后续动作。
- 数据、权重、提交包不进入 Git，只记录服务器路径、大小、生成脚本和对应 commit。
- 线上提交版本还必须同步更新 `experiments/submission_log.md`，并按 `submit-YYYYMMDD-S1` 到 `submit-YYYYMMDD-S4` 打 tag。
- 若只是文档修正，也需要记录原因，避免后续误解技术路线。

## 2026-06-05

### 待提交 - add-interactive-sam3-qwen-web-console

- 目的：减少每次必须跑完整批处理后再翻 JSONL/目录的诊断成本，提供本地网页交互查看 SAM3 分割和 Qwen3-VL 质检结果。
- 主要改动：
  - 新增 `scripts/sam3_infer_server.py`，在 `esam3_312` 环境常驻加载 SAM3，提供 `GET /health` 与 `POST /segment`。
  - 新增 `scripts/qwen_qc_server.py`，在 `thgs` 环境常驻加载本地 Qwen3-VL-8B，提供 `GET /health` 与 `POST /qc`。
  - 本地新增 `tools/sam3_qwen_web/app.py`，使用 Streamlit 提供上传图片、远端路径、批量路径、prompt 输入、SAM3/Qwen 结果展示和 CSV/JSONL 下载。
  - 本地新增 `tools/sam3_qwen_web/README.md`，记录启动方式、SSH tunnel 和输出目录。
- 验证：
  - 本地 `python -m py_compile tools/sam3_qwen_web/app.py` 通过。
  - 远端 `py_compile` 通过。
  - 远端服务已启动：SAM3 `127.0.0.1:18081`，Qwen `127.0.0.1:18082`。
  - 本地 SSH tunnel health check 通过：`http://127.0.0.1:18081/health` 与 `http://127.0.0.1:18082/health`。
  - 单图 smoke：`data1/00044.jpg` + `road,car,building` 成功返回 3 个 SAM3 candidate，Qwen 对 road candidate 返回完整指标。
  - 批量 smoke：3 张 data1 图、2 个 prompt 共返回 6 个 SAM3 candidate。
- 输出目录：`/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/interactive_web_runs/`。
- 后续：如果交互频率高，可进一步加入历史任务列表、阈值对比和人工 accept/reject 标注。

### 待提交 - record-data1-random10-pilot

- 目的：对比 data3 电力场景表现，随机抽取 data1 城市场景 10 张图复跑 SAM3 + Qwen3-VL 小样本实验。
- 实验设置：
  - 输出目录：`/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_data1_random10_pilot_current`。
  - 场景：`data1`，`scene_type=urban_scene`。
  - 采样：随机 10 张，不使用旧伪标签命中优先。
  - 类别：prompt bank 中 12 类。
  - SAM3：`threshold=0.35`，`resolution=768`，`modes=text_only`，`prompts_per_class=4`。
  - Qwen：本地 `Qwen3-VL-8B-Instruct`，真实质检，0 rule fallback。
- 结果：
  - data1 总图像数 35000，本轮选中 10 张。
  - Prompt proposal：120 条。
  - SAM3 candidates：480 个，145 个非空。
  - Qwen QC：480 个 task，477 条成功输出，0 fallback。
  - Manifest：`kept_total=61`，`train_hq=53`，`val_hq=8`。
  - 保留类别：road=20、building=10、car=10、tree=9、vehicle=8、person=3、pole=1。
- 结论：data1 随机城市场景明显优于 data3 电力场景，说明流水线可用；data3 的核心瓶颈是电力细目标的 SAM3 候选生成不足。

### 待提交 - add-data3-power-qwen-pilot

- 目的：实现并验证 data3 电力场景小数据集 pilot，按一级文件夹名记录场景，并接入真实 Qwen3-VL-8B 质检。
- 主要改动：
  - 新增 `scripts/09_select_scene_pilot_images.py`，支持按 `scene_dir` 选择图像，输出 `image_list.txt`、`scene_manifest.jsonl` 和 `summary.json`。
  - 更新 `scripts/02_build_prompt_proposals.py`，为 fallback proposal 写入 `scene_dir` 和 `default_scene_type`。
  - 更新 `scripts/03_run_sam3_multi_prompt.py`、`04_build_qwen_qc_panels.py`、`05_run_qwen8b_qc.py`、`06_merge_filter_manifest.py`，贯穿 `scene_dir/scene_type` 字段。
  - `scripts/05_run_qwen8b_qc.py` 新增 `--local_qwen_model`，可一次加载本地 Qwen3-VL 模型批量质检。
  - 新增 `scripts/qwen3_vl_qc_worker.py`，保留单 panel Qwen worker 入口。
  - 修复小样本 manifest split：当保留图像少于 2 张时不强制切出 val。
- 验证：
  - `py_compile` 通过。
  - data3 实际图像数为 29，无 data2 目录；本轮覆盖全部 data3 图像。
  - SAM3 正式 pilot：261 条 proposal，1044 个 candidates，108 个非空。
  - Qwen3-VL 正式 QC：1044 个 task，1043 条成功输出，0 条 rule fallback。
  - Manifest：`kept_total=1`，`train_hq=1`，`val_hq=0`；电力类暂未进入高置信 manifest。
- 输出目录：`/home/Groups/group2/Working/TJY/sam3_ir_test/outputs/dataset_v2_data3_power_pilot_current`。
- 后续：不要直接训练；优先改进电力细目标 proposal 或加入 box/point 辅助，再复跑 `pole/power line/insulator`。

## 2026-06-04

### 待提交 - dd-targeted-pilot-image-selection

- 目的：解决随机 SAM3 pilot 对电力类和小目标没有诊断意义的问题。
- 主要改动：
  - 更新 scripts/02_build_prompt_proposals.py，支持 --image_list 和 --include_classes。
  - 新增 scripts/08_select_targeted_pilot_images.py，从旧伪标签命中中筛选专项 pilot 图片。
  - 更新 experiments/dataset_v2_audit.md，记录目标采样策略和首轮统计。
- 验证：
  - py_compile 通过。
  - 目标采样 dry-run 选出 30 张图，候选图像总数 5049。
- 后续：使用目标采样列表跑电力/小目标专项 SAM3 pilot。


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
