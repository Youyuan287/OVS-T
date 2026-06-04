# 每日提交计划

## 日期：2026-06-04

今日线上提交额度：4 次。当前线上已知最好分数：61。

原则：先做低风险推理侧消融，不重新训练；每个版本先跑提交前 QC，commit + push 后再上传。

| 槽位 | 计划改动 | 环境变量/主变量 | 预期效果 | 主要风险 | 上传前门禁 | Commit | Tag | 线上结果 |
|---|---|---|---|---|---|---|---|---|
| S1 | 降低存在性阈值，保持 `top1` | `EXIST_THRESHOLD=0.45` | 验证当前 61 是否主要受漏检影响 | 误检略增 | `tools/pre_submit_qc.py` 通过；空 mask 比例下降但面积不过激 | 待填 | `submit-20260604-S1` | 待填 |
| S2 | 更激进降低存在性阈值，保持 `top1` | `EXIST_THRESHOLD=0.35` | 快速判断漏检是否是主矛盾 | 大量低置信误检 | QC 通过；超大 mask 数不能明显异常 | 待填 | `submit-20260604-S2` | 待填 |
| S3 | 阈值回到 0.45，并打开面积异常保护 | `EXIST_THRESHOLD=0.45 POSTPROCESS_MASKS=1 AREA_FILTER=1 MAX_AREA_RATIO_OBJECT=0.70` | 在提高召回的同时压制大面积误检 | 小目标被误删 | QC 通过；抽样 overlay 确认小目标未被系统性删除 | 待填 | `submit-20260604-S3` | 待填 |
| S4 | 阈值 0.45，并测试训练相似的小 `top-k` 并集 | `EXIST_THRESHOLD=0.45 SELECT_MODE=train_like TOPK=5 MASK_SCORE_THRESHOLD=0.5` | 改善多实例同语义目标召回 | mask 合并造成粘连或误检 | QC 通过；平均面积比例不能明显膨胀 | 待填 | `submit-20260604-S4` | 待填 |

## 上传前门禁命令

使用官方或自定义 smoke tasks：

```bash
python3 tools/pre_submit_qc.py \
  --code_root submit_epoch4_best \
  --test_root /path/to/test_root \
  --env EXIST_THRESHOLD=0.45
```

如果已经生成 `predictions.json`，只做格式与 RLE 检查：

```bash
python3 tools/pre_submit_qc.py \
  --code_root submit_epoch4_best \
  --test_root /path/to/test_root \
  --predictions /path/to/predictions.json \
  --skip_inference
```

## 门禁检查项

- `submit_epoch4_best/model/sam3.pt` 是纯 FP32 权重。
- 提交包不包含优化器、EMA 或训练 wrapper。
- `predictions.json` 中所有 `ann_id` 出现且只出现一次。
- COCO RLE 可解码，mask 尺寸等于原图尺寸。
- 空 mask 比例、平均面积比例、超大 mask 数不出现异常突变。
- 至少抽样查看 50 张 overlay 后再消耗线上提交次数。

## 决策规则

- 若 S1 或 S2 超过 61，下一天围绕对应阈值做小步后处理叠加。
- 若 S1/S2 均低于 61，说明单纯放低存在性阈值不能解决问题，优先分析误检类别和大面积 mask。
- 若 S3 优于 S1，面积保护有效；否则保持后处理关闭。
- 若 S4 优于 S1，后续围绕 `SELECT_MODE=train_like` 和 `TOPK` 做细化；否则保持 `top1`。
