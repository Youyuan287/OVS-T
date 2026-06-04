# Daily Submission Plan

## Date: 2026-06-04

Daily quota: 4 online submissions.

| Slot | Planned change | Expected effect | Risk | Pre-upload checks | Commit | Tag | Result |
|---|---|---|---|---|---|---|---|
| S1 | Baseline/package sanity check if needed | Confirm current 61-point package is reproducible | Low | Format, RLE, ann_id, weight size | TBD | submit-20260604-S1 | TBD |
| S2 | Single inference-threshold change | Recover missed targets if gating is too strict | Medium | Empty-mask ratio and area distribution | TBD | submit-20260604-S2 | TBD |
| S3 | Prompt normalization change only | Improve Chinese/English and long-tail prompt mapping | Medium | Prompt mapping spot check | TBD | submit-20260604-S3 | TBD |
| S4 | Exploratory mask post-processing or IRGPT pseudo-label version | Improve mask quality or semantics | High | Overlay review and small-target audit | TBD | submit-20260604-S4 | TBD |

## Pre-upload Gate

-  is pure FP32 .
- No optimizer, EMA, or training wrapper in the submitted weight.
- Predicted masks decode from COCO RLE.
- All  values are present exactly once.
- Mask size equals source image size.
- Empty-mask ratio and area distribution are not extreme.
- At least 50 overlays are reviewed before spending a submission.
