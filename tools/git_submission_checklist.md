# Git Submission Checklist

Use this before every online submission commit.

## Git Safety

 M README_OVS_T.md
 M experiments/daily_plan.md
 M experiments/submission_log.md
 M tools/git_submission_checklist.md
?? "docs/\350\265\233\351\242\230\344\272\214_\346\234\211\351\231\220\346\217\220\344\272\244_IRGPT_SAM3_\346\217\220\345\210\206\350\267\257\347\272\277.md"
origin	git@github.com:Youyuan287/OVS-T.git (fetch)
origin	git@github.com:Youyuan287/OVS-T.git (push)

Expected:

- only lightweight source, config, docs, and experiment logs are staged;
- no model weights, datasets, archives, generated masks, or visualization images are staged;
- remote  points to .

## Commit And Tag

 README_OVS_T.md                                    | 39 +++++++------
 ...217\220\345\210\206\350\267\257\347\272\277.md" | 67 ++++++++++++++++++++++
 experiments/daily_plan.md                          | 32 +++++------
 experiments/submission_log.md                      | 20 +++----
 tools/git_submission_checklist.md                  | 30 ----------
 5 files changed, 114 insertions(+), 74 deletions(-)

After the online score returns, update  and, if useful, add a follow-up commit with the score record.

## Large File Rule

Never stage these:

- , , , 
- , , datasets, generated masks
- visual inspection images or local render folders
- outputs, checkpoints, submission packages
