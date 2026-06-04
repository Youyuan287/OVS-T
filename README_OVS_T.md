# OVS-T

Competition code and experiment bookkeeping for the Infrared Open-World Full-Modal Segmentation challenge.

This repository tracks the server project at .

## Remote Paths

- Official unlabeled IR data: 
- Current pseudo labels: 
- Server project: 
- GitHub repository: 

## Submission Discipline

The online leaderboard is the only strong validation signal, and the daily quota is limited to 4 submissions. Every uploaded version should have:

- a unique commit;
- a tag named  through ;
- a row in ;
- a pre-upload plan in ;
- a recorded package path and score once available.

Do not commit model weights, checkpoints, datasets, generated masks, visualization images, or submission ZIP files. Keep them on the server and reference paths in logs.

## Recommended Git Workflow



Use explicit  paths; avoid  in this project.
e1_geometry_finetune eval data pyproject.toml README*.md sam3
git diff --cached --stat
git commit -m "submission: YYYYMMDD S1 short description"
git tag -a submit-YYYYMMDD-S1 -m "score=pending; change=<one variable>; package=<path>"
```

????? `git add` ??????? `git add .`?
