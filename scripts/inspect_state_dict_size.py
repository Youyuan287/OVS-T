import torch
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
args = parser.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu")
sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

groups = defaultdict(int)
total = 0

for k, v in sd.items():
    if not torch.is_tensor(v):
        continue
    n = v.numel()
    total += n
    prefix = k.split(".")[0]
    if prefix == "backbone" and len(k.split(".")) > 1:
        prefix = "backbone." + k.split(".")[1]
    groups[prefix] += n

print(f"total params: {total:,}")
print(f"fp32 size approx: {total * 4 / 1024 / 1024:.2f} MiB")
print("\nTop groups:")
for name, n in sorted(groups.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:40s} {n:15,} params  {n*4/1024/1024:8.2f} MiB")
