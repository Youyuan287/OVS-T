import os
import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="training checkpoint, e.g. best_by_iou.pth")
    parser.add_argument("--out", required=True, help="output pure state_dict path, e.g. model/sam3.pt")
    args = parser.parse_args()

    ckpt = torch.load(args.src, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    pure_sd = {}
    total_params = 0

    for k, v in sd.items():
        if torch.is_tensor(v):
            total_params += v.numel()
            if torch.is_floating_point(v):
                pure_sd[k] = v.float().cpu()
            else:
                pure_sd[k] = v.cpu()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(pure_sd, out_path)

    size_mb = os.path.getsize(out_path) / 1024 / 1024

    print(f"params: {total_params:,}")
    print(f"saved to: {out_path}")
    print(f"file size: {size_mb:.2f} MiB")


if __name__ == "__main__":
    main()
