import json
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter


CANONICAL_PROMPTS = [
    "person",
    "car",
    "vehicle",
    "truck",
    "road",
    "tree",
    "building",
    "pole",
    "power line",
    "wire",
    "insulator",
    "transformer",
    "animal",
]


ALIAS_MAP = {
    "human": "person",
    "people": "person",
    "pedestrian": "person",
    "man": "person",
    "woman": "person",

    "automobile": "car",
    "sedan": "car",
    "suv": "car",

    "van": "vehicle",
    "bus": "vehicle",

    "street": "road",
    "urban road": "road",
    "lane": "road",

    "vegetation": "tree",
    "plant": "tree",
    "forest": "tree",

    "utility pole": "pole",
    "electric pole": "pole",
    "power pole": "pole",
    "telegraph pole": "pole",
    "transmission tower": "pole",
    "power tower": "pole",
    "tower": "pole",

    "power cable": "power line",
    "wire": "power line",
    "cable": "power line",

    "electrical insulator": "insulator",
    "power insulator": "insulator",

    "bear": "animal",
    "dog": "animal",
}


CONFLICT_GROUPS = [
    {"car", "automobile", "sedan", "suv", "vehicle", "truck", "bus", "van"},
    {"person", "human", "people", "pedestrian", "man", "woman"},
    {"pole", "utility pole", "electric pole", "power pole", "telegraph pole", "transmission tower", "power tower", "tower"},
    {"power line", "wire", "cable", "power cable"},
    {"tree", "vegetation", "plant", "forest"},
    {"road", "street", "urban road", "lane"},
    {"animal", "bear", "dog"},
]


def norm_prompt(p):
    p = str(p).strip().lower()
    p = p.replace("_", " ").replace("-", " ")
    p = " ".join(p.split())

    if p in ALIAS_MAP:
        return ALIAS_MAP[p]

    for key, value in ALIAS_MAP.items():
        if key in p:
            return value

    return p


def get_conflict_set(pos_set):
    out = set(pos_set)

    for group in CONFLICT_GROUPS:
        normalized_group = {norm_prompt(x) for x in group}
        if out & normalized_group:
            out |= normalized_group

    return out


def pick(row, keys, default=None):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_jsonl", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--neg_per_image", type=int, default=2)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()

    random.seed(args.seed)

    rows = load_jsonl(args.in_jsonl)

    groups = defaultdict(list)

    for r in rows:
        image = pick(r, ["image", "image_path", "img", "img_path"])
        mask = pick(r, ["mask", "mask_path", "pseudo_mask", "mask_file"])
        prompt = pick(r, ["prompt", "text_prompt", "text", "category", "class_name"])

        if image is None or prompt is None:
            continue

        prompt = norm_prompt(prompt)

        groups[image].append({
            "image": image,
            "prompt": prompt,
            "mask": mask if mask is not None else "",
            "exists": 1,
            "source": "pos",
        })

    out_rows = []
    num_pos = 0
    num_neg = 0

    for image, items in groups.items():
        pos_prompts = {x["prompt"] for x in items}
        avoid = get_conflict_set(pos_prompts)

        # 保留正样本
        for x in items:
            if x["mask"]:
                out_rows.append(x)
                num_pos += 1

        # 生成负样本：从当前图没有出现过、且不冲突的类别中采样
        candidates = []
        for p in CANONICAL_PROMPTS:
            p_norm = norm_prompt(p)
            if p_norm not in avoid:
                candidates.append(p_norm)

        # 去重
        candidates = list(dict.fromkeys(candidates))
        random.shuffle(candidates)

        for neg_prompt in candidates[:args.neg_per_image]:
            out_rows.append({
                "image": image,
                "prompt": neg_prompt,
                "mask": "",
                "exists": 0,
                "source": "neg",
            })
            num_neg += 1

    random.shuffle(out_rows)

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counter = Counter()
    prompt_counter = Counter()

    for r in out_rows:
        counter[r["exists"]] += 1
        prompt_counter[(r["exists"], r["prompt"])] += 1

    print("[Done]")
    print("images:", len(groups))
    print("positive samples:", num_pos)
    print("negative samples:", num_neg)
    print("total samples:", len(out_rows))
    print("exists counter:", dict(counter))
    print("top prompts:", prompt_counter.most_common(20))
    print("out:", args.out_jsonl)


if __name__ == "__main__":
    main()
