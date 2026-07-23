#!/usr/bin/env python3
"""生成 cv3 三折划分 manifest（contract_v1 格式）。

用法:
    PYTHONPATH=src python scripts/build_cv3.py \
        --data-root data/raw \
        --imagesets third_party/mar20 \
        --near-duplicates reports/data/near_duplicates_mar20.json \
        --output data/splits/cv3_v1.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from rsdet.data.splits import (
    UnionFind,
    derive_group,
    load_official_sides,
    read_class_ids,
    save_split_manifest,
)
from rsdet.data.xh_dataset import FINE_NAMES, XHDataset, coarse_name

VERSION = "cv3_v1"
SEED = 42
FOLD_COUNT = 3


def assign_folds(groups: list[dict], fold_count: int, seed: int) -> dict[str, int]:
    """把分组分配到各折；-1 表示只做训练、不参与验证。

    稀有细类优先轮转分配，保证每折都有代表。
    """
    rng = random.Random(seed)
    assignment: dict[str, int] = {g["gid"]: -1 for g in groups if g["train_only"]}
    free = [g for g in groups if g["gid"] not in assignment]
    rng.shuffle(free)

    groups_with: defaultdict[int, list[dict]] = defaultdict(list)
    for group in free:
        for class_id in group["classes"]:
            groups_with[class_id].append(group)

    fold_images = [0] * fold_count

    # 组数最少的细类先分配，依次轮转到各折
    for class_id in sorted(groups_with, key=lambda c: len(groups_with[c])):
        cursor = 0
        for group in sorted(groups_with[class_id], key=lambda g: len(g["images"])):
            if group["gid"] in assignment:
                continue
            fold = cursor % fold_count
            assignment[group["gid"]] = fold
            fold_images[fold] += len(group["images"])
            cursor += 1

    # 其余分组填给当前最小的折
    for group in free:
        if group["gid"] in assignment:
            continue
        fold = min(range(fold_count), key=lambda f: fold_images[f])
        assignment[group["gid"]] = fold
        fold_images[fold] += len(group["images"])
    return assignment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--imagesets", type=Path, required=True)
    parser.add_argument("--near-duplicates", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_side, segment_of = load_official_sides(args.imagesets)
    dataset = XHDataset(args.data_root, "train", load_images=False)

    records = []
    for ref in dataset.refs:
        gid, rule, train_only = derive_group(ref.stem, train_side, segment_of)
        classes = read_class_ids(ref.label_path)
        records.append(
            {
                "image_id": ref.image_id,
                "stem": ref.stem,
                "relative_path": f"images/train/{ref.image_path.name}",
                "raw_group": gid,
                "rule": rule,
                "train_only": train_only,
                "classes": classes,
                "coarse": coarse_name(classes[0]) if classes else "unknown",
            }
        )

    union = UnionFind()
    for record in records:
        union.add(record["raw_group"])
    merged = 0
    if args.near_duplicates is not None and args.near_duplicates.exists():
        payload = json.loads(args.near_duplicates.read_text(encoding="utf-8"))
        by_stem = {r["stem"]: r for r in records}
        for members in payload.get("duplicate_groups", []):
            present = [by_stem[m] for m in members if m in by_stem]
            for other in present[1:]:
                if union.find(present[0]["raw_group"]) != union.find(other["raw_group"]):
                    merged += 1
                union.union(present[0]["raw_group"], other["raw_group"])
    print(f"近重复合并: {merged} 次")

    for record in records:
        record["group_id"] = union.find(record["raw_group"])

    train_only_groups = {r["group_id"] for r in records if r["train_only"]}

    by_coarse: defaultdict[str, dict[str, dict]] = defaultdict(dict)
    for record in records:
        bucket = by_coarse[record["coarse"]]
        group = bucket.setdefault(
            record["group_id"],
            {
                "gid": record["group_id"],
                "images": [],
                "classes": Counter(),
                "train_only": record["group_id"] in train_only_groups,
            },
        )
        group["images"].append(record["image_id"])
        group["classes"].update(record["classes"])

    assignment: dict[str, int] = {}
    for bucket in by_coarse.values():
        assignment.update(assign_folds(list(bucket.values()), FOLD_COUNT, SEED))

    samples = [
        {
            "image_id": r["image_id"],
            "relative_path": r["relative_path"],
            "fold": assignment[r["group_id"]],
            "group_id": r["group_id"],
            "group_rule": r["rule"],
        }
        for r in sorted(records, key=lambda r: r["image_id"])
    ]

    save_split_manifest(
        samples,
        args.output,
        version=VERSION,
        seed=SEED,
        fold_count=FOLD_COUNT,
        train_only_fold=-1,
        note="fold=-1 表示该图仅用于训练，不参与任何一折的验证",
    )

    # ---- 统计 ----
    fold_of = {s["image_id"]: s["fold"] for s in samples}
    counts = Counter(fold_of.values())
    print(f"\n总计 {len(samples)} 张")
    print(f"仅训练(fold=-1): {counts[-1]}")
    for fold in range(FOLD_COUNT):
        print(f"fold {fold}: {counts[fold]} 张")

    print(f"\n{'大类':<9}" + "".join(f"{'fold' + str(f):>8}" for f in range(FOLD_COUNT))
          + f"{'仅训练':>8}")
    coarse_fold: defaultdict[tuple[str, int], int] = defaultdict(int)
    for record in records:
        coarse_fold[(record["coarse"], fold_of[record["image_id"]])] += 1
    for coarse in ("ship", "aircraft", "vehicle"):
        row = "".join(f"{coarse_fold[(coarse, f)]:>8}" for f in range(FOLD_COUNT))
        print(f"{coarse:<9}{row}{coarse_fold[(coarse, -1)]:>8}")

    print(f"\n{'ID':>3} {'细类':<12}" + "".join(f"{'f' + str(f) + '框':>8}" for f in range(FOLD_COUNT)) + "  提示")
    fine_fold: defaultdict[tuple[int, int], int] = defaultdict(int)
    fine_groups: defaultdict[tuple[int, int], set[str]] = defaultdict(set)
    for record in records:
        fold = fold_of[record["image_id"]]
        for class_id in record["classes"]:
            fine_fold[(class_id, fold)] += 1
            fine_groups[(class_id, fold)].add(record["group_id"])
    for class_id in range(len(FINE_NAMES)):
        row = "".join(f"{fine_fold[(class_id, f)]:>8}" for f in range(FOLD_COUNT))
        empty = [f for f in range(FOLD_COUNT) if fine_fold[(class_id, f)] == 0]
        thin = [f for f in range(FOLD_COUNT) if 0 < len(fine_groups[(class_id, f)]) <= 2]
        note = f"!! fold {empty} 无样本" if empty else (f"!  fold {thin} 组数≤2" if thin else "")
        print(f"{class_id:>3} {FINE_NAMES[class_id]:<12}{row}  {note}")

    spans = 0
    group_folds: defaultdict[str, set[int]] = defaultdict(set)
    for record in records:
        group_folds[record["group_id"]].add(fold_of[record["image_id"]])
    spans = sum(1 for folds in group_folds.values() if len(folds) > 1)
    print(f"\n跨折的分组数: {spans}（应为 0）")
    print(f"已写出: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())