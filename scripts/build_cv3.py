#!/usr/bin/env python3
"""生成 cv3 三折划分 manifest（contract_v1 格式）。

飞机分组以 K=60 机场代理组为原子（来源见 data/groups/），
舰船与发射车沿用文件名景 ID / 经纬度分组。

用法:
    PYTHONPATH=src python scripts/build_cv3.py \
        --data-root data/raw \
        --imagesets third_party/mar20 \
        --airport-groups data/groups/mar20_airport_proxy_k60_for_b.csv \
        --near-duplicates reports/data/near_duplicates_mar20.json \
        --output data/splits/cv3_airport_proxy_k60_v1.json
"""

from __future__ import annotations

import argparse
import csv
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

VERSION = "cv3_airport_proxy_k60_v1"
SEED = 42
FOLD_COUNT = 3
MIN_TRAIN_BOXES = 30


def load_airport_groups(path: Path) -> dict[str, str]:
    """读取 image_name,group_id 两列 CSV，返回 文件主名 -> 分组。"""
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            mapping[Path(row["image_name"].strip()).stem] = row["group_id"].strip()
    return mapping


def assign_folds(groups: list[dict], fold_count: int, seed: int) -> dict[str, int]:
    """把分组分配到各折；-1 表示只做训练、不参与验证。

    按组从大到小依次试放，选使各细类在三折间分布标准差最小的那一折
    （分层分组交叉验证的标准贪心解法）。
    """
    rng = random.Random(seed)
    assignment: dict[str, int] = {g["gid"]: -1 for g in groups if g["train_only"]}
    free = [g for g in groups if g["gid"] not in assignment]
    rng.shuffle(free)
    free.sort(key=lambda g: sum(g["classes"].values()), reverse=True)

    class_total: Counter[int] = Counter()
    for group in free:
        class_total.update(group["classes"])

    fold_classes: list[Counter[int]] = [Counter() for _ in range(fold_count)]
    fold_images = [0] * fold_count

    def spread_after(fold: int, group: dict) -> float:
        """试放入 fold 后，所有细类占比标准差之和。越小越均衡。"""
        total = 0.0
        for class_id, class_count in class_total.items():
            if class_count == 0:
                continue
            ratios = []
            for f in range(fold_count):
                value = fold_classes[f][class_id]
                if f == fold:
                    value += group["classes"].get(class_id, 0)
                ratios.append(value / class_count)
            mean = sum(ratios) / fold_count
            total += (sum((r - mean) ** 2 for r in ratios) / fold_count) ** 0.5
        return total

    for group in free:
        best = min(
            range(fold_count), key=lambda f: (spread_after(f, group), fold_images[f])
        )
        assignment[group["gid"]] = best
        fold_classes[best].update(group["classes"])
        fold_images[best] += len(group["images"])
    return assignment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--imagesets", type=Path, required=True)
    parser.add_argument("--airport-groups", type=Path, default=None,
                        help="MAR20 机场代理分组 CSV（image_name,group_id）")
    parser.add_argument("--near-duplicates", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_side, segment_of = load_official_sides(args.imagesets)
    airport_groups = (
        load_airport_groups(args.airport_groups) if args.airport_groups else {}
    )
    if airport_groups:
        print(f"机场代理分组: {len(set(airport_groups.values()))} 组，"
              f"覆盖 {len(airport_groups)} 张")

    dataset = XHDataset(args.data_root, "train", load_images=False)

    records = []
    for ref in dataset.refs:
        gid, rule, train_only = derive_group(ref.stem, train_side, segment_of)
        if ref.stem in airport_groups:
            gid = airport_groups[ref.stem]
            rule = "mar20_airport_proxy_k60"
            train_only = False
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

    fine_fold: defaultdict[tuple[int, int], int] = defaultdict(int)
    fine_groups: defaultdict[tuple[int, int], set[str]] = defaultdict(set)
    for record in records:
        fold = fold_of[record["image_id"]]
        for class_id in record["classes"]:
            fine_fold[(class_id, fold)] += 1
            fine_groups[(class_id, fold)].add(record["group_id"])

    print(f"\n各折作为验证集时的框数（训练框 = 总框 - 该折框）")
    print(f"{'ID':>3} {'细类':<12}"
          + "".join(f"{'f' + str(f):>8}" for f in range(FOLD_COUNT))
          + f"{'总框':>8}  提示")
    problems: list[str] = []
    skew_notes: list[str] = []
    for class_id in range(len(FINE_NAMES)):
        per_fold = [fine_fold[(class_id, f)] for f in range(FOLD_COUNT)]
        total = sum(per_fold) + fine_fold[(class_id, -1)]
        row = "".join(f"{n:>8}" for n in per_fold)
        empty = [f for f in range(FOLD_COUNT) if per_fold[f] == 0]
        starved = [
            f for f in range(FOLD_COUNT)
            if total and total - per_fold[f] < max(10, total * 0.10)
        ]
        skewed = [
            f for f in range(FOLD_COUNT) if total and per_fold[f] > total * 0.60
        ]
        thin = [f for f in range(FOLD_COUNT) if 0 < len(fine_groups[(class_id, f)]) <= 1]
        if empty:
            note = f"!! fold {empty} 验证无样本"
            problems.append(f"{FINE_NAMES[class_id]}: {note}")
        elif starved:
            note = f"!! fold {starved} 作验证时训练框不足总量 10%"
            problems.append(f"{FINE_NAMES[class_id]}: {note}")
        elif skewed:
            note = f"!  fold {skewed} 占比 >60%"
            skew_notes.append(f"{FINE_NAMES[class_id]}: 各折 {per_fold}，折间波动大")
        elif thin:
            note = f"!  fold {thin} 仅 1 组"
        else:
            note = ""
        print(f"{class_id:>3} {FINE_NAMES[class_id]:<12}{row}{total:>8}  {note}")

    group_folds: defaultdict[str, set[int]] = defaultdict(set)
    for record in records:
        group_folds[record["group_id"]].add(fold_of[record["image_id"]])
    spans = sum(1 for folds in group_folds.values() if len(folds) > 1)
    print(f"\n跨折的分组数: {spans}（应为 0）")
    if problems:
        print(f"\n需要处理的问题 {len(problems)} 条:")
        for line in problems:
            print(f"  - {line}")
    if skew_notes:
        print(f"\n折间波动较大（可用，但解读时需注明）{len(skew_notes)} 条:")
        for line in skew_notes:
            print(f"  - {line}")
    print(f"\n已写出: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())