#!/usr/bin/env python3
"""生成 dev_v1 划分 manifest（contract_v1 格式）。

用法:
    PYTHONPATH=src python scripts/build_split.py \
        --data-root data/raw \
        --imagesets /d/BaiduNetdiskDownload/MAR20/ImageSets/ImageSets/Main \
        --near-duplicates reports/data/near_duplicates_mar20.json \
        --output data/splits/dev_v1.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from rsdet.data.xh_dataset import FINE_NAMES, XHDataset, coarse_name

SCENE_RE = re.compile(r"(L\d+A\d+|L\d{11})")
LATLON_RE = re.compile(r"(N[\d.]+-E[\d.]+)")
MAR20_RE = re.compile(r"^MAR20_(\d+)$")

VERSION = "dev_v1"
DATA_VERSION = "official_raw_v1"
SEED = 42
VAL_FRACTION = 0.20


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def read_ids(path: Path) -> list[int]:
    return [int(t) for t in path.read_text(encoding="utf-8").split() if t.strip()]


def read_class_ids(label_path: Path | None) -> list[int]:
    if label_path is None:
        return []
    result = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if parts:
            result.append(int(parts[0]))
    return result


def load_official_sides(imagesets: Path) -> tuple[set[int], dict[int, int]]:
    """返回 (官方 train 侧编号集合, 官方 test 侧编号 -> 递增段号)。"""
    train_ids = set(read_ids(imagesets / "train.txt"))
    test_ids = read_ids(imagesets / "test.txt")
    segment_of: dict[int, int] = {}
    segment = 0
    for index, number in enumerate(test_ids):
        if index > 0 and number <= test_ids[index - 1]:
            segment += 1
        segment_of[number] = segment
    return train_ids, segment_of


def derive_group(stem: str, train_side: set[int], segment_of: dict[int, int]) -> tuple[str, str, bool]:
    """返回 (group_id, 依据, 是否强制进训练集)。"""
    scene = SCENE_RE.search(stem)
    if scene:
        return f"scene_{scene.group(1)}", "ship_scene_id", False
    latlon = LATLON_RE.search(stem)
    if latlon:
        return f"site_{latlon.group(1)}", "vehicle_latlon", False
    mar20 = MAR20_RE.match(stem)
    if mar20:
        number = int(mar20.group(1))
        if number in train_side:
            return f"mar20_trainside_{number}", "mar20_official_train_side", True
        if number in segment_of:
            return f"mar20_seg_{segment_of[number]:03d}", "mar20_testset_segment", False
    raise ValueError(f"无法为 {stem} 推导分组")


def assign_splits(groups: list[dict], val_fraction: float, seed: int) -> dict[str, str]:
    """按组分配 train/val，稀有细类优先保证验证集覆盖。"""
    rng = random.Random(seed)
    total_images = sum(len(g["images"]) for g in groups)
    target_val = round(total_images * val_fraction)
    cap = target_val * 1.15

    assignment: dict[str, str] = {
        g["gid"]: "train" for g in groups if g["force_train"]
    }
    free = [g for g in groups if g["gid"] not in assignment]
    rng.shuffle(free)

    class_total: Counter[int] = Counter()
    for g in groups:
        class_total.update(g["classes"])
    class_target = {c: n * val_fraction for c, n in class_total.items()}

    groups_with: defaultdict[int, list[dict]] = defaultdict(list)
    for g in free:
        for class_id in g["classes"]:
            groups_with[class_id].append(g)

    class_val: Counter[int] = Counter()
    val_images = 0

    # 组数最少的细类先分配，确保稀有类在验证集中有代表
    for class_id in sorted(groups_with, key=lambda c: len(groups_with[c])):
        for g in sorted(groups_with[class_id], key=lambda g: len(g["images"])):
            if g["gid"] in assignment:
                continue
            if class_val[class_id] >= class_target[class_id]:
                break
            if val_images + len(g["images"]) > cap:
                continue
            assignment[g["gid"]] = "val"
            val_images += len(g["images"])
            class_val.update(g["classes"])

    for g in free:
        if g["gid"] in assignment:
            continue
        if val_images < target_val and val_images + len(g["images"]) <= cap:
            assignment[g["gid"]] = "val"
            val_images += len(g["images"])
            class_val.update(g["classes"])
        else:
            assignment[g["gid"]] = "train"
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
        gid, rule, force = derive_group(ref.stem, train_side, segment_of)
        classes = read_class_ids(ref.label_path)
        records.append(
            {
                "image_id": ref.image_id,
                "stem": ref.stem,
                "relative_path": f"images/train/{ref.image_path.name}",
                "raw_group": gid,
                "rule": rule,
                "force_train": force,
                "classes": classes,
                "coarse": coarse_name(classes[0]) if classes else "unknown",
            }
        )

    # 合并 dHash 近重复组
    union = UnionFind()
    for record in records:
        union.add(record["raw_group"])
    merged_edges = 0
    if args.near_duplicates is not None and args.near_duplicates.exists():
        payload = json.loads(args.near_duplicates.read_text(encoding="utf-8"))
        by_stem = {r["stem"]: r for r in records}
        for members in payload.get("duplicate_groups", []):
            present = [by_stem[m] for m in members if m in by_stem]
            for other in present[1:]:
                if union.find(present[0]["raw_group"]) != union.find(other["raw_group"]):
                    merged_edges += 1
                union.union(present[0]["raw_group"], other["raw_group"])
    print(f"近重复合并: {merged_edges} 次")

    for record in records:
        record["group_id"] = union.find(record["raw_group"])

    # 组内任一图强制进 train，则整组强制
    forced_groups = {r["group_id"] for r in records if r["force_train"]}

    groups_by_coarse: defaultdict[str, dict[str, dict]] = defaultdict(dict)
    for record in records:
        bucket = groups_by_coarse[record["coarse"]]
        group = bucket.setdefault(
            record["group_id"],
            {"gid": record["group_id"], "images": [], "classes": Counter(),
             "force_train": record["group_id"] in forced_groups},
        )
        group["images"].append(record["image_id"])
        group["classes"].update(record["classes"])

    assignment: dict[str, str] = {}
    for coarse, bucket in groups_by_coarse.items():
        assignment.update(assign_splits(list(bucket.values()), VAL_FRACTION, SEED))

    samples = [
        {
            "image_id": r["image_id"],
            "relative_path": r["relative_path"],
            "split": assignment[r["group_id"]],
            "group_id": r["group_id"],
            "group_rule": r["rule"],
        }
        for r in sorted(records, key=lambda r: r["image_id"])
    ]

    manifest = {
        "version": VERSION,
        "data_version": DATA_VERSION,
        "seed": SEED,
        "val_fraction": VAL_FRACTION,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- 统计 ----
    by_split = Counter(s["split"] for s in samples)
    print(f"\n总计 {len(samples)} 张：train {by_split['train']}，val {by_split['val']} "
          f"({by_split['val'] / len(samples):.1%})\n")

    split_of = {s["image_id"]: s["split"] for s in samples}
    coarse_images: defaultdict[tuple[str, str], int] = defaultdict(int)
    fine_boxes: defaultdict[tuple[int, str], int] = defaultdict(int)
    fine_images: defaultdict[tuple[int, str], set[int]] = defaultdict(set)
    fine_groups: defaultdict[tuple[int, str], set[str]] = defaultdict(set)
    for r in records:
        split = split_of[r["image_id"]]
        coarse_images[(r["coarse"], split)] += 1
        for class_id in r["classes"]:
            fine_boxes[(class_id, split)] += 1
            fine_images[(class_id, split)].add(r["image_id"])
            fine_groups[(class_id, split)].add(r["group_id"])

    print(f"{'大类':<9} {'train图':>7} {'val图':>6} {'val占比':>7}")
    for coarse in ("ship", "aircraft", "vehicle"):
        n_train = coarse_images[(coarse, "train")]
        n_val = coarse_images[(coarse, "val")]
        total = n_train + n_val
        print(f"{coarse:<9} {n_train:>7} {n_val:>6} {n_val / total:>7.1%}")

    print(f"\n{'ID':>3} {'细类':<12} {'val框':>6} {'val图':>6} {'val组':>6}  提示")
    for class_id in range(len(FINE_NAMES)):
        v_boxes = fine_boxes[(class_id, "val")]
        v_groups = len(fine_groups[(class_id, "val")])
        note = "!! 验证集无样本" if v_boxes == 0 else ("!  证据量低，以 cv3 为准" if v_groups <= 2 else "")
        print(f"{class_id:>3} {FINE_NAMES[class_id]:<12} {v_boxes:>6} "
              f"{len(fine_images[(class_id, 'val')]):>6} {v_groups:>6}  {note}")

    # 泄漏自检
    spans = {
        gid for gid in {r["group_id"] for r in records}
        if len({split_of[r["image_id"]] for r in records if r["group_id"] == gid}) > 1
    }
    print(f"\n跨 train/val 的分组数: {len(spans)}（应为 0）")
    print(f"已写出: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())