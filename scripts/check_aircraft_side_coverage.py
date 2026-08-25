#!/usr/bin/env python3
"""统计 20 种机型在官方 train/test 两侧的分布，评估验证集覆盖风险。

用法:
    PYTHONPATH=src python scripts/check_aircraft_side_coverage.py \
        --data-root data/raw --imagesets /d/BaiduNetdiskDownload/MAR20/ImageSets/ImageSets/Main
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from rsdet.data.xh_dataset import FINE_NAMES, XHDataset

MAR20_RE = re.compile(r"^MAR20_(\d+)$")


def read_ids(path: Path) -> list[int]:
    return [int(t) for t in path.read_text(encoding="utf-8").split() if t.strip()]


def class_ids(label_path: Path | None) -> list[int]:
    if label_path is None:
        return []
    result = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if parts:
            result.append(int(parts[0]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--imagesets", type=Path, required=True)
    args = parser.parse_args()

    train_ids = set(read_ids(args.imagesets / "train.txt"))
    test_ids = read_ids(args.imagesets / "test.txt")
    test_set = set(test_ids)

    # test.txt 的递增段 → 段号
    segment_of: dict[int, int] = {}
    segment = 0
    for index, number in enumerate(test_ids):
        if index > 0 and number <= test_ids[index - 1]:
            segment += 1
        segment_of[number] = segment

    dataset = XHDataset(args.data_root, "train", load_images=False)

    boxes_train: Counter[int] = Counter()
    boxes_test: Counter[int] = Counter()
    images_train: defaultdict[int, set[int]] = defaultdict(set)
    images_test: defaultdict[int, set[int]] = defaultdict(set)
    segments_of_class: defaultdict[int, set[int]] = defaultdict(set)

    for ref in dataset.refs:
        matched = MAR20_RE.match(ref.stem)
        if not matched:
            continue
        number = int(matched.group(1))
        ids = class_ids(ref.label_path)
        if number in train_ids:
            for class_id in ids:
                boxes_train[class_id] += 1
                images_train[class_id].add(number)
        elif number in test_set:
            for class_id in ids:
                boxes_test[class_id] += 1
                images_test[class_id].add(number)
                segments_of_class[class_id].add(segment_of[number])

    print("侧 = 官方 MAR20 划分；val 只能从 test 侧抽，故 test 侧证据量决定该类可评估性\n")
    header = (
        f"{'ID':>3} {'机型':<12} {'train图':>7} {'test图':>7} {'test框':>7} {'test段':>7}  风险"
    )
    print(header)
    print("-" * len(header))
    for class_id in range(4, 24):
        n_train = len(images_train[class_id])
        n_test = len(images_test[class_id])
        n_seg = len(segments_of_class[class_id])
        if n_seg == 0:
            risk = "!! test 侧无样本，无法评估"
        elif n_seg <= 2:
            risk = "!  段数过少，val 可能取不到"
        elif n_test < 30:
            risk = "?  样本偏少"
        else:
            risk = ""
        print(
            f"{class_id:>3} {FINE_NAMES[class_id]:<12} {n_train:>7} {n_test:>7} "
            f"{boxes_test[class_id]:>7} {n_seg:>7}  {risk}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
