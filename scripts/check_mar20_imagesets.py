#!/usr/bin/env python3
"""核对 MAR20 官方 train/test 列表，并检验 test.txt 的递增段是否按场景聚集。

用法:
    PYTHONPATH=src python scripts/check_mar20_imagesets.py \
        --data-root data/raw --imagesets /d/BaiduNetdiskDownload/MAR20/ImageSets/Main
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from rsdet.data.xh_dataset import FINE_NAMES, XHDataset

MAR20_RE = re.compile(r"^MAR20_(\d+)$")


def read_ids(path: Path) -> list[int]:
    return [int(line) for line in path.read_text(encoding="utf-8").split() if line.strip()]


def class_ids(label_path: Path | None) -> set[int]:
    if label_path is None:
        return set()
    result = set()
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if parts:
            result.add(int(parts[0]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--imagesets", type=Path, required=True)
    args = parser.parse_args()

    train_ids = read_ids(args.imagesets / "train.txt")
    test_ids = read_ids(args.imagesets / "test.txt")
    print(f"官方 train.txt: {len(train_ids)} 行   test.txt: {len(test_ids)} 行")
    print(f"合计 {len(train_ids) + len(test_ids)}（论文为 1331 + 2511 = 3842）")
    print(f"重复行: train {len(train_ids) - len(set(train_ids))}, "
          f"test {len(test_ids) - len(set(test_ids))}")
    print(f"两侧交集: {len(set(train_ids) & set(test_ids))} （应为 0）\n")

    # 我们手上的 MAR20 编号
    dataset = XHDataset(args.data_root, "train", load_images=False)
    ours: dict[int, set[int]] = {}
    for ref in dataset.refs:
        matched = MAR20_RE.match(ref.stem)
        if matched:
            ours[int(matched.group(1))] = class_ids(ref.label_path)

    in_train = sorted(set(ours) & set(train_ids))
    in_test = sorted(set(ours) & set(test_ids))
    missing = sorted(set(ours) - set(train_ids) - set(test_ids))
    print(f"我们的 {len(ours)} 张 MAR20 图中：")
    print(f"  原属官方 train: {len(in_train)}")
    print(f"  原属官方 test : {len(in_test)}")
    print(f"  两边都找不到  : {len(missing)}  {missing[:10]}\n")

    # 检验 test.txt 的递增段
    runs: list[list[int]] = [[test_ids[0]]]
    for previous, current in zip(test_ids, test_ids[1:]):
        if current > previous:
            runs[-1].append(current)
        else:
            runs.append([current])

    sizes = sorted((len(r) for r in runs), reverse=True)
    print(f"test.txt 切分为 {len(runs)} 个递增段")
    print(f"段大小: 最大 {sizes[0]}, 中位 {sizes[len(sizes) // 2]}, 最小 {sizes[-1]}")
    print(f"仅 1 个元素的段: {sum(1 for s in sizes if s == 1)}\n")

    print("=== 最大的 12 个段：段内出现过的机型 ===")
    for run in sorted(runs, key=len, reverse=True)[:12]:
        seen: Counter[int] = Counter()
        covered = 0
        for number in run:
            if number in ours:
                covered += 1
                for class_id in ours[number]:
                    seen[class_id] += 1
        names = ", ".join(
            f"{FINE_NAMES[c]}×{n}" for c, n in seen.most_common(5)
        ) or "(本段图像不在我们的数据里)"
        print(f"{len(run):>4} 个（我们有 {covered:>3} 张）: {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
