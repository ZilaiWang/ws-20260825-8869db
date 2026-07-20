#!/usr/bin/env python3
"""按编号顺序检测 MAR20 的连续同类分段。

用法:
    PYTHONPATH=src python scripts/check_mar20_blocks.py --data-root data/raw
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from rsdet.data.xh_dataset import FINE_NAMES, XHDataset

MAR20_RE = re.compile(r"^MAR20_(\d+)$")


def dominant_class(label_path: Path | None) -> int | None:
    """返回该图中框数最多的类别 ID。"""
    if label_path is None:
        return None
    counter: Counter[int] = Counter()
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if parts:
            counter[int(parts[0])] += 1
    return counter.most_common(1)[0][0] if counter else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--max-gap", type=int, default=20,
                        help="编号间隔超过该值则强制断段")
    args = parser.parse_args()

    dataset = XHDataset(args.data_root, "train", load_images=False)

    rows: list[tuple[int, int | None]] = []
    for ref in dataset.refs:
        matched = MAR20_RE.match(ref.stem)
        if matched:
            rows.append((int(matched.group(1)), dominant_class(ref.label_path)))
    rows.sort()

    # 切分连续块：主类别变化或编号跳跃过大时断开
    blocks: list[list[tuple[int, int | None]]] = [[rows[0]]]
    for previous, current in zip(rows, rows[1:]):
        same_class = current[1] == previous[1]
        small_gap = current[0] - previous[0] <= args.max_gap
        if same_class and small_gap:
            blocks[-1].append(current)
        else:
            blocks.append([current])

    sizes = sorted((len(b) for b in blocks), reverse=True)
    print(f"图像 {len(rows)} 张，切分为 {len(blocks)} 个连续块")
    print(f"块大小: 最大 {sizes[0]}，中位 {sizes[len(sizes) // 2]}，最小 {sizes[-1]}")
    print(f"大于 100 张的块: {sum(1 for s in sizes if s > 100)}")
    print(f"仅 1 张的块:     {sum(1 for s in sizes if s == 1)}\n")

    print("=== 最大的 15 个块 ===")
    for block in sorted(blocks, key=len, reverse=True)[:15]:
        class_id = block[0][1]
        name = FINE_NAMES[class_id] if class_id is not None else "(空)"
        print(f"{name:<12} MAR20_{block[0][0]}--{block[-1][0]:<6} {len(block):>4} 张")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())