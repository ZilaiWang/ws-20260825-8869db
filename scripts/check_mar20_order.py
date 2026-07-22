#!/usr/bin/env python3
"""检验 MAR20 编号是否按机场/机型聚集。

用法:
    PYTHONPATH=src python scripts/check_mar20_order.py --data-root data/raw
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from rsdet.data.xh_dataset import FINE_NAMES, XHDataset

MAR20_RE = re.compile(r"^MAR20_(\d+)$")


def read_class_ids(label_path: Path | None) -> set[int]:
    if label_path is None:
        return set()
    ids = set()
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if parts:
            ids.add(int(parts[0]))
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--show", type=int, default=120, help="打印前 N 张")
    args = parser.parse_args()

    dataset = XHDataset(args.data_root, "train", load_images=False)

    rows: list[tuple[int, set[int]]] = []
    for ref in dataset.refs:
        matched = MAR20_RE.match(ref.stem)
        if matched:
            rows.append((int(matched.group(1)), read_class_ids(ref.label_path)))
    rows.sort()

    print(f"MAR20 图像数: {len(rows)}，编号范围 {rows[0][0]}--{rows[-1][0]}\n")

    # 相邻编号的类别是否相同
    same = sum(1 for i in range(1, len(rows)) if rows[i][1] == rows[i - 1][1])
    print(f"相邻编号类别集合完全相同的比例: {same}/{len(rows) - 1} = {same / (len(rows) - 1):.1%}")
    print("（若明显高于随机，说明编号按场景聚集）\n")

    print(f"=== 前 {args.show} 张的编号与类别 ===")
    for number, classes in rows[: args.show]:
        names = ",".join(FINE_NAMES[c] for c in sorted(classes)) or "(空)"
        print(f"MAR20_{number:<6} {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
