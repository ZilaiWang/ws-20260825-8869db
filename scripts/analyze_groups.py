#!/usr/bin/env python3
"""历史诊断脚本：按文件名规律输出初期分组统计。

MAR20 在这里仍是“一图一组”的未解决占位逻辑，不能生成当前开发或
正式划分。当前 K60 分组与 CV3 入口见 scripts/build_cv3.py。

用法:
    PYTHONPATH=src python scripts/analyze_groups.py --data-root data/raw
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from rsdet.data.xh_dataset import FINE_NAMES, XHDataset, coarse_name

SCENE_RE = re.compile(r"(L\d+A\d+|L\d{11})")
LATLON_RE = re.compile(r"(N[\d.]+-E[\d.]+)")
MAR20_RE = re.compile(r"^MAR20_(\d+)$")


def derive_group_id(stem: str) -> tuple[str, str]:
    """从文件主名推导 (group_id, 依据)。"""
    scene = SCENE_RE.search(stem)
    if scene:
        return f"scene_{scene.group(1)}", "ship_scene"
    latlon = LATLON_RE.search(stem)
    if latlon:
        return f"site_{latlon.group(1)}", "vehicle_site"
    mar20 = MAR20_RE.match(stem)
    if mar20:
        # 占位：MAR20 文件名不含场景信息，暂时每张图自成一组（无防泄漏效果）
        return f"mar20_{mar20.group(1)}", "aircraft_UNRESOLVED"
    return f"unknown_{stem}", "unknown"


def read_class_ids(label_path: Path | None) -> list[int]:
    if label_path is None:
        return []
    ids = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if parts:
            ids.append(int(parts[0]))
    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    dataset = XHDataset(args.data_root, "train", load_images=False)
    print(f"图像总数: {len(dataset)}")

    rule_counter: Counter[str] = Counter()
    group_images: defaultdict[str, list[str]] = defaultdict(list)
    class_boxes: Counter[int] = Counter()
    class_images: defaultdict[int, set[str]] = defaultdict(set)
    class_groups: defaultdict[int, set[str]] = defaultdict(set)

    for ref in dataset.refs:
        group_id, rule = derive_group_id(ref.stem)
        rule_counter[rule] += 1
        group_images[group_id].append(ref.stem)

        for class_id in read_class_ids(ref.label_path):
            class_boxes[class_id] += 1
            class_images[class_id].add(ref.stem)
            class_groups[class_id].add(group_id)

    print("\n=== 分组依据 ===")
    for rule, count in rule_counter.most_common():
        groups = {g for g, s in group_images.items() if derive_group_id(s[0])[1] == rule}
        print(f"{rule:24s} 图像 {count:5d}  分组 {len(groups):5d}")

    print(f"\n分组总数: {len(group_images)}")

    print("\n=== 最大的 10 个分组 ===")
    for group_id, stems in sorted(group_images.items(), key=lambda kv: len(kv[1]), reverse=True)[
        :10
    ]:
        print(f"{group_id:32s} {len(stems):4d} 张")

    print("\n=== 每个细类的证据量 ===")
    print(f"{'ID':>3} {'类别':<12} {'大类':<9} {'框数':>6} {'图数':>6} {'分组数':>7}")
    for class_id in range(len(FINE_NAMES)):
        print(
            f"{class_id:>3} {FINE_NAMES[class_id]:<12} {coarse_name(class_id):<9} "
            f"{class_boxes[class_id]:>6} {len(class_images[class_id]):>6} "
            f"{len(class_groups[class_id]):>7}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
