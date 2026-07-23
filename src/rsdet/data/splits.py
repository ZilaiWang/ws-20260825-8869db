"""训练/验证/交叉验证划分管理。

产出格式遵循 docs/INTEGRATION_CONTRACT.md 第 4 节（contract_v1）。
分组依据见 reports/data/ 下的划分说明。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

SCENE_RE = re.compile(r"(L\d+A\d+|L\d{11})")
LATLON_RE = re.compile(r"(N[\d.]+-E[\d.]+)")
MAR20_RE = re.compile(r"^MAR20_(\d+)$")

DATA_VERSION = "official_raw_v1"


class UnionFind:
    """字符串键的并查集，用于合并同源分组。"""

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


def read_id_list(path: Path) -> list[int]:
    """读取 MAR20 ImageSets 的编号列表。"""
    return [int(token) for token in path.read_text(encoding="utf-8").split() if token.strip()]


def load_official_sides(imagesets: Path) -> tuple[set[int], dict[int, int]]:
    """返回 (官方 train 侧编号集合, 官方 test 侧编号 -> 递增段号)。

    MAR20 论文说明 train/test 按机场划分且两侧机场互斥；test.txt 内部
    呈递增段排列，每段视为一个候选场景分组。该推断非官方逐图标注。
    """
    train_ids = set(read_id_list(imagesets / "train.txt"))
    test_ids = read_id_list(imagesets / "test.txt")
    segment_of: dict[int, int] = {}
    segment = 0
    for index, number in enumerate(test_ids):
        if index > 0 and number <= test_ids[index - 1]:
            segment += 1
        segment_of[number] = segment
    return train_ids, segment_of


def derive_group(
    stem: str, train_side: set[int], segment_of: Mapping[int, int]
) -> tuple[str, str, bool]:
    """由文件主名推导 (group_id, 依据, 是否只能进训练集)。"""
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


def read_class_ids(label_path: Path | None) -> list[int]:
    """读取 YOLO 标签文件中出现的细类 ID（含重复，按框计数）。"""
    if label_path is None:
        return []
    result: list[int] = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        if parts:
            result.append(int(parts[0]))
    return result


def save_split_manifest(
    samples: Iterable[Mapping[str, Any]],
    output_path: Path,
    *,
    version: str,
    data_version: str = DATA_VERSION,
    **extra: Any,
) -> None:
    """按 contract_v1 保存 split manifest。

    Args:
        samples: 每项至少含 image_id、relative_path、group_id，
            并含 split（train/val）或 fold（折号）之一。
        output_path: 输出 JSON 路径。
        version: 划分版本号，如 ``dev_v1``、``cv3_v1``。
        data_version: 数据版本号。
        extra: 写入 manifest 顶层的附加字段（seed、fold_count 等）。
    """
    manifest: dict[str, Any] = {"version": version, "data_version": data_version}
    manifest.update(extra)
    manifest["samples"] = list(samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )