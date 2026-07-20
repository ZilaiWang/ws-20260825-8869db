#!/usr/bin/env python3
"""用 dHash 找视觉近重复图像，输出连通分组。

用法:
    PYTHONPATH=src python scripts/find_near_duplicates.py --data-root data/raw --threshold 6
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from rsdet.data.xh_dataset import XHDataset


def dhash_bits(path: Path, size: int = 8) -> tuple[np.ndarray, float]:
    """返回 (指纹, 缩略图灰度标准差)。标准差过低说明画面几乎无纹理。"""
    with Image.open(path) as source:
        gray = source.convert("L").resize((size + 1, size), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).astype(np.uint8).ravel()
    return bits, float(pixels.std())


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=6,
                        help="汉明距离阈值，越小越严格")
    parser.add_argument("--prefix", default="MAR20",
                        help="只处理文件名以此开头的图像")
    parser.add_argument("--min-std", type=float, default=0.0,
                        help="缩略图灰度标准差下限，低于此值视为无纹理并排除")
    parser.add_argument("--output", type=Path, default=None,
                        help="把多图分组结果保存为 JSON")
    args = parser.parse_args()

    dataset = XHDataset(args.data_root, "train", load_images=False)
    refs = [r for r in dataset.refs if r.stem.startswith(args.prefix)]
    print(f"待处理图像: {len(refs)} 张，计算指纹中……")

    bits = np.zeros((len(refs), 64), dtype=np.uint8)
    stds = np.zeros(len(refs), dtype=np.float32)
    for index, ref in enumerate(refs):
        bits[index], stds[index] = dhash_bits(ref.image_path)
        if (index + 1) % 500 == 0:
            print(f"  {index + 1}/{len(refs)}")

    usable = stds >= args.min_std
    print(f"纹理过低被排除: {int((~usable).sum())} 张（阈值 std < {args.min_std}）")

    print("比对中……")
    union_find = UnionFind(len(refs))
    edges = 0
    chunk = 256
    for start in range(0, len(refs), chunk):
        block = bits[start : start + chunk]
        distances = (block[:, None, :] != bits[None, :, :]).sum(axis=2)
        for local_row, row in enumerate(distances):
            i = start + local_row
            if not usable[i]:
                continue
            for j in np.nonzero(row <= args.threshold)[0]:
                j = int(j)
                if j > i and usable[j]:
                    union_find.union(i, j)
                    edges += 1

    groups: dict[int, list[str]] = {}
    for index, ref in enumerate(refs):
        groups.setdefault(union_find.find(index), []).append(ref.stem)

    sizes = Counter(len(v) for v in groups.values())
    print(f"\n阈值 {args.threshold}: 命中边 {edges} 条，分组 {len(groups)} 个")
    print(f"含 2 张及以上的组: {sum(1 for v in groups.values() if len(v) > 1)}")
    print(f"最大组: {max(len(v) for v in groups.values())} 张")
    print(f"组大小分布(大小:个数): {dict(sorted(sizes.items()))}\n")

    print("=== 最大的 10 个组 ===")
    for members in sorted(groups.values(), key=len, reverse=True)[:10]:
        if len(members) < 2:
            break
        preview = ", ".join(sorted(members)[:6])
        print(f"{len(members):>4} 张: {preview}{' ...' if len(members) > 6 else ''}")

    if args.output is not None:
        payload = {
            "method": "dhash8_hamming",
            "threshold": args.threshold,
            "min_std": args.min_std,
            "prefix": args.prefix,
            "total_images": len(refs),
            "duplicate_groups": [
                sorted(members)
                for members in sorted(groups.values(), key=len, reverse=True)
                if len(members) > 1
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n已保存: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())