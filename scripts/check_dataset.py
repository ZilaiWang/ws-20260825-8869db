"""快速检查 XH-202625 数据是否能被统一加载器完整读取。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from rsdet.data.xh_dataset import FINE_NAMES, XHDataset

OFFICIAL_TRAIN_IMAGES = 4481
OFFICIAL_TRAIN_BOXES = 20933
OFFICIAL_TRAIN_CLASS_COUNTS = np.asarray(
    [
        17,
        30,
        641,
        1994,
        1317,
        1297,
        998,
        500,
        1017,
        361,
        547,
        750,
        895,
        762,
        432,
        583,
        1265,
        1424,
        493,
        2147,
        1114,
        262,
        933,
        752,
        402,
    ],
    dtype=np.int64,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="data/ 目录")
    parser.add_argument("--split", default="train", help="默认 train")
    parser.add_argument(
        "--official-train",
        action="store_true",
        help="同时核对当前官方训练集的固定数量",
    )
    args = parser.parse_args()

    dataset = XHDataset(args.data_root, split=args.split, load_images=True)
    class_counts = np.zeros(len(FINE_NAMES), dtype=np.int64)
    format_counts: Counter[str] = Counter()
    box_count = 0

    for sample in dataset:
        image = sample["image"]
        target = sample["target"]
        meta = sample["meta"]

        assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
        assert tuple(image.shape[:2]) == (meta["height"], meta["width"])
        assert target["boxes"].shape == (len(target["labels"]), 4)
        assert target["boxes"].dtype == np.float32
        assert target["labels"].dtype == np.int64
        assert np.isfinite(target["boxes"]).all()
        assert (target["boxes"][:, 2] > target["boxes"][:, 0]).all()
        assert (target["boxes"][:, 3] > target["boxes"][:, 1]).all()

        labels = target["labels"]
        class_counts += np.bincount(labels, minlength=len(FINE_NAMES))
        box_count += len(labels)
        format_counts[str(meta["file_format"])] += 1

    if args.official_train:
        if args.split != "train":
            raise ValueError("--official-train 只能与 --split train 一起使用")
        assert len(dataset) == OFFICIAL_TRAIN_IMAGES
        assert box_count == OFFICIAL_TRAIN_BOXES
        np.testing.assert_array_equal(class_counts, OFFICIAL_TRAIN_CLASS_COUNTS)

    print("数据加载检查通过")
    print(f"split={args.split}, images={len(dataset)}, boxes={box_count}, classes=25")
    print(
        "decoded_formats="
        + ", ".join(f"{key}:{value}" for key, value in sorted(format_counts.items()))
    )
    print("class_counts=" + ", ".join(f"{i}:{count}" for i, count in enumerate(class_counts)))


if __name__ == "__main__":
    main()
