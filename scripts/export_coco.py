"""把当前 YOLO 训练标注转换为 COCO instances JSON。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rsdet.data.xh_dataset import FINE_NAMES, FINE_NAMES_CN, XHDataset, coarse_name, xyxy_to_coco


def export_coco(data_root: Path, split: str, output: Path) -> dict[str, Any]:
    dataset = XHDataset(data_root, split=split, load_images=False, require_labels=True)
    coco: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": class_id,
                "name": name,
                "name_cn": FINE_NAMES_CN[class_id],
                "supercategory": coarse_name(class_id),
            }
            for class_id, name in enumerate(FINE_NAMES)
        ],
    }

    annotation_id = 1
    for sample in dataset:
        target = sample["target"]
        meta = sample["meta"]
        image_id = int(target["image_id"])
        image_path = Path(meta["image_path"])
        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.relative_to(data_root.resolve()).as_posix(),
                "width": int(meta["width"]),
                "height": int(meta["height"]),
            }
        )
        for box, class_id, area in zip(
            target["boxes"], target["labels"], target["area"], strict=True
        ):
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id),
                    "bbox": xyxy_to_coco(box),
                    "area": float(area),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    return coco


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="data/ 目录")
    parser.add_argument("--split", default="train", help="默认 train")
    parser.add_argument("--output", type=Path, required=True, help="输出 JSON")
    args = parser.parse_args()

    coco = export_coco(args.data_root.resolve(), args.split, args.output.resolve())
    print(
        f"COCO 已写入 {args.output}: "
        f"images={len(coco['images'])}, annotations={len(coco['annotations'])}, "
        f"categories={len(coco['categories'])}"
    )


if __name__ == "__main__":
    main()
