"""把当前 YOLO 训练标注转换为 COCO instances JSON。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

from rsdet.data.xh_dataset import FINE_NAMES, FINE_NAMES_CN, XHDataset, coarse_name, xyxy_to_coco


def _manifest_membership(
    manifest: Path,
    *,
    manifest_split: str,
    held_out_fold: int | None,
) -> dict[str, int]:
    if manifest.suffix.lower() == ".csv":
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            samples: list[Any] = list(csv.DictReader(handle))
    else:
        document = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping) or not isinstance(document.get("samples"), list):
            raise ValueError("manifest 必须包含 samples 列表")
        samples = document["samples"]
    result: dict[str, int] = {}
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise TypeError(f"manifest sample {index} 必须是对象")
        if "fold" in sample:
            if held_out_fold is None:
                raise ValueError("fold manifest 必须提供 --held-out-fold")
            is_validation = int(sample["fold"]) == held_out_fold
            selected = is_validation if manifest_split == "val" else not is_validation
        else:
            selected = str(sample.get("split", "")) == manifest_split
        if not selected:
            continue
        relative = str(sample.get("relative_path", "")).replace("\\", "/")
        path = Path(relative)
        image_id = int(sample.get("image_id", 0))
        if (
            not relative
            or path.is_absolute()
            or ".." in path.parts
            or image_id <= 0
            or relative in result
        ):
            raise ValueError(f"manifest sample {index} 的 path/image_id 非法")
        result[relative] = image_id
    if not result:
        raise ValueError(f"manifest 中没有 split={manifest_split} 的样本")
    return result


def export_coco(
    data_root: Path,
    split: str,
    output: Path,
    *,
    manifest: Path | None = None,
    manifest_path: Path | None = None,
    manifest_split: str = "val",
    held_out_fold: int | None = None,
) -> dict[str, Any]:
    if manifest is not None and manifest_path is not None:
        raise ValueError("provide only one of manifest and manifest_path")
    selected_manifest = manifest if manifest is not None else manifest_path
    dataset = XHDataset(data_root, split=split, load_images=False, require_labels=True)
    membership = (
        None
        if selected_manifest is None
        else _manifest_membership(
            selected_manifest,
            manifest_split=manifest_split,
            held_out_fold=held_out_fold,
        )
    )
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
        image_path = Path(meta["image_path"])
        relative_path = image_path.relative_to(data_root.resolve()).as_posix()
        if membership is not None and relative_path not in membership:
            continue
        image_id = int(target["image_id"]) if membership is None else membership.pop(relative_path)
        coco["images"].append(
            {
                "id": image_id,
                "file_name": relative_path,
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

    if membership:
        raise FileNotFoundError(
            f"manifest 中有 {len(membership)} 张图未在 data root 找到: {list(membership)[:3]}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    return coco


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="data/ 目录")
    parser.add_argument("--split", default="train", help="默认 train")
    parser.add_argument("--manifest", type=Path, default=None, help="可选冻结 JSON/CSV manifest")
    parser.add_argument(
        "--manifest-split",
        choices=("train", "val"),
        default="val",
        help="从 manifest 导出的逻辑划分",
    )
    parser.add_argument("--held-out-fold", type=int, default=None, help="fold manifest 的验证折")
    parser.add_argument("--output", type=Path, required=True, help="输出 JSON")
    args = parser.parse_args()

    coco = export_coco(
        args.data_root.resolve(),
        args.split,
        args.output.resolve(),
        manifest=args.manifest.resolve() if args.manifest else None,
        manifest_split=args.manifest_split,
        held_out_fold=args.held_out_fold,
    )
    print(
        f"COCO 已写入 {args.output}: "
        f"images={len(coco['images'])}, annotations={len(coco['annotations'])}, "
        f"categories={len(coco['categories'])}"
    )


if __name__ == "__main__":
    main()
