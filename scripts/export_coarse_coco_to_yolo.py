#!/usr/bin/env python3
"""Export an audited four-class COCO dataset to an Ultralytics YOLO layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_yolo(coco_path: Path, dataset_root: Path, split: str) -> dict:
    payload = json.loads(coco_path.read_text(encoding="utf-8"))
    categories = sorted(payload["categories"], key=lambda row: int(row["id"]))
    category_ids = [int(row["id"]) for row in categories]
    if category_ids != list(range(len(categories))):
        raise ValueError(f"categories must be contiguous from zero, got {category_ids}")
    image_root = dataset_root / "images" / split
    label_root = dataset_root / "labels" / split
    label_root.mkdir(parents=True, exist_ok=True)
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    label_count = 0
    empty_image_count = 0
    class_counts: Counter[int] = Counter()
    image_paths: list[str] = []
    for image in sorted(payload["images"], key=lambda row: int(row["id"])):
        image_path = image_root / image["file_name"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        width = float(image["width"])
        height = float(image["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid image dimensions for {image_path}")
        lines: list[str] = []
        for annotation in sorted(
            annotations_by_image[int(image["id"])], key=lambda row: int(row["id"])
        ):
            category_id = int(annotation["category_id"])
            if category_id not in category_ids:
                raise ValueError(f"unknown category_id {category_id}")
            x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
            center_x = (x + box_width / 2.0) / width
            center_y = (y + box_height / 2.0) / height
            norm_width = box_width / width
            norm_height = box_height / height
            values = (center_x, center_y, norm_width, norm_height)
            if not all(0.0 <= value <= 1.0 for value in values):
                raise ValueError(f"out-of-range normalized bbox for annotation {annotation['id']}")
            lines.append(
                f"{category_id} {center_x:.8f} {center_y:.8f} "
                f"{norm_width:.8f} {norm_height:.8f}"
            )
            label_count += 1
            class_counts[category_id] += 1
        if not lines:
            empty_image_count += 1
        label_path = label_root / f"{Path(image['file_name']).stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        image_paths.append(str(image_path.resolve()))

    list_path = dataset_root / f"{split}.txt"
    list_path.write_text("\n".join(image_paths) + "\n", encoding="utf-8")
    yaml_path = dataset_root / "dataset.yaml"
    existing = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) if yaml_path.exists() else {}
    existing["path"] = str(dataset_root.resolve())
    existing[split] = str(list_path.resolve())
    existing["names"] = {int(row["id"]): row["name"] for row in categories}
    yaml_path.write_text(yaml.safe_dump(existing, sort_keys=False), encoding="utf-8")
    return {
        "protocol": "coarse_coco_to_ultralytics_yolo_v1",
        "split": split,
        "image_count": len(payload["images"]),
        "empty_image_count": empty_image_count,
        "annotation_count": label_count,
        "class_counts": {
            categories[category_id]["name"]: class_counts[category_id]
            for category_id in category_ids
        },
        "coco_sha256": _sha256(coco_path),
        "dataset_yaml": str(yaml_path.resolve()),
        "dataset_yaml_sha256": _sha256(yaml_path),
        "image_list_sha256": _sha256(list_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    audit = export_yolo(args.coco, args.dataset_root, args.split)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
