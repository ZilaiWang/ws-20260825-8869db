#!/usr/bin/env python3
"""Convert one frozen CV3 split view and YOLO labels to D-FINE COCO JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError as exc:
        raise ValueError(f"image path has no images component: {image_path}") from exc
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _annotations(
    label_path: Path,
    *,
    image_id: int,
    width: int,
    height: int,
    first_annotation_id: int,
    num_classes: int,
) -> list[dict[str, Any]]:
    output = []
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    for offset, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"invalid YOLO row in {label_path}: {line}")
        category = int(values[0])
        if not 0 <= category < num_classes:
            raise ValueError(f"category {category} outside [0, {num_classes})")
        cx, cy, bw, bh = (float(value) for value in values[1:])
        x = (cx - bw / 2.0) * width
        y = (cy - bh / 2.0) * height
        box_width = bw * width
        box_height = bh * height
        if box_width <= 0.0 or box_height <= 0.0:
            raise ValueError(f"non-positive box in {label_path}: {line}")
        x = min(max(x, 0.0), float(width))
        y = min(max(y, 0.0), float(height))
        box_width = min(box_width, float(width) - x)
        box_height = min(box_height, float(height) - y)
        output.append(
            {
                "id": first_annotation_id + offset,
                "image_id": image_id,
                "category_id": category,
                "bbox": [x, y, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-view", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=25)
    args = parser.parse_args()

    split_view = json.loads(args.split_view.read_text(encoding="utf-8"))
    if int(split_view.get("fold_count", -1)) != 3:
        raise ValueError("expected a formal three-fold split view")
    samples = split_view.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("split view has no samples")
    seen_ids: set[int] = set()
    payloads: dict[str, dict[str, Any]] = {}
    next_annotation_id = 1
    for split in ("train", "val"):
        images: list[dict[str, Any]] = []
        annotations: list[dict[str, Any]] = []
        for sample in samples:
            if sample["split"] != split:
                continue
            image_id = int(sample["image_id"])
            if image_id in seen_ids:
                raise ValueError(f"duplicate image id: {image_id}")
            seen_ids.add(image_id)
            relative = Path(sample["relative_path"])
            image_path = args.data_root / relative
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as image:
                width, height = image.size
            image_annotations = _annotations(
                _label_path(image_path),
                image_id=image_id,
                width=width,
                height=height,
                first_annotation_id=next_annotation_id,
                num_classes=args.num_classes,
            )
            next_annotation_id += len(image_annotations)
            images.append(
                {
                    "id": image_id,
                    "file_name": relative.as_posix(),
                    "width": width,
                    "height": height,
                }
            )
            annotations.extend(image_annotations)
        payloads[split] = {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": category, "name": str(category)}
                for category in range(args.num_classes)
            ],
        }

    expected = int(split_view["train_images"]) + int(split_view["val_images"])
    if len(seen_ids) != expected or len(seen_ids) != len(samples):
        raise RuntimeError("split-view image counts do not reconcile")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, payload in payloads.items():
        path = args.output_dir / f"instances_{split}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        paths[split] = path
    audit = {
        "status": "pass",
        "protocol": "formal_cv3_split_view_yolo_to_dfine_coco_v1",
        "held_out_fold": int(split_view["held_out_fold"]),
        "image_counts": {split: len(payloads[split]["images"]) for split in payloads},
        "annotation_counts": {
            split: len(payloads[split]["annotations"]) for split in payloads
        },
        "cross_split_image_ids": len(
            {item["id"] for item in payloads["train"]["images"]}
            & {item["id"] for item in payloads["val"]["images"]}
        ),
        "input_sha256": {"split_view": _sha256(args.split_view)},
        "output_sha256": {split: _sha256(path) for split, path in paths.items()},
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
