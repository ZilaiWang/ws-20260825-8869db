#!/usr/bin/env python3
"""Build native-source COCO and exact source-to-pseudo projection metadata."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

CELL_SIZE = 1000
GRID = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _basename(path_text: str) -> str:
    return Path(path_text).name


def build(
    full: dict[str, Any], pseudo: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    full_by_name: dict[str, dict[str, Any]] = {}
    for row in full["images"]:
        name = _basename(str(row["file_name"]))
        if name in full_by_name:
            raise ValueError(f"duplicate basename in full COCO: {name}")
        full_by_name[name] = row

    occurrences: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for pseudo_image in pseudo["images"]:
        sources = pseudo_image.get("source_images")
        if not isinstance(sources, list) or len(sources) != GRID * GRID:
            raise ValueError("pseudo image must contain exactly 100 source_images")
        for cell_index, source_text in enumerate(sources):
            name = _basename(str(source_text))
            if name not in full_by_name:
                raise ValueError(f"pseudo source is absent from full COCO: {name}")
            source = full_by_name[name]
            source_id = int(source["id"])
            width = int(source["width"])
            height = int(source["height"])
            scale = min(CELL_SIZE / width, CELL_SIZE / height)
            resized_width = max(1, round(width * scale))
            resized_height = max(1, round(height * scale))
            row, column = divmod(cell_index, GRID)
            x_offset = column * CELL_SIZE + (CELL_SIZE - resized_width) // 2
            y_offset = row * CELL_SIZE + (CELL_SIZE - resized_height) // 2
            occurrences.append(
                {
                    "pseudo_image_id": int(pseudo_image["id"]),
                    "cell_index": cell_index,
                    "source_image_id": source_id,
                    "source_file_name": str(source["file_name"]),
                    "source_width": width,
                    "source_height": height,
                    "scale": scale,
                    "resized_width": resized_width,
                    "resized_height": resized_height,
                    "x_offset": x_offset,
                    "y_offset": y_offset,
                }
            )
            selected_ids.add(source_id)

    source_images = [dict(row) for row in full["images"] if int(row["id"]) in selected_ids]
    source_annotations = [
        dict(row) for row in full["annotations"] if int(row["image_id"]) in selected_ids
    ]
    pseudo_counts = collections.Counter(
        int(row["category_id"]) for row in pseudo["annotations"]
    )
    source_counts = collections.Counter(
        int(row["category_id"]) for row in source_annotations
    )
    if pseudo_counts != source_counts:
        raise ValueError(
            "pseudo and selected source fine-class counts differ; projection would not be paired: "
            f"pseudo={dict(pseudo_counts)} source={dict(source_counts)}"
        )
    source_document = {
        "images": source_images,
        "annotations": source_annotations,
        "categories": full["categories"],
    }
    mapping = {
        "status": "complete",
        "role": "native_source_to_pseudo10k_exact_projection_map",
        "source_images": len(source_images),
        "pseudo_images": len(pseudo["images"]),
        "annotations": len(source_annotations),
        "occurrences": occurrences,
    }
    return source_document, mapping


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-ground-truth", type=Path, required=True)
    parser.add_argument("--pseudo-ground-truth", type=Path, required=True)
    parser.add_argument("--output-source-ground-truth", type=Path, required=True)
    parser.add_argument("--output-mapping", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_source_ground_truth, args.output_mapping):
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
    source, mapping = build(
        json.loads(args.full_ground_truth.read_text(encoding="utf-8")),
        json.loads(args.pseudo_ground_truth.read_text(encoding="utf-8")),
    )
    args.output_source_ground_truth.write_text(
        json.dumps(source, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    mapping["input_sha256"] = {
        "full_ground_truth": _sha256(args.full_ground_truth),
        "pseudo_ground_truth": _sha256(args.pseudo_ground_truth),
    }
    mapping["source_ground_truth_sha256"] = _sha256(args.output_source_ground_truth)
    args.output_mapping.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in mapping.items() if key != "occurrences"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
