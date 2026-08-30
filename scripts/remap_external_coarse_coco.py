#!/usr/bin/env python3
"""Remap a COCO-style external remote-sensing dataset to coarse pretraining labels.

The output detector has four foreground classes:
0 aircraft, 1 ship, 2 vehicle, 3 other_remote_object.  Background remains
implicit, as in standard object detection.  This script does not map civil
aircraft/ships to the competition's 25 fine classes.

The mapping JSON is keyed by original category name and has values among:
"aircraft", "ship", "vehicle", "other_remote_object", or null (drop).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TARGETS = {
    "aircraft": 0,
    "ship": 1,
    "vehicle": 2,
    "other_remote_object": 3,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--license-note", default="")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    categories = {int(row["id"]): str(row["name"]) for row in payload["categories"]}
    target_for_id: dict[int, int | None] = {}
    unknown_names: list[str] = []
    for category_id, name in categories.items():
        if name not in mapping:
            unknown_names.append(name)
            continue
        target_name = mapping[name]
        if target_name is None:
            target_for_id[category_id] = None
        elif target_name not in TARGETS:
            raise ValueError(f"unsupported target {target_name!r} for category {name!r}")
        else:
            target_for_id[category_id] = TARGETS[target_name]
    if unknown_names:
        raise ValueError(
            "mapping file lacks categories: " + ", ".join(sorted(unknown_names))
        )

    annotations: list[dict[str, Any]] = []
    for annotation in payload["annotations"]:
        target = target_for_id[int(annotation["category_id"])]
        if target is None:
            continue
        bbox = [float(value) for value in annotation["bbox"]]
        if len(bbox) != 4 or bbox[2] <= 0.0 or bbox[3] <= 0.0:
            continue
        row = dict(annotation)
        row["category_id"] = target
        row["bbox"] = bbox
        annotations.append(row)

    output = {
        **{key: value for key, value in payload.items() if key not in {"categories", "annotations"}},
        "categories": [
            {"id": category_id, "name": name}
            for name, category_id in sorted(TARGETS.items(), key=lambda item: item[1])
        ],
        "annotations": annotations,
        "hera_guard_external_audit": {
            "dataset_name": args.dataset_name,
            "license_note": args.license_note,
            "source_annotation_file": str(args.input.resolve()),
            "mapping_file": str(args.mapping.resolve()),
            "original_annotations": len(payload["annotations"]),
            "retained_annotations": len(annotations),
            "policy": "external labels used for coarse/objectness pretraining only",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["hera_guard_external_audit"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
