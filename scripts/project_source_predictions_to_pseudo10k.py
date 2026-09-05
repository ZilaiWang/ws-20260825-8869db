#!/usr/bin/env python3
"""Project detections on native source crops into their pseudo-10K cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project(
    predictions: list[dict[str, Any]], mapping: dict[str, Any]
) -> list[dict[str, Any]]:
    by_source: dict[int, list[dict[str, Any]]] = {}
    for row in predictions:
        by_source.setdefault(int(row["image_id"]), []).append(row)
    output: list[dict[str, Any]] = []
    for occurrence in mapping["occurrences"]:
        source_id = int(occurrence["source_image_id"])
        scale = float(occurrence["scale"])
        x_offset = float(occurrence["x_offset"])
        y_offset = float(occurrence["y_offset"])
        for row in by_source.get(source_id, []):
            x, y, width, height = (float(value) for value in row["bbox"])
            output.append(
                {
                    "image_id": int(occurrence["pseudo_image_id"]),
                    "category_id": int(row["category_id"]),
                    "bbox": [
                        x_offset + x * scale,
                        y_offset + y * scale,
                        width * scale,
                        height * scale,
                    ],
                    "score": float(row["score"]),
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-predictions", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists():
        raise FileExistsError("output and summary must not already exist")
    predictions = json.loads(args.source_predictions.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError("source predictions must be a COCO detection list")
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    output = project(predictions, mapping)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(",", ":")) + "\n", encoding="utf-8")
    summary = {
        "status": "complete",
        "role": "native_source_predictions_projected_to_pseudo10k",
        "source_predictions": len(predictions),
        "projected_predictions": len(output),
        "input_sha256": {
            "source_predictions": _sha256(args.source_predictions),
            "mapping": _sha256(args.mapping),
        },
        "output_sha256": _sha256(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
