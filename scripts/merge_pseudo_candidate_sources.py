#!/usr/bin/env python3
"""Merge any number of pseudo-10K candidate sources with deterministic NMS."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.multi_detector_oer import class_aware_nms_records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip().upper()
    if not label:
        raise argparse.ArgumentTypeError("source label cannot be empty")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"source path does not exist: {path}")
    return label, path


def normalize_sources(
    sources: Iterable[tuple[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    order = 0
    for source_label, rows in sources:
        model_family = "M3" if source_label.startswith("M3") else "Y5"
        for raw in rows:
            x, y, width, height = (float(value) for value in raw["bbox"])
            score = float(raw["score"])
            if width <= 0.0 or height <= 0.0:
                continue
            if not all(math.isfinite(value) for value in (x, y, width, height, score)):
                raise ValueError("candidate contains NaN/Inf")
            fold = int(raw["source_fold"])
            if fold not in {0, 1, 2}:
                raise ValueError(f"invalid source_fold={fold}")
            output.append(
                {
                    **dict(raw),
                    "bbox_xyxy": [x, y, x + width, y + height],
                    "score": score,
                    "detector_score": score,
                    "source_model": model_family,
                    "source_variant": source_label,
                    "stable_order": order,
                }
            )
            order += 1
    return output


def to_coco(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in rows:
        x0, y0, x1, y1 = (float(value) for value in item["bbox_xyxy"])
        output.append(
            {
                "image_id": int(item["image_id"]),
                "category_id": int(item["category_id"]),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": float(item["score"]),
                "detector_score": float(item["detector_score"]),
                "source_fold": int(item["source_fold"]),
                "source_model": str(item["source_model"]),
                "source_variant": str(item["source_variant"]),
            }
        )
    return output


def fast_class_aware_nms(
    records: list[dict[str, Any]], *, iou_threshold: float
) -> list[dict[str, Any]]:
    """Vectorized per-image/per-class NMS with deterministic output ordering."""

    if not records:
        return []
    import torch
    from torchvision.ops import nms

    # Do not call one global ``batched_nms`` here.  Its coordinate-offset path
    # still compares tens of thousands of boxes in one kernel.  Trial 10K has
    # only 6 images x 25 classes, so explicit groups are materially faster.
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, item in enumerate(records):
        groups[(int(item["image_id"]), int(item["category_id"]))].append(index)
    kept_indices: list[int] = []
    for key in sorted(groups):
        indices = groups[key]
        boxes = torch.tensor(
            [records[index]["bbox_xyxy"] for index in indices], dtype=torch.float32
        )
        scores = torch.tensor(
            [float(records[index]["score"]) for index in indices], dtype=torch.float32
        )
        local_keep = nms(boxes, scores, iou_threshold).cpu().tolist()
        kept_indices.extend(indices[int(index)] for index in local_keep)
    selected = [records[index] for index in kept_indices]
    selected.sort(key=lambda item: int(item["stable_order"]))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--nms-backend", choices=("torchvision", "python"), default="torchvision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    loaded = [
        (label, json.loads(path.read_text(encoding="utf-8")))
        for label, path in args.source
    ]
    records = normalize_sources(loaded)
    kept = (
        fast_class_aware_nms(records, iou_threshold=args.nms_iou)
        if args.nms_backend == "torchvision"
        else class_aware_nms_records(records, iou_threshold=args.nms_iou)
    )
    predictions = to_coco(kept)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    input_counts = Counter(item["source_variant"] for item in records)
    output_counts = Counter(item["source_variant"] for item in kept)
    summary = {
        "status": "complete",
        "protocol": "multi_source_class_aware_nms_v1",
        "nms_iou": args.nms_iou,
        "nms_backend": args.nms_backend,
        "sources": [
            {"label": label, "path": str(path), "sha256": _sha256(path)}
            for label, path in args.source
        ],
        "input_candidates": len(records),
        "output_candidates": len(predictions),
        "input_by_source": dict(sorted(input_counts.items())),
        "output_by_source": dict(sorted(output_counts.items())),
        "output_sha256": _sha256(args.output),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
