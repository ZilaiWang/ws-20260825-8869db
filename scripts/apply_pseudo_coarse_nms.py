#!/usr/bin/env python3
"""Apply one evidence-frozen, coarse-specific same-fine NMS policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coarse_name(category_id: int) -> str:
    if 0 <= category_id < 4:
        return "ship"
    if 4 <= category_id < 24:
        return "aircraft"
    if category_id == 24:
        return "vehicle"
    raise ValueError(f"invalid category_id={category_id}")


def apply_policy(
    rows: list[dict[str, Any]], *, thresholds: dict[str, float]
) -> list[dict[str, Any]]:
    import torch
    from torchvision.ops import nms

    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, item in enumerate(rows):
        groups[(int(item["image_id"]), int(item["category_id"]))].append(index)
    keep: list[int] = []
    for image_id, category_id in sorted(groups):
        indices = groups[(image_id, category_id)]
        boxes = []
        scores = []
        for index in indices:
            x, y, width, height = (float(value) for value in rows[index]["bbox"])
            boxes.append([x, y, x + width, y + height])
            scores.append(float(rows[index].get("detector_score", rows[index]["score"])))
        selected = nms(
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(scores, dtype=torch.float32),
            thresholds[coarse_name(category_id)],
        ).tolist()
        keep.extend(indices[int(local)] for local in selected)
    return [dict(rows[index]) for index in sorted(keep)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--ship-iou", type=float, default=0.60)
    parser.add_argument("--aircraft-iou", type=float, default=0.50)
    parser.add_argument("--vehicle-iou", type=float, default=0.50)
    parser.add_argument("--restore-detector-score", action="store_true")
    args = parser.parse_args()
    thresholds = {
        "ship": args.ship_iou,
        "aircraft": args.aircraft_iou,
        "vehicle": args.vehicle_iou,
    }
    if any(not 0.0 < value <= 1.0 for value in thresholds.values()):
        raise ValueError("NMS thresholds must be in (0,1]")
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if args.restore_detector_score:
        rows = [
            {
                **item,
                "score": float(item.get("detector_score", item["score"])),
            }
            for item in rows
        ]
    output = apply_policy(rows, thresholds=thresholds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False) + "\n", encoding="utf-8")
    before = Counter(coarse_name(int(item["category_id"])) for item in rows)
    after = Counter(coarse_name(int(item["category_id"])) for item in output)
    summary = {
        "status": "complete",
        "protocol": "evidence_frozen_coarse_specific_same_fine_nms_v1",
        "thresholds": thresholds,
        "restore_detector_score": args.restore_detector_score,
        "input_count": len(rows),
        "output_count": len(output),
        "input_by_coarse": dict(before),
        "output_by_coarse": dict(after),
        "input_sha256": _sha256(args.input),
        "output_sha256": _sha256(args.output),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
