#!/usr/bin/env python3
"""Rerank two-view pseudo-10K proposals using identity-view support.

The augmented prediction set remains the candidate source.  A candidate is
"supported" only when a one-to-one, same-image, same-fine-class identity-view
candidate overlaps it.  Unsupported candidates are retained (so candidate
recall is unchanged) but conservatively downweighted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xyxy(item: dict[str, Any]) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in item["bbox"])
    return x, y, x + width, y + height


def _iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = _xyxy(first)
    bx1, by1, bx2, by2 = _xyxy(second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def rerank_with_support(
    identity: list[dict[str, Any]],
    augmented: list[dict[str, Any]],
    *,
    match_iou: float,
    unsupported_factor: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not 0.0 <= match_iou <= 1.0:
        raise ValueError("match_iou must be in [0, 1]")
    if not 0.0 <= unsupported_factor <= 1.0:
        raise ValueError("unsupported_factor must be in [0, 1]")
    identity_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    augmented_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for item in identity:
        identity_groups[(int(item["image_id"]), int(item["category_id"]))].append(item)
    for item in augmented:
        augmented_groups[(int(item["image_id"]), int(item["category_id"]))].append(item)

    output: list[dict[str, Any]] = []
    supported = 0
    unsupported = 0
    for key in sorted(augmented_groups):
        base = sorted(
            identity_groups.get(key, []),
            key=lambda item: -float(item["score"]),
        )
        used: set[int] = set()
        candidates = sorted(
            augmented_groups[key], key=lambda item: -float(item["score"])
        )
        for item in candidates:
            best_index = -1
            best_iou = -1.0
            for index, base_item in enumerate(base):
                if index in used:
                    continue
                overlap = _iou(item, base_item)
                if overlap > best_iou:
                    best_index = index
                    best_iou = overlap
            result = dict(item)
            augmented_score = float(item["score"])
            if best_index >= 0 and best_iou >= match_iou:
                used.add(best_index)
                identity_score = float(base[best_index]["score"])
                result["score"] = math.sqrt(
                    max(0.0, augmented_score) * max(0.0, identity_score)
                )
                result["tta_supported"] = True
                result["tta_identity_iou"] = best_iou
                result["tta_identity_score"] = identity_score
                supported += 1
            else:
                result["score"] = augmented_score * unsupported_factor
                result["tta_supported"] = False
                result["tta_identity_iou"] = max(0.0, best_iou)
                unsupported += 1
            output.append(result)
    output.sort(
        key=lambda item: (
            int(item["image_id"]),
            -float(item["score"]),
            int(item["category_id"]),
        )
    )
    return output, {"supported": supported, "unsupported": unsupported}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-pred", type=Path, required=True)
    parser.add_argument("--augmented-pred", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--unsupported-factor", type=float, default=0.25)
    args = parser.parse_args()
    identity = json.loads(args.identity_pred.read_text(encoding="utf-8"))
    augmented = json.loads(args.augmented_pred.read_text(encoding="utf-8"))
    if not isinstance(identity, list) or not isinstance(augmented, list):
        raise ValueError("prediction files must contain COCO prediction lists")
    output, counts = rerank_with_support(
        identity,
        augmented,
        match_iou=args.match_iou,
        unsupported_factor=args.unsupported_factor,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "complete",
        "protocol": "identity_supported_two_view_rerank_v1",
        "match_iou": args.match_iou,
        "unsupported_factor": args.unsupported_factor,
        "identity_predictions": len(identity),
        "augmented_predictions": len(augmented),
        **counts,
        "identity_sha256": _sha256(args.identity_pred),
        "augmented_sha256": _sha256(args.augmented_pred),
        "output_sha256": _sha256(args.output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
