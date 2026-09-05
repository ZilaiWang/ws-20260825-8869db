#!/usr/bin/env python3
"""Audit whether a low-floor COCO prediction ledger saturates ``max_det``."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(rows: Sequence[dict[str, Any]], max_det: int) -> dict[str, Any]:
    if max_det <= 0:
        raise ValueError("max_det must be positive")
    per_image: Counter[int] = Counter()
    per_image_label: dict[int, Counter[int]] = defaultdict(Counter)
    for row in rows:
        image_id = int(row["image_id"])
        label = int(row["category_id"])
        if not 0 <= label < 25:
            raise ValueError(f"invalid official category_id: {label}")
        per_image[image_id] += 1
        per_image_label[image_id][label] += 1
    counts = sorted(per_image.values())
    saturated = sorted(image_id for image_id, count in per_image.items() if count >= max_det)

    def nearest_rank(q: float) -> float:
        if not counts:
            return 0.0
        index = min(round(q * (len(counts) - 1)), len(counts) - 1)
        return float(counts[index])

    return {
        "schema_version": "max_det_saturation_audit_v1",
        "max_det": max_det,
        "prediction_count": len(rows),
        "image_count_with_predictions": len(per_image),
        "saturated_image_count": len(saturated),
        "saturated_image_ids": saturated,
        "count_distribution_nonempty_images": {
            "min": counts[0] if counts else 0,
            "p50": nearest_rank(0.50),
            "p90": nearest_rank(0.90),
            "p95": nearest_rank(0.95),
            "p99": nearest_rank(0.99),
            "max": counts[-1] if counts else 0,
        },
        "saturated_label_counts": {
            str(image_id): {
                str(label): count for label, count in sorted(per_image_label[image_id].items())
            }
            for image_id in saturated
        },
        "decision": (
            "rerun_paired_models_with_higher_max_det"
            if saturated
            else "no_observed_max_det_saturation"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("predictions must be a COCO JSON list")
    result = audit(rows, args.max_det)
    result["input_sha256"] = _sha256(args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
