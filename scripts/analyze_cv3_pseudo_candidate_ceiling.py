#!/usr/bin/env python3
"""Decompose pseudo-10K candidate recall before score thresholding.

This is an oracle-availability diagnostic, not an evaluation metric.  Each GT
is inspected independently to answer whether the prediction ledger contains a
geometrically valid candidate with the correct fine class, only the correct
coarse class, only a wrong coarse class, or no sufficiently overlapping box.
The exact official matcher remains the authority for reported Recall/FDR.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.data.xh_dataset import FINE_NAMES
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def _iou(left: list[float], right: list[float]) -> float:
    lx1, ly1, lw, lh = (float(value) for value in left)
    rx1, ry1, rw, rh = (float(value) for value in right)
    lx2, ly2 = lx1 + lw, ly1 + lh
    rx2, ry2 = rx1 + rw, ry1 + rh
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, lw) * max(0.0, lh) + max(0.0, rw) * max(0.0, rh) - inter
    return inter / union if union > 0.0 else 0.0


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "p10", "p25", "p50", "p75", "p90", "max")}
    ordered = sorted(values)

    def pick(fraction: float) -> float:
        index = round(fraction * (len(ordered) - 1))
        return float(ordered[index])

    return {
        "min": float(ordered[0]),
        "p10": pick(0.10),
        "p25": pick(0.25),
        "p50": pick(0.50),
        "p75": pick(0.75),
        "p90": pick(0.90),
        "max": float(ordered[-1]),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["status"]) for row in rows)
    total = len(rows)
    return {
        "gt": total,
        "counts": dict(sorted(counts.items())),
        "rates": {
            key: value / total if total else 0.0 for key, value in sorted(counts.items())
        },
        "correct_fine_best_score_quantiles": _quantiles(
            [float(row["best_fine_score"]) for row in rows if row["best_fine_score"] is not None]
        ),
        "best_any_iou_quantiles": _quantiles([float(row["best_any_iou"]) for row in rows]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    raw_gt = json.loads(args.gt.read_text(encoding="utf-8"))
    raw_pred = json.loads(args.pred.read_text(encoding="utf-8"))
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in raw_pred:
        score = float(prediction["score"])
        if not math.isfinite(score):
            raise ValueError("prediction contains a non-finite score")
        by_image[int(prediction["image_id"])].append(prediction)

    rows: list[dict[str, Any]] = []
    for annotation in raw_gt["annotations"]:
        image_id = int(annotation["image_id"])
        fine = int(annotation["category_id"])
        coarse = protocol.category_mapping[fine]
        threshold = float(protocol.iou_thresholds[coarse])
        candidates = by_image.get(image_id, [])
        scored = [
            (
                _iou(annotation["bbox"], item["bbox"]),
                float(item["score"]),
                int(item["category_id"]),
            )
            for item in candidates
        ]
        valid = [item for item in scored if item[0] >= threshold]
        fine_valid = [item for item in valid if item[2] == fine]
        coarse_valid = [
            item for item in valid if protocol.category_mapping[item[2]] == coarse
        ]
        if fine_valid:
            status = "correct_fine_candidate"
        elif coarse_valid:
            status = "fine_class_failure"
        elif valid:
            status = "coarse_class_failure"
        elif scored:
            status = "localization_failure"
        else:
            status = "no_candidate"
        best_fine_score = max((item[1] for item in fine_valid), default=None)
        rows.append(
            {
                "annotation_id": int(annotation.get("id", len(rows))),
                "image_id": image_id,
                "fine_category_id": fine,
                "fine_name": FINE_NAMES[fine],
                "coarse": coarse,
                "iou_threshold": threshold,
                "status": status,
                "best_any_iou": max((item[0] for item in scored), default=0.0),
                "best_fine_score": best_fine_score,
            }
        )

    per_coarse = {
        coarse: _summarize([row for row in rows if row["coarse"] == coarse])
        for coarse in ("ship", "aircraft", "vehicle")
    }
    per_fine = {
        name: _summarize([row for row in rows if row["fine_name"] == name])
        for name in FINE_NAMES
    }
    payload = {
        "status": "complete",
        "protocol": "independent_gt_oracle_candidate_availability_v1",
        "warning": (
            "Not an official score: GTs are inspected independently and a candidate may support "
            "more than one GT. Use the official prediction-first matcher for Recall/FDR."
        ),
        "overall": _summarize(rows),
        "per_coarse": per_coarse,
        "per_fine": per_fine,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "protocol", "overall", "per_coarse")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
