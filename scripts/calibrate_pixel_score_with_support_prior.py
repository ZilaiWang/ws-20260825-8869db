#!/usr/bin/env python3
"""Cross-fit a conservative source-support prior onto pixel risk scores.

The pixel score remains the primary posterior.  For each held-out fold, only
the other two folds estimate the validity odds associated with source support
count.  The train-fold odds ratio is then multiplied with the held-out pixel
odds.  This avoids fitting a second high-capacity stacker on six mosaics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.multi_detector_oer import candidate_validity_labels
from rsdet.evaluation.coco import load_coco_ground_truth
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def smoothed_rate(positive: int, total: int, *, alpha: float) -> float:
    if total < 0 or not 0 <= positive <= total:
        raise ValueError("invalid positive/total counts")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    return (positive + alpha) / (total + 2.0 * alpha)


def combine_probability_with_prior(
    probability: float,
    *,
    support_rate: float,
    global_rate: float,
    epsilon: float = 1e-6,
) -> float:
    values = (probability, support_rate, global_rate)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("probabilities must be finite")
    probability, support_rate, global_rate = (
        min(1.0 - epsilon, max(epsilon, value)) for value in values
    )
    base_odds = probability / (1.0 - probability)
    likelihood_ratio = (support_rate / (1.0 - support_rate)) / (
        global_rate / (1.0 - global_rate)
    )
    odds = base_odds * likelihood_ratio
    return odds / (1.0 + odds)


def crossfit_support_prior_scores(
    rows: Sequence[Mapping[str, Any]],
    labels: np.ndarray,
    *,
    alpha: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) != len(labels):
        raise ValueError("rows and labels differ in length")
    folds = np.asarray([int(row["source_fold"]) for row in rows], dtype=np.int64)
    if set(folds.tolist()) != {0, 1, 2}:
        raise ValueError("all three source folds are required")
    scores = np.full(len(rows), np.nan, dtype=np.float64)
    audits: list[dict[str, Any]] = []
    for held_out in (0, 1, 2):
        train = folds != held_out
        validation = folds == held_out
        train_labels = labels[train]
        global_rate = smoothed_rate(
            int(train_labels.sum()), int(train.sum()), alpha=alpha
        )
        rates: dict[int, float] = {}
        counts: dict[str, dict[str, int | float]] = {}
        support_values = sorted({int(rows[index]["source_support_count"]) for index in np.flatnonzero(train)})
        for support_count in support_values:
            mask = train & np.asarray(
                [int(row["source_support_count"]) == support_count for row in rows],
                dtype=bool,
            )
            positive = int(labels[mask].sum())
            total = int(mask.sum())
            rate = smoothed_rate(positive, total, alpha=alpha)
            rates[support_count] = rate
            counts[str(support_count)] = {
                "positive": positive,
                "total": total,
                "smoothed_rate": rate,
            }
        for index in np.flatnonzero(validation):
            support_count = int(rows[index]["source_support_count"])
            scores[index] = combine_probability_with_prior(
                float(rows[index]["score"]),
                support_rate=rates.get(support_count, global_rate),
                global_rate=global_rate,
            )
        audits.append(
            {
                "held_out_fold": held_out,
                "training_global_rate": global_rate,
                "training_support_counts": counts,
                "validation_count": int(validation.sum()),
            }
        )
    if not np.isfinite(scores).all():
        raise RuntimeError("incomplete cross-fit support scores")
    outputs = []
    for row, score in zip(rows, scores, strict=True):
        item = dict(row)
        item["pixel_score_before_support_prior"] = float(row["score"])
        item["score"] = float(score)
        outputs.append(item)
    return outputs, audits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    gt = load_coco_ground_truth(args.ground_truth)
    rows = json.loads(args.predictions.read_text(encoding="utf-8"))
    normalized = []
    for item in rows:
        x, y, width, height = (float(value) for value in item["bbox"])
        row = dict(item)
        row["bbox_xyxy"] = [x, y, x + width, y + height]
        normalized.append(row)
    labels = candidate_validity_labels(
        normalized,
        gt_boxes=gt,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    outputs, audits = crossfit_support_prior_scores(normalized, labels, alpha=args.alpha)
    for item in outputs:
        item.pop("bbox_xyxy", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outputs, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": "complete",
        "protocol": "pixel_score_plus_crossfit_support_odds_prior_v1",
        "warning": "Pseudo-10K diagnostic; support priors are trained on the other two folds only.",
        "alpha": args.alpha,
        "input_sha256": _sha256(args.predictions),
        "output_sha256": _sha256(args.output),
        "candidate_count": len(outputs),
        "audits": audits,
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
