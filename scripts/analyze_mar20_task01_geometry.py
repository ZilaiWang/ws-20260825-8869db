#!/usr/bin/env python3
"""Calibrate a review ranking on calibration only, then audit held-out geometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rsdet.grouping.contracts import atomic_write_json, sha256_file

FEATURES = (
    "exact_pixel",
    "phash_similarity",
    "formal_route_support",
    "formal_mutual_route_support",
    "inverse_best_rank",
    "best_similarity",
    "log_sift_mutual",
    "log_similarity_inliers",
    "similarity_inlier_ratio",
    "similarity_median_error_inverse",
    "sift_grid_min",
    "sift_hull_min",
    "sift_direction_entropy",
    "one_minus_sift_dominant_direction_ratio",
    "sift_ransac_repeat_pass_rate",
    "log_patch_mutual",
    "patch_similarity_median",
    "log_patch_matches_ge_0p8",
    "patch_grid_min",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MAR20 TASK-01 geometry")
    parser.add_argument("--pair-evidence", required=True)
    parser.add_argument("--geometry-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-q1-calibration-precision", type=float, default=0.90)
    parser.add_argument("--minimum-q1-calibration-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=202625)
    return parser.parse_args(argv)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _features(row: dict[str, str]) -> list[float]:
    rank = _number(row, "best_formal_rank", 999.0)
    median_error = _number(row, "similarity_median_error", 999.0)
    return [
        _number(row, "exact_pixel"),
        64.0 - _number(row, "phash_distance", 64.0),
        _number(row, "formal_route_support"),
        _number(row, "formal_mutual_route_support"),
        1.0 / (1.0 + max(rank, 0.0)),
        _number(row, "best_similarity"),
        math.log1p(_number(row, "sift_mutual_ratio_matches")),
        math.log1p(_number(row, "similarity_inliers")),
        _number(row, "similarity_inlier_ratio"),
        1.0 / (1.0 + max(median_error, 0.0)),
        min(_number(row, "sift_grid_occupancy_u"), _number(row, "sift_grid_occupancy_v")),
        min(_number(row, "sift_hull_fraction_u"), _number(row, "sift_hull_fraction_v")),
        _number(row, "sift_direction_entropy"),
        1.0 - _number(row, "sift_dominant_direction_ratio", 1.0),
        _number(row, "sift_ransac_repeat_pass_rate"),
        math.log1p(_number(row, "patch_mutual_count")),
        _number(row, "patch_similarity_median"),
        math.log1p(_number(row, "patch_matches_ge_0p8")),
        min(_number(row, "patch_grid_occupancy_u"), _number(row, "patch_grid_occupancy_v")),
    ]


def _metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    if not len(labels):
        return {"count": 0}
    predicted = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )
    return {
        "count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "predicted_positive_count": int(predicted.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) == 2 else None,
        "average_precision": float(average_precision_score(labels, scores))
        if len(set(labels.tolist())) == 2
        else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    evidence_path = Path(args.pair_evidence).expanduser().resolve()
    summary_path = Path(args.geometry_summary).expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "pass" or summary.get("pair_evidence_sha256") != sha256_file(
        evidence_path
    ):
        raise ValueError("geometry evidence is not admitted")
    rows = _read(evidence_path)
    controls = [
        row
        for row in rows
        if row.get("known_binary_role") in {"positive", "negative"}
        and row.get("known_split") in {"calibration", "held_out_audit"}
    ]
    calibration = [row for row in controls if row["known_split"] == "calibration"]
    heldout = [row for row in controls if row["known_split"] == "held_out_audit"]
    if len(calibration) < 100 or len(heldout) < 30:
        raise ValueError("insufficient calibration/held-out controls")

    def matrix(values: list[dict[str, str]]) -> np.ndarray:
        return np.asarray([_features(row) for row in values], dtype=np.float64)

    def labels(values: list[dict[str, str]]) -> np.ndarray:
        return np.asarray([row["known_binary_role"] == "positive" for row in values], dtype=int)

    x_cal, y_cal = matrix(calibration), labels(calibration)
    x_held, y_held = matrix(heldout), labels(heldout)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=args.seed,
            solver="liblinear",
        ),
    )
    model.fit(x_cal, y_cal)
    score_cal = model.predict_proba(x_cal)[:, 1]
    score_held = model.predict_proba(x_held)[:, 1]
    candidates = []
    for threshold in sorted(set(score_cal.tolist()), reverse=True):
        predicted = score_cal >= threshold
        count = int(predicted.sum())
        if count < args.minimum_q1_calibration_count:
            continue
        precision = float(y_cal[predicted].mean())
        recall = float(y_cal[predicted].sum() / max(y_cal.sum(), 1))
        if precision + 1e-12 >= args.minimum_q1_calibration_precision:
            candidates.append((recall, count, threshold, precision))
    if not candidates:
        threshold = float(np.quantile(score_cal[y_cal == 1], 0.75))
        threshold_status = "fallback_positive_q75"
    else:
        _, _, threshold, _ = max(candidates, key=lambda value: (value[0], value[1], value[2]))
        threshold = float(threshold)
        threshold_status = "calibration_precision_constrained"

    all_scores = model.predict_proba(matrix(rows))[:, 1]
    assignment_rows = []
    for row, score in zip(rows, all_scores, strict=True):
        sift_stable = (
            _number(row, "similarity_inliers") >= 8
            and _number(row, "similarity_inlier_ratio") >= 0.20
            and min(
                _number(row, "sift_grid_occupancy_u"),
                _number(row, "sift_grid_occupancy_v"),
            )
            >= 3
            and _number(row, "sift_dominant_direction_ratio", 1.0) <= 0.85
        )
        patch_stable = (
            _number(row, "patch_matches_ge_0p8") >= 10
            and min(
                _number(row, "patch_grid_occupancy_u"),
                _number(row, "patch_grid_occupancy_v"),
            )
            >= 4
        )
        if _number(row, "exact_pixel") >= 1:
            grade = "Q0"
        elif score >= threshold and (sift_stable or patch_stable):
            grade = "Q1"
        elif score >= 0.5 or sift_stable or patch_stable:
            grade = "Q2"
        elif _number(row, "formal_route_support") >= 2 or _number(row, "best_formal_rank", 999) <= 10:
            grade = "Q3"
        else:
            grade = "Q4"
        assignment_rows.append(
            {
                "pair_uid": row["pair_uid"],
                "review_score": f"{score:.9g}",
                "queue_grade": grade,
                "sift_stable": int(sift_stable),
                "patch_stable": int(patch_stable),
                "queue_source": row["queue_source"],
                "target_target": row["target_target"],
                "target_bridge": row["target_bridge"],
                "bridge_bridge": row["bridge_bridge"],
                "cross_official_side": row["cross_official_side"],
            }
        )
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    assignment_path = output / "geometry_queue_assignments.csv"
    with assignment_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(assignment_rows[0]))
        writer.writeheader()
        writer.writerows(assignment_rows)
    scaler = model.named_steps["standardscaler"]
    logistic = model.named_steps["logisticregression"]
    decision = {
        "status": "ready_for_blind_review_pack",
        "protocol": "calibration_only_logistic_review_ranking_v1",
        "feature_names": list(FEATURES),
        "threshold_status": threshold_status,
        "q1_threshold": threshold,
        "threshold_inputs": {
            "minimum_calibration_precision": args.minimum_q1_calibration_precision,
            "minimum_calibration_count": args.minimum_q1_calibration_count,
        },
        "calibration_metrics": _metrics(y_cal, score_cal, threshold),
        "heldout_metrics": _metrics(y_held, score_held, threshold),
        "model": {
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "coefficient": logistic.coef_[0].tolist(),
            "intercept": float(logistic.intercept_[0]),
            "random_seed": args.seed,
        },
        "queue_counts": dict(sorted(Counter(row["queue_grade"] for row in assignment_rows).items())),
        "pair_evidence_sha256": sha256_file(evidence_path),
        "assignment_sha256": sha256_file(assignment_path),
        "selection_uses_heldout": False,
        "heldout_is_audit_only": True,
        "q1_is_review_priority_not_automatic_edge": True,
        "formal_grouping_admission": False,
    }
    atomic_write_json(output / "geometry_calibration_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
