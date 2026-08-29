#!/usr/bin/env python3
"""Simulate a fold-heldout, per-10K budget for an expensive pixel verifier.

The cheap router sees detector metadata and box geometry only.  For each
formal fold it is trained on the other two folds to predict candidate
validity.  Candidates closest to the router's operating thresholds are sent
to the already cross-fitted pixel risk model; all remaining candidates retain
the cheap router probability.  This isolates the accuracy/latency trade-off
before any deployment code is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
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


def _key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(item["image_id"]),
        int(item["category_id"]),
        *(round(float(value), 5) for value in item["bbox"]),
        int(item.get("source_fold", -1)),
    )


def align_pixel_scores(
    base: Sequence[Mapping[str, Any]], pixel: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    """Align risk scores by immutable proposal identity and reject ambiguity."""

    mapping: dict[tuple[Any, ...], float] = {}
    for item in pixel:
        key = _key(item)
        if key in mapping:
            raise ValueError(f"duplicate pixel proposal key: {key}")
        mapping[key] = float(item["score"])
    scores = np.empty(len(base), dtype=np.float64)
    for index, item in enumerate(base):
        key = _key(item)
        if key not in mapping:
            raise ValueError(f"pixel score missing for proposal: {key}")
        scores[index] = mapping.pop(key)
    if mapping:
        raise ValueError(f"pixel predictions contain {len(mapping)} unmatched proposals")
    if not np.isfinite(scores).all():
        raise ValueError("pixel scores contain NaN/Inf")
    return scores


def normalize_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for stable_order, raw in enumerate(rows):
        x, y, width, height = (float(value) for value in raw["bbox"])
        if width <= 0.0 or height <= 0.0:
            raise ValueError("proposal has non-positive extent")
        fold = int(raw["source_fold"])
        if fold not in {0, 1, 2}:
            raise ValueError(f"invalid source_fold={fold}")
        result.append(
            {
                **dict(raw),
                "bbox_xyxy": [x, y, x + width, y + height],
                "fold": fold,
                "stable_order": stable_order,
            }
        )
    return result


def build_router_features(
    records: Sequence[Mapping[str, Any]],
    *,
    category_mapping: Mapping[int, str],
    image_sizes: Mapping[int, tuple[float, float]],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build deployable metadata-only features for budget routing."""

    columns = (
        "detector_score",
        "variant_coph",
        "variant_union",
        "coarse_ship",
        "coarse_aircraft",
        "coarse_vehicle",
        "log_short_edge",
        "log_area",
        "log_aspect",
        "center_x",
        "center_y",
        "edge_distance",
        "log_image_count",
        "score_rank_fraction",
    ) + tuple(f"fine_{index}" for index in range(25))
    by_image: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(records):
        by_image[int(item["image_id"])].append(index)
    ranks = np.zeros(len(records), dtype=np.float64)
    for indices in by_image.values():
        ordered = sorted(indices, key=lambda i: (-float(records[i]["score"]), i))
        denominator = max(1, len(ordered) - 1)
        for rank, index in enumerate(ordered):
            ranks[index] = rank / denominator

    matrix = np.zeros((len(records), len(columns)), dtype=np.float64)
    for index, item in enumerate(records):
        image_id = int(item["image_id"])
        category = int(item["category_id"])
        if not 0 <= category < 25:
            raise ValueError(f"invalid category_id={category}")
        coarse = category_mapping[category]
        x0, y0, x1, y1 = (float(value) for value in item["bbox_xyxy"])
        width, height = x1 - x0, y1 - y0
        image_width, image_height = image_sizes[image_id]
        cx = (x0 + x1) / (2.0 * image_width)
        cy = (y0 + y1) / (2.0 * image_height)
        edge = min(cx, cy, 1.0 - cx, 1.0 - cy)
        variant = str(item.get("source_variant", "UNION")).upper()
        values = [
            float(item.get("detector_score", item["score"])),
            float(variant == "COPH"),
            float(variant == "UNION"),
            float(coarse == "ship"),
            float(coarse == "aircraft"),
            float(coarse == "vehicle"),
            math.log1p(min(width, height)),
            math.log1p(width * height),
            math.log(max(width / height, height / width)),
            cx,
            cy,
            edge,
            math.log1p(len(by_image[image_id])),
            ranks[index],
        ]
        values.extend(float(category == fine) for fine in range(25))
        matrix[index] = values
    if not np.isfinite(matrix).all():
        raise RuntimeError("router feature matrix contains NaN/Inf")
    return matrix, columns


def _model(seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_depth=5,
        l2_regularization=2.0,
        class_weight="balanced",
        random_state=seed,
    )


def validity_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    targets: Sequence[float] = (0.10, 0.12, 0.15, 0.17, 0.20),
) -> list[float]:
    """Select deterministic proposal-validity thresholds for routing only."""

    grid = np.linspace(0.01, 0.99, 197)
    result: list[float] = []
    for target in targets:
        choices: list[tuple[int, int, float]] = []
        for threshold in grid:
            selected = probabilities >= threshold
            tp = int(labels[selected].sum())
            fp = int(selected.sum()) - tp
            fdr = fp / max(1, tp + fp)
            if fdr <= target:
                choices.append((tp, -fp, float(threshold)))
        result.append(max(choices)[2] if choices else 0.99)
    return sorted(set(result))


def select_per_image_budget(
    records: Sequence[Mapping[str, Any]], priorities: np.ndarray, budget: int
) -> np.ndarray:
    """Select at most ``budget`` stable candidates independently per image."""

    if budget < 0:
        raise ValueError("budget must be non-negative")
    selected = np.zeros(len(records), dtype=bool)
    by_image: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(records):
        by_image[int(item["image_id"])].append(index)
    for indices in by_image.values():
        ordered = sorted(indices, key=lambda i: (-float(priorities[i]), i))
        selected[ordered[:budget]] = True
    return selected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--pixel-risk-predictions", type=Path, required=True)
    parser.add_argument("--pixel-inference-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=(0, 128, 256, 512, 1024, 2048)
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    raw_gt = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    image_sizes = {
        int(item["id"]): (float(item["width"]), float(item["height"]))
        for item in raw_gt["images"]
    }
    base_raw = json.loads(args.base_predictions.read_text(encoding="utf-8"))
    pixel_raw = json.loads(args.pixel_risk_predictions.read_text(encoding="utf-8"))
    records = normalize_records(base_raw)
    pixel_scores = align_pixel_scores(records, pixel_raw)
    gt = load_coco_ground_truth(args.ground_truth)
    labels = candidate_validity_labels(
        records,
        gt_boxes=gt,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    features, columns = build_router_features(
        records,
        category_mapping=protocol.category_mapping,
        image_sizes=image_sizes,
    )
    folds = np.asarray([int(item["fold"]) for item in records], dtype=np.int64)
    cheap_scores = np.full(len(records), np.nan, dtype=np.float64)
    priorities = np.full(len(records), np.nan, dtype=np.float64)
    fold_contracts: list[dict[str, Any]] = []
    for held_out in (0, 1, 2):
        train = folds != held_out
        validation = folds == held_out
        model = _model(20260829 + held_out)
        model.fit(features[train], labels[train])
        cheap_scores[validation] = model.predict_proba(features[validation])[:, 1]
        train_probabilities = model.predict_proba(features[train])[:, 1]
        thresholds = validity_thresholds(labels[train], train_probabilities)
        validation_logits = np.log(np.clip(cheap_scores[validation], 1e-6, 1 - 1e-6)) - np.log(
            np.clip(1 - cheap_scores[validation], 1e-6, 1 - 1e-6)
        )
        threshold_logits = [math.log(t / (1.0 - t)) for t in thresholds]
        distance = np.min(
            np.abs(validation_logits[:, None] - np.asarray(threshold_logits)[None, :]), axis=1
        )
        priorities[validation] = -distance
        fold_contracts.append(
            {
                "held_out_fold": held_out,
                "n_train": int(train.sum()),
                "n_validation": int(validation.sum()),
                "routing_thresholds": thresholds,
            }
        )
    if not np.isfinite(cheap_scores).all() or not np.isfinite(priorities).all():
        raise RuntimeError("incomplete fold-heldout router outputs")

    inference = json.loads(args.pixel_inference_summary.read_text(encoding="utf-8"))
    seconds_per_crop = float(inference["elapsed_seconds"]) / float(
        inference["output_predictions"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    budget_rows: list[dict[str, Any]] = []
    for budget in sorted(set(args.budgets)):
        selected = select_per_image_budget(records, priorities, budget)
        hybrid = np.where(selected, pixel_scores, cheap_scores)
        output = []
        for item, score in zip(records, hybrid, strict=True):
            output.append(
                {
                    "image_id": int(item["image_id"]),
                    "category_id": int(item["category_id"]),
                    "bbox": [float(value) for value in item["bbox"]],
                    "score": float(score),
                    "source_fold": int(item["fold"]),
                    "source_model": str(item.get("source_model", "Y5")),
                    "source_variant": str(item.get("source_variant", "UNION")),
                }
            )
        path = args.output_dir / f"budget_{budget:04d}_predictions.json"
        path.write_text(json.dumps(output, ensure_ascii=False) + "\n", encoding="utf-8")
        counts = Counter(int(records[index]["image_id"]) for index in np.flatnonzero(selected))
        budget_rows.append(
            {
                "budget_per_image": budget,
                "selected_total": int(selected.sum()),
                "selected_per_image": {str(key): value for key, value in sorted(counts.items())},
                "estimated_pixel_seconds_per_image": {
                    str(image_id): count * seconds_per_crop
                    for image_id, count in sorted(counts.items())
                },
                "predictions": path.name,
                "predictions_sha256": _sha256(path),
            }
        )
    summary = {
        "status": "complete",
        "protocol": "fold_heldout_budgeted_tight_pixel_router_v1",
        "warning": (
            "Pseudo-10K is a deployment proxy. Estimated time covers pixel verification only; "
            "detector, tiling, fusion and JSON time must be added separately."
        ),
        "inputs": {
            "ground_truth_sha256": _sha256(args.ground_truth),
            "base_predictions_sha256": _sha256(args.base_predictions),
            "pixel_risk_predictions_sha256": _sha256(args.pixel_risk_predictions),
            "pixel_inference_summary_sha256": _sha256(args.pixel_inference_summary),
        },
        "feature_columns": list(columns),
        "candidate_count": len(records),
        "candidate_positive_rate": float(labels.mean()),
        "seconds_per_crop_observed": seconds_per_crop,
        "fold_contracts": fold_contracts,
        "budgets": budget_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
