"""Cross-fit score calibration for the formal YOLO OOF predictions.

The module deliberately separates four methods:

``C0``
    One global threshold selected on the other folds.
``C1``
    One threshold per official coarse class.
``C2``
    Fine-class prior adjustment followed by one global threshold.
``C3``
    A FRACAL-inspired fine-class prior and fractal-dimension adjustment.

The stored OOF predictions contain one post-NMS scalar score and one selected
fine class.  They do not contain the complete pre-NMS class-logit vector.
Consequently C2/C3 are explicitly a post-NMS screening experiment, not an
exact reproduction of FRACAL Eq. (10)/(11).  An exact integration is only
admitted if this inexpensive screen produces stable held-out gains.
"""

from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rsdet.analysis.crossfit_thresholds import (
    _filter_by_score,
    _merge_folds,
    evaluate_ranking_workpoint,
    evaluate_workpoint,
    load_cv3_aggregate,
    load_gt_from_formal_crop_manifest,
    scan_global_threshold,
    split_by_fold,
    split_gt_by_fold,
)
from rsdet.data.xh_dataset import FINE_NAMES, coarse_name
from rsdet.evaluation.official_metric import evaluate_ranking_metrics
from rsdet.evaluation.protocol import EvaluationProtocol

CALIBRATION_CONTRACT_VERSION = "yolo_crossfit_calibration_v1"


@dataclass(frozen=True)
class SpatialAnnotation:
    """One tight GT annotation with a normalised centre."""

    image_id: int
    fold: int
    category_id: int
    center_x: float
    center_y: float


@dataclass(frozen=True)
class ClassSpatialStatistics:
    """Fine-class frequency priors and box-counting dimensions."""

    counts: tuple[int, ...]
    priors: tuple[float, ...]
    fractal_dimensions: tuple[float, ...]
    source_folds: tuple[int, ...]


def _finite_probability(value: float, field: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} 必须是 [0, 1] 内的有限数")
    return number


def load_spatial_annotations(
    manifest_path: str | Path,
    *,
    expected_annotations: int,
) -> tuple[SpatialAnnotation, ...]:
    """Load one spatial record per ``tight`` annotation."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[SpatialAnnotation] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "crop_policy",
            "annotation_uid",
            "formal_image_id",
            "fold",
            "class_id",
            "source_width",
            "source_height",
            "gt_x0",
            "gt_y0",
            "gt_x1",
            "gt_y1",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"formal crop manifest 缺少列: {sorted(missing)}")
        for row in reader:
            if row["crop_policy"] != "tight":
                continue
            uid = row["annotation_uid"]
            if uid in seen:
                raise ValueError(f"重复 tight annotation_uid: {uid}")
            seen.add(uid)
            width = float(row["source_width"])
            height = float(row["source_height"])
            if width <= 0.0 or height <= 0.0:
                raise ValueError(f"非法源图尺寸: {width}x{height}")
            center_x = (float(row["gt_x0"]) + float(row["gt_x1"])) / (2.0 * width)
            center_y = (float(row["gt_y0"]) + float(row["gt_y1"])) / (2.0 * height)
            records.append(
                SpatialAnnotation(
                    image_id=int(row["formal_image_id"]),
                    fold=int(row["fold"]),
                    category_id=int(row["class_id"]),
                    center_x=_finite_probability(center_x, "center_x"),
                    center_y=_finite_probability(center_y, "center_y"),
                )
            )
    if len(records) != expected_annotations:
        raise ValueError(
            f"tight spatial annotations={len(records)} != expected={expected_annotations}"
        )
    return tuple(records)


def estimate_fractal_dimension(
    points: Sequence[tuple[float, float]],
    *,
    max_grid_size: int = 64,
) -> float:
    """Estimate the box-counting slope using the paper's quadratic rule.

    Grid sizes are ``1..floor(sqrt(n))`` and capped for predictable runtime.
    Classes with too little evidence receive the paper's neutral value ``1``.
    """

    if max_grid_size < 2:
        raise ValueError("max_grid_size 必须 >= 2")
    if len(points) < 4:
        return 1.0
    largest = min(max_grid_size, int(math.floor(math.sqrt(len(points)))))
    if largest < 2:
        return 1.0
    samples: list[tuple[float, float]] = []
    for grid_size in range(1, largest + 1):
        occupied = {
            (
                min(grid_size - 1, int(_finite_probability(x, "x") * grid_size)),
                min(grid_size - 1, int(_finite_probability(y, "y") * grid_size)),
            )
            for x, y in points
        }
        if grid_size > 1 and occupied:
            samples.append((math.log(grid_size), math.log(len(occupied))))
    if len(samples) < 2:
        return 1.0
    mean_x = sum(item[0] for item in samples) / len(samples)
    mean_y = sum(item[1] for item in samples) / len(samples)
    denominator = sum((item[0] - mean_x) ** 2 for item in samples)
    if denominator <= 1e-12:
        return 1.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in samples) / denominator
    return min(2.0, max(1e-3, slope))


def estimate_class_spatial_statistics(
    annotations: Sequence[SpatialAnnotation],
    *,
    source_folds: Iterable[int],
    class_count: int = len(FINE_NAMES),
    max_grid_size: int = 64,
) -> ClassSpatialStatistics:
    """Estimate statistics from the outer-training folds only."""

    folds = tuple(sorted(set(int(value) for value in source_folds)))
    if not folds:
        raise ValueError("source_folds 不能为空")
    points: list[list[tuple[float, float]]] = [[] for _ in range(class_count)]
    for item in annotations:
        if item.fold not in folds:
            continue
        if not 0 <= item.category_id < class_count:
            raise ValueError(f"非法 category_id={item.category_id}")
        points[item.category_id].append((item.center_x, item.center_y))
    counts = tuple(len(items) for items in points)
    if any(value <= 0 for value in counts):
        missing = [index for index, value in enumerate(counts) if value <= 0]
        raise ValueError(f"训练 folds 缺少细类: {missing}")
    total = sum(counts)
    priors = tuple(value / total for value in counts)
    fractals = tuple(
        estimate_fractal_dimension(items, max_grid_size=max_grid_size) for items in points
    )
    return ClassSpatialStatistics(
        counts=counts,
        priors=priors,
        fractal_dimensions=fractals,
        source_folds=folds,
    )


def calibrate_binary_score(
    score: float,
    *,
    category_id: int,
    statistics: ClassSpatialStatistics,
    beta: float,
    spatial_lambda: float,
) -> float:
    """Apply the FRACAL binary-detector scalar approximation.

    ``spatial_lambda=0`` is the C2 class-prior adjustment.  Positive values add
    the C3 fractal term.  The final multiplication by the original sigmoid
    score follows the foreground filter in FRACAL Eq. (11).
    """

    probability = _finite_probability(score, "score")
    if not 0 <= category_id < len(statistics.priors):
        raise ValueError(f"非法 category_id={category_id}")
    if not math.isfinite(beta) or beta <= 1.0:
        raise ValueError("beta 必须 > 1")
    if not math.isfinite(spatial_lambda) or spatial_lambda < 0.0:
        raise ValueError("spatial_lambda 必须 >= 0")
    epsilon = 1e-7
    clipped = min(1.0 - epsilon, max(epsilon, probability))
    logit = math.log(clipped / (1.0 - clipped))
    class_count = len(statistics.priors)
    log_beta = math.log(beta)
    class_term = (
        -math.log(statistics.priors[category_id]) / log_beta
        + math.log(1.0 / class_count) / log_beta
    )
    space_term = 0.0
    if spatial_lambda > 0.0:
        weights = [value**spatial_lambda for value in statistics.fractal_dimensions]
        normalizer = sum(weights)
        spatial_prior = weights[category_id] / normalizer
        space_term = -math.log(spatial_prior) / log_beta + math.log(1.0 / class_count) / log_beta
    adjusted_logit = logit + class_term + space_term
    adjusted = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, adjusted_logit))))
    return adjusted * probability


def calibrate_predictions(
    predictions: Mapping[int, list[dict[str, Any]]],
    *,
    statistics: ClassSpatialStatistics,
    beta: float,
    spatial_lambda: float,
) -> dict[int, list[dict[str, Any]]]:
    """Return a copy with calibrated scalar scores."""

    calibrated: dict[int, list[dict[str, Any]]] = {}
    for image_id, records in predictions.items():
        output: list[dict[str, Any]] = []
        for record in records:
            item = dict(record)
            item["score"] = calibrate_binary_score(
                float(record["score"]),
                category_id=int(record["category_id"]),
                statistics=statistics,
                beta=beta,
                spatial_lambda=spatial_lambda,
            )
            output.append(item)
        calibrated[image_id] = output
    return calibrated


def filter_by_coarse_thresholds(
    predictions: Mapping[int, list[dict[str, Any]]],
    thresholds: Mapping[str, float],
) -> dict[int, list[dict[str, Any]]]:
    """Filter with one threshold for ship, aircraft and vehicle."""

    required = {"ship", "aircraft", "vehicle"}
    if set(thresholds) != required:
        raise ValueError(f"coarse thresholds 必须恰好包含 {sorted(required)}")
    result: dict[int, list[dict[str, Any]]] = {}
    for image_id, records in predictions.items():
        kept = [
            record
            for record in records
            if float(record["score"]) >= float(thresholds[coarse_name(int(record["category_id"]))])
        ]
        if kept:
            result[image_id] = kept
    return result


def _threshold_values(start: float, stop: float, step: float) -> tuple[float, ...]:
    if not 0.0 <= start <= stop <= 1.0 or step <= 0.0:
        raise ValueError("非法 threshold grid")
    count = int(math.floor((stop - start) / step + 1e-9))
    values = [round(start + index * step, 8) for index in range(count + 1)]
    if values[-1] < stop - 1e-12:
        values.append(float(stop))
    return tuple(values)


def _metric_payload(metrics: Any, ranking: Any) -> dict[str, Any]:
    return {
        "recall": metrics.recall,
        "fdr": metrics.fdr,
        "tp": int(metrics.details.get("tp", 0)),
        "fp": int(metrics.details.get("fp", 0)),
        "fn": int(metrics.details.get("fn", 0)),
        "official_ranking": {
            "overall_macro_recall": ranking.overall_recall,
            "overall_macro_fdr": ranking.overall_fdr,
            "per_coarse": {
                name: {
                    "macro_recall": value.macro_recall,
                    "macro_fdr": value.macro_fdr,
                    "pooled_recall": value.pooled_recall,
                    "pooled_fdr": value.pooled_fdr,
                }
                for name, value in ranking.per_coarse.items()
            },
        },
    }


def _selection_rank(
    payload: Mapping[str, Any],
    *,
    threshold: float,
    protocol: EvaluationProtocol,
) -> tuple[float, ...]:
    passed = float(
        payload["recall"] >= protocol.recall_min
        and payload["fdr"] <= protocol.fdr_max
    )
    return (
        passed,
        float(payload["recall"]),
        -float(payload["fdr"]),
        -float(payload["official_ranking"]["overall_macro_fdr"]),
        threshold,
    )


def select_coarse_thresholds(
    gt_boxes: Mapping[int, list[dict[str, Any]]],
    predictions: Mapping[int, list[dict[str, Any]]],
    *,
    protocol: EvaluationProtocol,
    threshold_start: float,
    threshold_stop: float,
    threshold_step: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Select three thresholds using only the supplied calibration folds."""

    values = _threshold_values(threshold_start, threshold_stop, threshold_step)
    curves: dict[str, list[dict[str, Any]]] = {}
    category_sets = {
        name: {
            category_id for category_id, group in protocol.category_mapping.items() if group == name
        }
        for name in protocol.class_names
    }
    for name in protocol.class_names:
        category_ids = category_sets[name]
        group_gt = {
            image_id: [item for item in records if int(item["category_id"]) in category_ids]
            for image_id, records in gt_boxes.items()
        }
        group_pred = {
            image_id: [item for item in records if int(item["category_id"]) in category_ids]
            for image_id, records in predictions.items()
        }
        curve: list[dict[str, Any]] = []
        for threshold in values:
            metrics = evaluate_workpoint(
                group_gt,
                group_pred,
                threshold=threshold,
                protocol=EvaluationProtocol(
                    contract_version=protocol.contract_version,
                    eval_version=protocol.eval_version,
                    ranking_version=protocol.ranking_version,
                    class_names=[name],
                    category_mapping={category_id: name for category_id in category_ids},
                    iou_thresholds={name: protocol.iou_thresholds[name]},
                    recall_min=protocol.recall_min,
                    fdr_max=protocol.fdr_max,
                ),
            )
            curve.append(
                {
                    "threshold": threshold,
                    "tp": int(metrics.details["tp"]),
                    "fp": int(metrics.details["fp"]),
                    "fn": int(metrics.details["fn"]),
                }
            )
        curves[name] = curve

    best_thresholds: dict[str, float] | None = None
    best_counts: tuple[int, int, int] | None = None
    best_rank: tuple[float, ...] | None = None
    names = tuple(protocol.class_names)
    for indices in itertools.product(range(len(values)), repeat=len(names)):
        selected = [curves[name][index] for name, index in zip(names, indices)]
        tp = sum(item["tp"] for item in selected)
        fp = sum(item["fp"] for item in selected)
        fn = sum(item["fn"] for item in selected)
        recall = tp / (tp + fn) if tp + fn else 1.0
        fdr = fp / (tp + fp) if tp + fp else 0.0
        passed = float(recall >= protocol.recall_min and fdr <= protocol.fdr_max)
        thresholds = {name: values[index] for name, index in zip(names, indices)}
        rank = (passed, recall, -fdr, sum(thresholds.values()))
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_thresholds = thresholds
            best_counts = (tp, fp, fn)
    assert best_thresholds is not None and best_counts is not None
    selected_predictions = filter_by_coarse_thresholds(predictions, best_thresholds)
    metrics = evaluate_workpoint(gt_boxes, selected_predictions, threshold=0.0, protocol=protocol)
    ranking = evaluate_ranking_metrics(
        dict(gt_boxes),
        selected_predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=True,
    )
    return best_thresholds, _metric_payload(metrics, ranking)


def _select_adjusted_global(
    gt_boxes: Mapping[int, list[dict[str, Any]]],
    predictions: Mapping[int, list[dict[str, Any]]],
    *,
    statistics: ClassSpatialStatistics,
    protocol: EvaluationProtocol,
    betas: Sequence[float],
    lambdas: Sequence[float],
    threshold_start: float,
    threshold_stop: float,
    threshold_step: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    best_parameters: dict[str, float] | None = None
    best_payload: dict[str, Any] | None = None
    best_rank: tuple[float, ...] | None = None
    for beta, spatial_lambda in itertools.product(betas, lambdas):
        adjusted = calibrate_predictions(
            predictions,
            statistics=statistics,
            beta=float(beta),
            spatial_lambda=float(spatial_lambda),
        )
        threshold, metrics, _ = scan_global_threshold(
            gt_boxes,
            adjusted,
            protocol=protocol,
            threshold_start=threshold_start,
            threshold_stop=threshold_stop,
            threshold_step=threshold_step,
            internal_recall_min=protocol.recall_min,
            internal_fdr_max=protocol.fdr_max,
        )
        ranking = evaluate_ranking_workpoint(
            gt_boxes,
            adjusted,
            threshold=threshold,
            protocol=protocol,
            require_complete_taxonomy=True,
        )
        payload = _metric_payload(metrics, ranking)
        rank = _selection_rank(payload, threshold=threshold, protocol=protocol)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_parameters = {
                "beta": float(beta),
                "spatial_lambda": float(spatial_lambda),
                "threshold": float(threshold),
            }
            best_payload = payload
    assert best_parameters is not None and best_payload is not None
    return best_parameters, best_payload


def run_yolo_calibration_crossfit(
    *,
    aggregate_dir: str | Path,
    formal_crop_manifest_path: str | Path,
    protocol: EvaluationProtocol,
    expected_images: int = 4481,
    expected_annotations: int = 20933,
    candidate_floor: float = 0.001,
    threshold_start: float = 0.001,
    threshold_stop: float = 0.301,
    threshold_step: float = 0.01,
    betas: Sequence[float] = (2.0, 3.0, 5.0, 10.0),
    spatial_lambdas: Sequence[float] = (0.25, 0.5, 1.0, 2.0),
    max_grid_size: int = 64,
) -> dict[str, Any]:
    """Run C0-C3 with strict outer-fold separation."""

    metadata, predictions, image_folds = load_cv3_aggregate(
        aggregate_dir,
        candidate_floor=candidate_floor,
    )
    gt_boxes = load_gt_from_formal_crop_manifest(
        formal_crop_manifest_path,
        expected_images=expected_images,
        expected_annotations=expected_annotations,
    )
    annotations = load_spatial_annotations(
        formal_crop_manifest_path,
        expected_annotations=expected_annotations,
    )
    if set(gt_boxes) != set(predictions) or set(gt_boxes) != set(image_folds):
        raise ValueError("GT、OOF predictions 与 fold 映射的 image_id 集合不一致")
    fold_set = set(image_folds.values())
    if fold_set != {0, 1, 2}:
        raise ValueError(f"正式 cross-fit 需要 folds={{0,1,2}}，实际 {sorted(fold_set)}")
    folded_gt = split_gt_by_fold(gt_boxes, image_folds)
    folded_pred = split_by_fold(predictions, image_folds)
    methods = ("C0_global", "C1_coarse", "C2_prior", "C3_fractal_proxy")
    per_fold: list[dict[str, Any]] = []
    merged_gt: dict[int, list[dict[str, Any]]] = {}
    merged_predictions: dict[str, dict[int, list[dict[str, Any]]]] = {
        method: {} for method in methods
    }

    for held_out in sorted(fold_set):
        selection_folds = sorted(fold_set - {held_out})
        selection_gt = _merge_folds(folded_gt, selection_folds)
        selection_pred = _merge_folds(folded_pred, selection_folds)
        heldout_gt = _merge_folds(folded_gt, [held_out])
        heldout_pred = _merge_folds(folded_pred, [held_out])
        merged_gt.update(heldout_gt)
        statistics = estimate_class_spatial_statistics(
            annotations,
            source_folds=selection_folds,
            max_grid_size=max_grid_size,
        )
        fold_result: dict[str, Any] = {
            "held_out_fold": held_out,
            "selection_folds": selection_folds,
            "statistics": {
                "counts": list(statistics.counts),
                "priors": list(statistics.priors),
                "fractal_dimensions": list(statistics.fractal_dimensions),
            },
            "methods": {},
        }

        c0_threshold, c0_selection_metrics, _ = scan_global_threshold(
            selection_gt,
            selection_pred,
            protocol=protocol,
            threshold_start=threshold_start,
            threshold_stop=threshold_stop,
            threshold_step=threshold_step,
            internal_recall_min=protocol.recall_min,
            internal_fdr_max=protocol.fdr_max,
        )
        c0_selected = _filter_by_score(heldout_pred, c0_threshold)
        c0_hold = evaluate_workpoint(heldout_gt, c0_selected, threshold=0.0, protocol=protocol)
        c0_rank = evaluate_ranking_metrics(
            heldout_gt,
            c0_selected,
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
            require_complete_taxonomy=True,
        )
        fold_result["methods"]["C0_global"] = {
            "selected": {"threshold": c0_threshold},
            "selection": {
                "recall": c0_selection_metrics.recall,
                "fdr": c0_selection_metrics.fdr,
            },
            "held_out": _metric_payload(c0_hold, c0_rank),
        }
        merged_predictions["C0_global"].update(c0_selected)

        coarse_thresholds, c1_selection = select_coarse_thresholds(
            selection_gt,
            selection_pred,
            protocol=protocol,
            threshold_start=threshold_start,
            threshold_stop=threshold_stop,
            threshold_step=threshold_step,
        )
        c1_selected = filter_by_coarse_thresholds(heldout_pred, coarse_thresholds)
        c1_hold = evaluate_workpoint(heldout_gt, c1_selected, threshold=0.0, protocol=protocol)
        c1_rank = evaluate_ranking_metrics(
            heldout_gt,
            c1_selected,
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
            require_complete_taxonomy=True,
        )
        fold_result["methods"]["C1_coarse"] = {
            "selected": {"thresholds": coarse_thresholds},
            "selection": c1_selection,
            "held_out": _metric_payload(c1_hold, c1_rank),
        }
        merged_predictions["C1_coarse"].update(c1_selected)

        for method, lambdas in (
            ("C2_prior", (0.0,)),
            ("C3_fractal_proxy", tuple(float(value) for value in spatial_lambdas)),
        ):
            parameters, selection_payload = _select_adjusted_global(
                selection_gt,
                selection_pred,
                statistics=statistics,
                protocol=protocol,
                betas=betas,
                lambdas=lambdas,
                threshold_start=threshold_start,
                threshold_stop=threshold_stop,
                threshold_step=threshold_step,
            )
            adjusted_hold = calibrate_predictions(
                heldout_pred,
                statistics=statistics,
                beta=parameters["beta"],
                spatial_lambda=parameters["spatial_lambda"],
            )
            selected = _filter_by_score(adjusted_hold, parameters["threshold"])
            hold_metrics = evaluate_workpoint(
                heldout_gt, selected, threshold=0.0, protocol=protocol
            )
            hold_ranking = evaluate_ranking_metrics(
                heldout_gt,
                selected,
                class_names=protocol.class_names,
                category_mapping=protocol.category_mapping,
                iou_thresholds=protocol.iou_thresholds,
                require_complete_taxonomy=True,
            )
            fold_result["methods"][method] = {
                "selected": parameters,
                "selection": selection_payload,
                "held_out": _metric_payload(hold_metrics, hold_ranking),
            }
            merged_predictions[method].update(selected)
        per_fold.append(fold_result)

    merged: dict[str, Any] = {}
    for method in methods:
        metrics = evaluate_workpoint(
            merged_gt,
            merged_predictions[method],
            threshold=0.0,
            protocol=protocol,
        )
        ranking = evaluate_ranking_metrics(
            merged_gt,
            merged_predictions[method],
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
            require_complete_taxonomy=True,
        )
        payload = _metric_payload(metrics, ranking)
        payload["official_gate_passed"] = (
            metrics.recall >= protocol.recall_min and metrics.fdr <= protocol.fdr_max
        )
        payload["fold_direction_vs_c0"] = []
        if method != "C0_global":
            for fold_result in per_fold:
                baseline = fold_result["methods"]["C0_global"]["held_out"]
                candidate = fold_result["methods"][method]["held_out"]
                payload["fold_direction_vs_c0"].append(
                    {
                        "fold": fold_result["held_out_fold"],
                        "recall_delta": candidate["recall"] - baseline["recall"],
                        "fdr_delta": candidate["fdr"] - baseline["fdr"],
                    }
                )
        merged[method] = payload

    baseline = merged["C0_global"]
    for method, payload in merged.items():
        payload["delta_vs_c0"] = {
            "recall": payload["recall"] - baseline["recall"],
            "fdr": payload["fdr"] - baseline["fdr"],
            "overall_macro_recall": (
                payload["official_ranking"]["overall_macro_recall"]
                - baseline["official_ranking"]["overall_macro_recall"]
            ),
            "overall_macro_fdr": (
                payload["official_ranking"]["overall_macro_fdr"]
                - baseline["official_ranking"]["overall_macro_fdr"]
            ),
        }

    return {
        "contract_version": CALIBRATION_CONTRACT_VERSION,
        "scientific_scope": {
            "C0_global": "exact_post_nms_threshold_baseline",
            "C1_coarse": "exact_post_nms_coarse_thresholds",
            "C2_prior": "post_nms_scalar_screen_not_full_logit_calibration",
            "C3_fractal_proxy": "post_nms_scalar_screen_not_exact_fracal",
        },
        "source": {
            "model_key": metadata.get("model_key"),
            "seed": metadata.get("seed"),
            "candidate_floor": candidate_floor,
            "image_count": len(gt_boxes),
            "annotation_count": sum(len(items) for items in gt_boxes.values()),
            "cv3_manifest_sha256": metadata.get("source_manifest_sha256"),
        },
        "grid": {
            "threshold_start": threshold_start,
            "threshold_stop": threshold_stop,
            "threshold_step": threshold_step,
            "betas": [float(value) for value in betas],
            "spatial_lambdas": [float(value) for value in spatial_lambdas],
            "max_grid_size": max_grid_size,
        },
        "per_fold": per_fold,
        "merged_held_out": merged,
    }


__all__ = [
    "CALIBRATION_CONTRACT_VERSION",
    "ClassSpatialStatistics",
    "SpatialAnnotation",
    "calibrate_binary_score",
    "calibrate_predictions",
    "estimate_class_spatial_statistics",
    "estimate_fractal_dimension",
    "filter_by_coarse_thresholds",
    "load_spatial_annotations",
    "run_yolo_calibration_crossfit",
    "select_coarse_thresholds",
]
