#!/usr/bin/env python3
"""Audit fold-safe D-FINE support for below-threshold Y5 proposals.

The experiment never imports a specialist box.  For one coarse class it first
selects a primary-detector threshold on the two training folds, then selects a
threshold over ``primary_score * specialist_support`` subject to a marginal
FDR constraint.  Only below-primary-threshold proposals are eligible for
recovery, and all held-out counts use the repository official matcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.cross_detector_agreement import (
    best_same_fine_support,
    marginal_false_detection_rate,
)
from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_predictions
from rsdet.evaluation.protocol import EvaluationProtocol, parse_evaluation_protocol
from rsdet.postprocess.calibration import build_threshold_grid
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coarse_mapping(protocol: EvaluationProtocol, coarse: str) -> dict[int, str]:
    result = {
        category_id: name
        for category_id, name in protocol.category_mapping.items()
        if name == coarse
    }
    if not result:
        raise ValueError(f"coarse class has no fine categories: {coarse}")
    return result


def _filter_coarse(
    rows: dict[int, list[dict[str, Any]]],
    *,
    mapping: dict[int, str],
    threshold: float | None = None,
    image_ids: set[int] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    scope = set(rows) if image_ids is None else image_ids
    return {
        image_id: [
            item
            for item in rows.get(image_id, ())
            if int(item["category_id"]) in mapping
            and (threshold is None or float(item.get("score", 1.0)) >= threshold)
        ]
        for image_id in sorted(scope)
    }


def _metrics(
    gt: dict[int, list[dict[str, Any]]],
    pred: dict[int, list[dict[str, Any]]],
    *,
    coarse: str,
    mapping: dict[int, str],
    iou_threshold: float,
) -> dict[str, Any]:
    value = evaluate_predictions(
        gt,
        pred,
        class_names=[coarse],
        category_mapping=mapping,
        iou_thresholds={coarse: iou_threshold},
    )
    item = value.per_class[coarse]
    return {
        "recall": value.recall,
        "fdr": value.fdr,
        "tp": item.tp,
        "fp": item.fp,
        "fn": item.fn,
    }


def _select_primary_threshold(
    gt: dict[int, list[dict[str, Any]]],
    pred: dict[int, list[dict[str, Any]]],
    *,
    coarse: str,
    mapping: dict[int, str],
    iou_threshold: float,
    thresholds: list[float],
    target_fdr: float,
) -> dict[str, Any]:
    points: list[tuple[float, dict[str, Any]]] = []
    for threshold in thresholds:
        metrics = _metrics(
            gt,
            _filter_coarse(pred, mapping=mapping, threshold=threshold),
            coarse=coarse,
            mapping=mapping,
            iou_threshold=iou_threshold,
        )
        points.append((threshold, metrics))
    feasible = [point for point in points if point[1]["fdr"] <= target_fdr]
    if feasible:
        threshold, metrics = max(
            feasible,
            key=lambda point: (
                point[1]["recall"],
                -point[1]["fdr"],
                point[0],
            ),
        )
    else:
        threshold, metrics = min(
            points,
            key=lambda point: (point[1]["fdr"], -point[1]["recall"], -point[0]),
        )
    return {"threshold": threshold, "metrics": metrics}


def _annotate_support(
    primary: dict[int, list[dict[str, Any]]],
    specialist: dict[int, list[dict[str, Any]]],
    *,
    image_ids: set[int],
    mapping: dict[int, str],
    iou_threshold: float,
) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for image_id in sorted(image_ids):
        primary_rows = [
            dict(item)
            for item in primary.get(image_id, ())
            if int(item["category_id"]) in mapping
        ]
        specialist_rows = [
            item
            for item in specialist.get(image_id, ())
            if int(item["category_id"]) in mapping
        ]
        support = best_same_fine_support(
            [dict(item, image_id=image_id) for item in primary_rows],
            [dict(item, image_id=image_id) for item in specialist_rows],
            iou_threshold=iou_threshold,
        )
        for item, evidence in zip(primary_rows, support, strict=True):
            item.update(evidence)
        result[image_id] = primary_rows
    return result


def _combined(
    supported: dict[int, list[dict[str, Any]]],
    *,
    image_ids: set[int],
    primary_threshold: float,
    agreement_threshold: float | None,
) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for image_id in sorted(image_ids):
        selected: list[dict[str, Any]] = []
        for item in supported.get(image_id, ()):
            primary_active = float(item["score"]) >= primary_threshold
            recovered = (
                agreement_threshold is not None
                and not primary_active
                and float(item["agreement_product"]) >= agreement_threshold
            )
            if primary_active or recovered:
                selected.append(item)
        result[image_id] = selected
    return result


def _product_filtered(
    supported: dict[int, list[dict[str, Any]]],
    *,
    image_ids: set[int],
    threshold: float,
) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [
            dict(item, score=float(item["agreement_product"]))
            for item in supported.get(image_id, ())
            if float(item["agreement_product"]) >= threshold
        ]
        for image_id in sorted(image_ids)
    }


def _select_product_threshold(
    gt: dict[int, list[dict[str, Any]]],
    supported: dict[int, list[dict[str, Any]]],
    *,
    image_ids: set[int],
    thresholds: list[float],
    target_fdr: float,
    coarse: str,
    mapping: dict[int, str],
    iou_threshold: float,
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = _metrics(
            gt,
            _product_filtered(
                supported,
                image_ids=image_ids,
                threshold=threshold,
            ),
            coarse=coarse,
            mapping=mapping,
            iou_threshold=iou_threshold,
        )
        points.append({"threshold": threshold, "metrics": metrics})
    feasible = [row for row in points if row["metrics"]["fdr"] <= target_fdr]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                row["metrics"]["recall"],
                -row["metrics"]["fdr"],
                row["threshold"],
            ),
        )
        selected["status"] = "selected_at_total_fdr"
        return selected
    selected = min(
        points,
        key=lambda row: (
            row["metrics"]["fdr"],
            -row["metrics"]["recall"],
            -row["threshold"],
        ),
    )
    selected["status"] = "no_feasible_total_fdr_used_minimum_fdr"
    return selected


def _select_agreement_threshold(
    gt: dict[int, list[dict[str, Any]]],
    supported: dict[int, list[dict[str, Any]]],
    *,
    image_ids: set[int],
    primary_threshold: float,
    agreement_thresholds: list[float],
    target_marginal_fdr: float,
    coarse: str,
    mapping: dict[int, str],
    iou_threshold: float,
) -> dict[str, Any]:
    baseline = _metrics(
        gt,
        _combined(
            supported,
            image_ids=image_ids,
            primary_threshold=primary_threshold,
            agreement_threshold=None,
        ),
        coarse=coarse,
        mapping=mapping,
        iou_threshold=iou_threshold,
    )
    feasible: list[dict[str, Any]] = []
    for threshold in agreement_thresholds:
        candidate = _metrics(
            gt,
            _combined(
                supported,
                image_ids=image_ids,
                primary_threshold=primary_threshold,
                agreement_threshold=threshold,
            ),
            coarse=coarse,
            mapping=mapping,
            iou_threshold=iou_threshold,
        )
        delta_tp = candidate["tp"] - baseline["tp"]
        delta_fp = candidate["fp"] - baseline["fp"]
        if delta_tp < 0 or delta_fp < 0:
            raise RuntimeError("adding below-threshold proposals reduced TP/FP counts")
        row = {
            "threshold": threshold,
            "baseline": baseline,
            "candidate": candidate,
            "delta_tp": delta_tp,
            "delta_fp": delta_fp,
            "marginal_fdr": marginal_false_detection_rate(delta_tp, delta_fp),
        }
        if delta_tp > 0 and row["marginal_fdr"] <= target_marginal_fdr:
            feasible.append(row)
    if not feasible:
        return {
            "threshold": None,
            "baseline": baseline,
            "candidate": baseline,
            "delta_tp": 0,
            "delta_fp": 0,
            "marginal_fdr": 0.0,
            "status": "no_feasible_recovery",
        }
    result = max(
        feasible,
        key=lambda row: (
            row["delta_tp"],
            -row["delta_fp"],
            row["threshold"],
        ),
    )
    result["status"] = "selected"
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-name", default="y5_predictions.json")
    parser.add_argument("--specialist-name", default="dfine_predictions.json")
    parser.add_argument(
        "--evidence-cache",
        type=Path,
        help="Aligned candidate cache used instead of primary/specialist JSON files.",
    )
    parser.add_argument(
        "--evidence-scores",
        type=Path,
        help="NPZ containing candidate_index/score/fold evidence for cache rows.",
    )
    parser.add_argument("--coarse", default="vehicle")
    parser.add_argument(
        "--route",
        choices=("recovery", "rerank"),
        default="recovery",
        help=(
            "Recover below-threshold Y5 boxes, or select a product-score threshold "
            "at each requested total FDR level."
        ),
    )
    parser.add_argument("--primary-target-fdr", type=float, default=0.15)
    parser.add_argument(
        "--fixed-primary-threshold",
        type=float,
        help="Use a frozen deployed primary threshold instead of fitting FDR.",
    )
    parser.add_argument(
        "--marginal-fdr-levels", type=float, nargs="+", default=(0.15, 0.20)
    )
    parser.add_argument("--primary-step", type=float, default=0.005)
    parser.add_argument("--agreement-step", type=float, default=0.001)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    return parser.parse_args()


def _load_external_evidence(
    cache_path: Path,
    score_path: Path,
    *,
    mapping: dict[int, str],
) -> dict[int, dict[int, list[dict[str, Any]]]]:
    with np.load(cache_path, allow_pickle=False) as payload:
        required = (
            "image_id",
            "category_id",
            "bbox_xyxy",
            "detector_score",
            "fold",
            "candidate_index",
        )
        arrays = {name: np.asarray(payload[name]) for name in required}
    row_count = len(arrays["image_id"])
    if any(len(value) != row_count for value in arrays.values()):
        raise ValueError("external evidence cache arrays are not aligned")
    if not np.array_equal(
        arrays["candidate_index"].astype(np.int64),
        np.arange(row_count, dtype=np.int64),
    ):
        raise ValueError("external evidence cache candidate_index is not canonical")
    with np.load(score_path, allow_pickle=False) as payload:
        indices = np.asarray(payload["candidate_index"], dtype=np.int64)
        scores = np.asarray(payload["score"], dtype=np.float64)
        score_folds = np.asarray(payload["fold"], dtype=np.int64)
    if not (len(indices) == len(scores) == len(score_folds)):
        raise ValueError("external evidence score arrays are not aligned")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("external evidence candidate_index contains duplicates")
    if np.any((indices < 0) | (indices >= row_count)):
        raise ValueError("external evidence candidate_index is out of bounds")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("external evidence scores must be finite and in [0, 1]")
    if not np.array_equal(arrays["fold"][indices].astype(np.int64), score_folds):
        raise ValueError("external evidence score folds do not match cache folds")
    expected = np.isin(
        arrays["category_id"].astype(np.int64),
        np.asarray(sorted(mapping), dtype=np.int64),
    )
    if not np.array_equal(np.sort(indices), np.flatnonzero(expected)):
        raise ValueError("external evidence scores do not exactly cover the coarse rows")

    full_scores = np.full(row_count, np.nan, dtype=np.float64)
    full_scores[indices] = scores
    by_fold: dict[int, dict[int, list[dict[str, Any]]]] = {
        0: {},
        1: {},
        2: {},
    }
    for index in indices:
        fold = int(arrays["fold"][index])
        if fold not in by_fold:
            raise ValueError(f"unexpected external evidence fold={fold}")
        image_id = int(arrays["image_id"][index])
        detector_score = float(arrays["detector_score"][index])
        if not 0.0 <= detector_score <= 1.0:
            raise ValueError("external detector score is outside [0, 1]")
        row = {
            "bbox_xyxy": [float(value) for value in arrays["bbox_xyxy"][index]],
            "category_id": int(arrays["category_id"][index]),
            "score": detector_score,
            "agreement_product": float(full_scores[index]),
        }
        by_fold[fold].setdefault(image_id, []).append(row)
    return by_fold


def main() -> int:
    args = _parse_args()
    if (args.evidence_cache is None) != (args.evidence_scores is None):
        raise ValueError("evidence-cache and evidence-scores must be provided together")
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    if args.coarse not in protocol.class_names:
        raise ValueError(f"unknown coarse class={args.coarse}")
    for value in (args.primary_target_fdr, *args.marginal_fdr_levels):
        if not 0.0 <= value <= 1.0:
            raise ValueError("FDR levels must be in [0, 1]")
    if args.fixed_primary_threshold is not None and not (
        0.0 <= args.fixed_primary_threshold <= 1.0
    ):
        raise ValueError("fixed-primary-threshold must be in [0, 1]")
    mapping = _coarse_mapping(protocol, args.coarse)
    iou_threshold = float(protocol.iou_thresholds[args.coarse])
    primary_grid = build_threshold_grid(0.001, 0.996, args.primary_step)
    agreement_grid = build_threshold_grid(0.001, 1.0, args.agreement_step)
    external = None
    if args.evidence_cache is not None and args.evidence_scores is not None:
        external = _load_external_evidence(
            args.evidence_cache,
            args.evidence_scores,
            mapping=mapping,
        )

    folds: dict[int, dict[str, Any]] = {}
    input_sha: dict[str, dict[str, str]] = {}
    seen_images: set[int] = set()
    for fold in (0, 1, 2):
        folder = args.fold_root / f"fold_{fold}"
        paths = {
            "gt": folder / "instances_val.json",
        }
        if external is None:
            paths.update(
                {
                    "primary": folder / args.primary_name,
                    "specialist": folder / args.specialist_name,
                }
            )
        raw_gt = json.loads(paths["gt"].read_text(encoding="utf-8"))
        image_ids = {int(item["id"]) for item in raw_gt["images"]}
        if seen_images & image_ids:
            raise ValueError("image IDs overlap across folds")
        seen_images |= image_ids
        gt = load_coco_ground_truth(paths["gt"])
        if external is None:
            primary = load_coco_predictions(paths["primary"])
            specialist = load_coco_predictions(paths["specialist"])
            supported = _annotate_support(
                primary,
                specialist,
                image_ids=image_ids,
                mapping=mapping,
                iou_threshold=iou_threshold,
            )
        else:
            unexpected = set(external[fold]) - image_ids
            if unexpected:
                raise ValueError(
                    f"external evidence fold={fold} contains out-of-fold image IDs"
                )
            supported = {
                image_id: list(external[fold].get(image_id, ()))
                for image_id in sorted(image_ids)
            }
            primary = supported
        folds[fold] = {
            "image_ids": image_ids,
            "gt": _filter_coarse(gt, mapping=mapping, image_ids=image_ids),
            "primary": primary,
            "supported": supported,
        }
        input_sha[str(fold)] = {name: _sha256(path) for name, path in paths.items()}

    if args.evidence_cache is not None and args.evidence_scores is not None:
        input_sha["external_evidence"] = {
            "cache": _sha256(args.evidence_cache),
            "scores": _sha256(args.evidence_scores),
        }

    output: dict[str, Any] = {
        "status": "complete_diagnostic_only",
        "protocol": (
            f"cv3_nested_{args.route}_external_evidence_v2"
            if external is not None
            else f"cv3_nested_same_fine_product_{args.route}_v2"
        ),
        "warning": (
            "D-FINE supplies offline evidence only. This audit does not admit a dual-detector "
            "deployment or a student model."
        ),
        "coarse": args.coarse,
        "primary_target_fdr": args.primary_target_fdr,
        "marginal_fdr_levels": list(args.marginal_fdr_levels),
        "input_sha256": input_sha,
        "levels": {},
    }
    for marginal_level in args.marginal_fdr_levels:
        pooled_gt: dict[int, list[dict[str, Any]]] = {}
        pooled_baseline: dict[int, list[dict[str, Any]]] = {}
        pooled_candidate: dict[int, list[dict[str, Any]]] = {}
        heldout_payload: dict[str, Any] = {}
        for held_out in (0, 1, 2):
            train_folds = [fold for fold in (0, 1, 2) if fold != held_out]
            train_images = set().union(*(folds[fold]["image_ids"] for fold in train_folds))
            train_gt: dict[int, list[dict[str, Any]]] = {}
            train_primary: dict[int, list[dict[str, Any]]] = {}
            train_supported: dict[int, list[dict[str, Any]]] = {}
            for fold in train_folds:
                train_gt.update(folds[fold]["gt"])
                train_primary.update(folds[fold]["primary"])
                train_supported.update(folds[fold]["supported"])
            if args.fixed_primary_threshold is None:
                selected_primary = _select_primary_threshold(
                    train_gt,
                    _filter_coarse(
                        train_primary,
                        mapping=mapping,
                        image_ids=train_images,
                    ),
                    coarse=args.coarse,
                    mapping=mapping,
                    iou_threshold=iou_threshold,
                    thresholds=primary_grid,
                    target_fdr=args.primary_target_fdr,
                )
            else:
                frozen_threshold = float(args.fixed_primary_threshold)
                selected_primary = {
                    "threshold": frozen_threshold,
                    "metrics": _metrics(
                        train_gt,
                        _filter_coarse(
                            train_primary,
                            mapping=mapping,
                            threshold=frozen_threshold,
                            image_ids=train_images,
                        ),
                        coarse=args.coarse,
                        mapping=mapping,
                        iou_threshold=iou_threshold,
                    ),
                    "status": "fixed_deployment_threshold",
                }
            if args.route == "recovery":
                selected_agreement = _select_agreement_threshold(
                    train_gt,
                    train_supported,
                    image_ids=train_images,
                    primary_threshold=float(selected_primary["threshold"]),
                    agreement_thresholds=agreement_grid,
                    target_marginal_fdr=float(marginal_level),
                    coarse=args.coarse,
                    mapping=mapping,
                    iou_threshold=iou_threshold,
                )
            else:
                selected_agreement = _select_product_threshold(
                    train_gt,
                    train_supported,
                    image_ids=train_images,
                    thresholds=agreement_grid,
                    target_fdr=float(marginal_level),
                    coarse=args.coarse,
                    mapping=mapping,
                    iou_threshold=iou_threshold,
                )

            heldout_images = folds[held_out]["image_ids"]
            heldout_gt = folds[held_out]["gt"]
            baseline_pred = _combined(
                folds[held_out]["supported"],
                image_ids=heldout_images,
                primary_threshold=float(selected_primary["threshold"]),
                agreement_threshold=None,
            )
            if args.route == "recovery":
                candidate_pred = _combined(
                    folds[held_out]["supported"],
                    image_ids=heldout_images,
                    primary_threshold=float(selected_primary["threshold"]),
                    agreement_threshold=selected_agreement["threshold"],
                )
            else:
                candidate_pred = _product_filtered(
                    folds[held_out]["supported"],
                    image_ids=heldout_images,
                    threshold=float(selected_agreement["threshold"]),
                )
            baseline_metrics = _metrics(
                heldout_gt,
                baseline_pred,
                coarse=args.coarse,
                mapping=mapping,
                iou_threshold=iou_threshold,
            )
            candidate_metrics = _metrics(
                heldout_gt,
                candidate_pred,
                coarse=args.coarse,
                mapping=mapping,
                iou_threshold=iou_threshold,
            )
            delta_tp = candidate_metrics["tp"] - baseline_metrics["tp"]
            delta_fp = candidate_metrics["fp"] - baseline_metrics["fp"]
            heldout_payload[str(held_out)] = {
                "primary_selection": selected_primary,
                "agreement_selection_on_train": selected_agreement,
                "heldout_baseline": baseline_metrics,
                "heldout_candidate": candidate_metrics,
                "heldout_delta": {
                    "tp": delta_tp,
                    "fp": delta_fp,
                    "recall": candidate_metrics["recall"] - baseline_metrics["recall"],
                    "fdr": candidate_metrics["fdr"] - baseline_metrics["fdr"],
                    "marginal_fdr": (
                        marginal_false_detection_rate(delta_tp, delta_fp)
                        if args.route == "recovery"
                        else None
                    ),
                },
            }
            pooled_gt.update(heldout_gt)
            pooled_baseline.update(baseline_pred)
            pooled_candidate.update(candidate_pred)

        baseline = _metrics(
            pooled_gt,
            pooled_baseline,
            coarse=args.coarse,
            mapping=mapping,
            iou_threshold=iou_threshold,
        )
        candidate = _metrics(
            pooled_gt,
            pooled_candidate,
            coarse=args.coarse,
            mapping=mapping,
            iou_threshold=iou_threshold,
        )
        delta_tp = candidate["tp"] - baseline["tp"]
        delta_fp = candidate["fp"] - baseline["fp"]
        output["levels"][f"{marginal_level:.3f}"] = {
            "heldout": heldout_payload,
            "pooled_baseline": baseline,
            "pooled_candidate": candidate,
            "pooled_delta": {
                "tp": delta_tp,
                "fp": delta_fp,
                "recall": candidate["recall"] - baseline["recall"],
                "fdr": candidate["fdr"] - baseline["fdr"],
                "marginal_fdr": (
                    marginal_false_detection_rate(delta_tp, delta_fp)
                    if args.route == "recovery"
                    else None
                ),
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
