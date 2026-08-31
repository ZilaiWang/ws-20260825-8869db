#!/usr/bin/env python3
"""Cross-fit a coarse-class route between a primary and specialist detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_predictions, evaluate_ranking_metrics
from rsdet.evaluation.protocol import EvaluationProtocol, parse_evaluation_protocol
from rsdet.postprocess.calibration import build_threshold_grid
from rsdet.utils.config import load_config

COARSE = ("ship", "aircraft", "vehicle")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coarse_rows(
    values: dict[int, list[dict[str, Any]]],
    *,
    image_ids: set[int],
    coarse: str,
    mapping: dict[int, str],
    threshold: float | None = None,
) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [
            item
            for item in values.get(image_id, [])
            if mapping[int(item["category_id"])] == coarse
            and (threshold is None or float(item["score"]) >= threshold)
        ]
        for image_id in sorted(image_ids)
    }


def _sweep(
    gt: dict[int, list[dict[str, Any]]],
    pred: dict[int, list[dict[str, Any]]],
    *,
    coarse: str,
    thresholds: list[float],
    protocol: EvaluationProtocol,
) -> list[tuple[float, Any]]:
    mapping = {
        category_id: name
        for category_id, name in protocol.category_mapping.items()
        if name == coarse
    }
    points: list[tuple[float, Any]] = []
    for threshold in thresholds:
        filtered = {
            image_id: [item for item in items if float(item["score"]) >= threshold]
            for image_id, items in pred.items()
        }
        metrics = evaluate_predictions(
            gt,
            filtered,
            class_names=[coarse],
            category_mapping=mapping,
            iou_thresholds={coarse: protocol.iou_thresholds[coarse]},
        )
        points.append((threshold, metrics))
    return points


def select_threshold(points: list[tuple[float, Any]], target_fdr: float) -> dict[str, Any]:
    """Maximize recall subject to the frozen training-fold FDR constraint."""
    feasible = [point for point in points if point[1].fdr <= target_fdr]
    if feasible:
        threshold, metrics = max(
            feasible,
            key=lambda point: (point[1].recall, -point[1].fdr, point[0]),
        )
    else:
        threshold, metrics = min(
            points,
            key=lambda point: (point[1].fdr, -point[1].recall, -point[0]),
        )
    return {
        "threshold": float(threshold),
        "train_recall": float(metrics.recall),
        "train_fdr": float(metrics.fdr),
        "train_tp": int(metrics.details["tp"]),
        "train_fp": int(metrics.details["fp"]),
        "train_fn": int(metrics.details["fn"]),
    }


def _payload(
    gt: dict[int, list[dict[str, Any]]],
    pred: dict[int, list[dict[str, Any]]],
    protocol: EvaluationProtocol,
) -> dict[str, Any]:
    metrics = evaluate_predictions(
        gt,
        pred,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    ranking = evaluate_ranking_metrics(
        gt,
        pred,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=True,
    )
    return {
        "recall": metrics.recall,
        "fdr": metrics.fdr,
        "tp": metrics.details["tp"],
        "fp": metrics.details["fp"],
        "fn": metrics.details["fn"],
        "macro_recall": ranking.overall_recall,
        "macro_fdr": ranking.overall_fdr,
        "per_coarse": {
            name: {
                "recall": item.recall,
                "fdr": item.fdr,
                "tp": item.tp,
                "fp": item.fp,
                "fn": item.fn,
            }
            for name, item in metrics.per_class.items()
        },
    }


def _delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "recall": candidate["recall"] - baseline["recall"],
        "fdr": candidate["fdr"] - baseline["fdr"],
        "macro_recall": candidate["macro_recall"] - baseline["macro_recall"],
        "macro_fdr": candidate["macro_fdr"] - baseline["macro_fdr"],
        "per_coarse": {
            name: {
                "recall": candidate["per_coarse"][name]["recall"]
                - baseline["per_coarse"][name]["recall"],
                "fdr": candidate["per_coarse"][name]["fdr"]
                - baseline["per_coarse"][name]["fdr"],
            }
            for name in baseline["per_coarse"]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--primary-name", default="y5_predictions.json")
    parser.add_argument("--specialist-name", default="dfine_predictions.json")
    parser.add_argument("--specialist-coarse", default="vehicle")
    parser.add_argument("--threshold-start", type=float, default=0.001)
    parser.add_argument("--threshold-stop", type=float, default=0.996)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument(
        "--fdr-levels", type=float, nargs="+", default=(0.10, 0.12, 0.15, 0.17, 0.20)
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    if args.specialist_coarse not in COARSE:
        raise ValueError(f"specialist coarse must be one of {COARSE}")
    thresholds = build_threshold_grid(
        args.threshold_start, args.threshold_stop, args.threshold_step
    )
    folds: dict[int, dict[str, Any]] = {}
    input_sha: dict[str, dict[str, str]] = {}
    seen_images: set[int] = set()
    for fold in (0, 1, 2):
        folder = args.fold_root / f"fold_{fold}"
        paths = {
            "gt": folder / "instances_val.json",
            "primary": folder / args.primary_name,
            "specialist": folder / args.specialist_name,
        }
        raw_gt = json.loads(paths["gt"].read_text(encoding="utf-8"))
        image_ids = {int(item["id"]) for item in raw_gt["images"]}
        overlap = seen_images & image_ids
        if overlap:
            raise ValueError(f"image ids overlap across folds: {sorted(overlap)[:5]}")
        seen_images |= image_ids
        folds[fold] = {
            "image_ids": image_ids,
            "gt": load_coco_ground_truth(paths["gt"]),
            "primary": load_coco_predictions(paths["primary"]),
            "specialist": load_coco_predictions(paths["specialist"]),
        }
        input_sha[str(fold)] = {name: _sha256(path) for name, path in paths.items()}

    all_gt: dict[int, list[dict[str, Any]]] = {}
    for fold in folds.values():
        all_gt.update(fold["gt"])

    primary_points: dict[tuple[int, str], list[tuple[float, Any]]] = {}
    specialist_points: dict[int, list[tuple[float, Any]]] = {}
    for held_out in (0, 1, 2):
        train_images = set().union(
            *(folds[fold]["image_ids"] for fold in (0, 1, 2) if fold != held_out)
        )
        train_gt = {
            image_id: all_gt[image_id]
            for image_id in train_images
        }
        train_primary: dict[int, list[dict[str, Any]]] = {}
        train_specialist: dict[int, list[dict[str, Any]]] = {}
        for fold in (0, 1, 2):
            if fold != held_out:
                train_primary.update(folds[fold]["primary"])
                train_specialist.update(folds[fold]["specialist"])
        for coarse in COARSE:
            primary_points[(held_out, coarse)] = _sweep(
                _coarse_rows(
                    train_gt,
                    image_ids=train_images,
                    coarse=coarse,
                    mapping=protocol.category_mapping,
                ),
                _coarse_rows(
                    train_primary,
                    image_ids=train_images,
                    coarse=coarse,
                    mapping=protocol.category_mapping,
                ),
                coarse=coarse,
                thresholds=thresholds,
                protocol=protocol,
            )
        specialist_points[held_out] = _sweep(
            _coarse_rows(
                train_gt,
                image_ids=train_images,
                coarse=args.specialist_coarse,
                mapping=protocol.category_mapping,
            ),
            _coarse_rows(
                train_specialist,
                image_ids=train_images,
                coarse=args.specialist_coarse,
                mapping=protocol.category_mapping,
            ),
            coarse=args.specialist_coarse,
            thresholds=thresholds,
            protocol=protocol,
        )

    frontiers: dict[str, Any] = {}
    for target_fdr in args.fdr_levels:
        baseline_stitched = {image_id: [] for image_id in seen_images}
        routed_stitched = {image_id: [] for image_id in seen_images}
        selections: dict[str, Any] = {}
        heldout_metrics: dict[str, Any] = {}
        for held_out in (0, 1, 2):
            selections[str(held_out)] = {"primary": {}}
            image_ids = folds[held_out]["image_ids"]
            for coarse in COARSE:
                selected = select_threshold(
                    primary_points[(held_out, coarse)], float(target_fdr)
                )
                selections[str(held_out)]["primary"][coarse] = selected
                selected_rows = _coarse_rows(
                    folds[held_out]["primary"],
                    image_ids=image_ids,
                    coarse=coarse,
                    mapping=protocol.category_mapping,
                    threshold=float(selected["threshold"]),
                )
                for image_id, rows in selected_rows.items():
                    baseline_stitched[image_id].extend(rows)
                    if coarse != args.specialist_coarse:
                        routed_stitched[image_id].extend(rows)
            selected_specialist = select_threshold(
                specialist_points[held_out], float(target_fdr)
            )
            selections[str(held_out)]["specialist"] = {
                args.specialist_coarse: selected_specialist
            }
            selected_rows = _coarse_rows(
                folds[held_out]["specialist"],
                image_ids=image_ids,
                coarse=args.specialist_coarse,
                mapping=protocol.category_mapping,
                threshold=float(selected_specialist["threshold"]),
            )
            for image_id, rows in selected_rows.items():
                routed_stitched[image_id].extend(rows)

            fold_baseline = {
                image_id: baseline_stitched[image_id] for image_id in image_ids
            }
            fold_candidate = {
                image_id: routed_stitched[image_id] for image_id in image_ids
            }
            fold_gt = folds[held_out]["gt"]
            baseline_fold_payload = _payload(fold_gt, fold_baseline, protocol)
            candidate_fold_payload = _payload(fold_gt, fold_candidate, protocol)
            heldout_metrics[str(held_out)] = {
                "primary_baseline": baseline_fold_payload,
                "routed_candidate": candidate_fold_payload,
                "delta": _delta(candidate_fold_payload, baseline_fold_payload),
            }

        baseline = _payload(all_gt, baseline_stitched, protocol)
        candidate = _payload(all_gt, routed_stitched, protocol)
        frontiers[f"{float(target_fdr):.3f}"] = {
            "crossfit_thresholds": selections,
            "heldout_metrics": heldout_metrics,
            "primary_baseline": baseline,
            "routed_candidate": candidate,
            "delta": _delta(candidate, baseline),
        }

    gate = frontiers["0.150"]
    vehicle_delta = gate["delta"]["per_coarse"][args.specialist_coarse]
    non_specialist_exact = all(
        gate["delta"]["per_coarse"][coarse] == {"recall": 0.0, "fdr": 0.0}
        for coarse in COARSE
        if coarse != args.specialist_coarse
    )
    passed = bool(
        non_specialist_exact
        and vehicle_delta["recall"] >= 0.01
        and vehicle_delta["fdr"] <= 0.0
        and gate["delta"]["recall"] >= 0.0
        and gate["delta"]["fdr"] <= 0.0
    )
    result = {
        "status": "complete",
        "protocol": "formal_cv3_two_folds_select_coarse_detector_route_v1",
        "specialist_coarse": args.specialist_coarse,
        "threshold_grid": {
            "start": args.threshold_start,
            "stop": args.threshold_stop,
            "step": args.threshold_step,
        },
        "frontiers": frontiers,
        "admission": {
            "target_fdr": 0.15,
            "requirements": {
                "non_specialist_outputs_exact": True,
                "specialist_recall_delta_min": 0.01,
                "specialist_fdr_delta_max": 0.0,
                "pooled_recall_delta_min": 0.0,
                "pooled_fdr_delta_max": 0.0,
            },
            "non_specialist_outputs_exact": non_specialist_exact,
            "passed": passed,
            "next_action": "train_full_specialist" if passed else "stop_specialist_route",
        },
        "input_sha256": input_sha,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
