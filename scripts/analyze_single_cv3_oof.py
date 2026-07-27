#!/usr/bin/env python3
"""Run official descriptive analysis for one completed formal CV3 OOF model.

This is the single-model counterpart of ``analyze_cv3_oof_models.py``.  It is
used when M1 is complete but M3 is still pending; it deliberately omits
cross-model complementarity and never promotes a same-OOF threshold to a
deployment threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsdet.analysis.oof_detection import (
    build_threshold_curve,
    decompose_official_errors,
    load_analysis_config,
    load_formal_ground_truth,
    load_oof_aggregate,
    select_exploratory_workpoint,
)
from rsdet.evaluation.official_metric import evaluate_predictions
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.postprocess.calibration import filter_predictions_by_score
from rsdet.utils.config import load_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="单模型正式 CV3 OOF 官方指标和错误分解",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/m1_m3_cv3_oof_analysis_v1.yaml"
        ),
    )
    parser.add_argument(
        "--project-config",
        type=Path,
        default=Path("configs/project.yaml"),
    )
    parser.add_argument("--formal-crop-manifest", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--model-key", default="M1")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _metrics_payload(metrics: Any) -> dict[str, Any]:
    return {
        "overall_recall": metrics.recall,
        "overall_fdr": metrics.fdr,
        "details": dict(metrics.details),
        "per_class": {
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_analysis_config(args.config)
        protocol = parse_evaluation_protocol(load_config(args.project_config))
        if protocol.eval_version != "official_eval_v1":
            raise ValueError("只接受冻结 official_eval_v1")
        formal_gt = load_formal_ground_truth(
            args.formal_crop_manifest,
            expected_sha256=str(config["formal_crop_manifest_sha256"]),
            expected_images=int(config["expected_images"]),
            expected_annotations=int(config["expected_annotations"]),
        )
        for item in formal_gt.objects.values():
            if protocol.category_mapping.get(item.category_id) != item.class_name:
                raise ValueError(
                    "formal GT major_class 与官方协议映射不一致: "
                    f"{item.annotation_uid}"
                )
        aggregate, predictions = load_oof_aggregate(
            args.aggregate,
            expected_model_key=args.model_key,
            expected_manifest_sha256=str(config["cv3_manifest_sha256"]),
            expected_formal_crop_sha256=str(
                config["formal_crop_manifest_sha256"]
            ),
            expected_images=int(config["expected_images"]),
            candidate_floor=float(config["candidate_floor"]),
        )
        if aggregate["image_ids"] != set(formal_gt.image_ids):
            raise ValueError("OOF 与 formal GT 的 image_id 集合不一致")
        aggregate.pop("image_ids")

        output = args.output_dir.expanduser().resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"输出目录非空，禁止覆盖: {output}")
        output.mkdir(parents=True, exist_ok=True)

        curve, parity = build_threshold_curve(
            formal_gt.boxes,
            predictions,
            thresholds=config["thresholds"],
            protocol=protocol,
        )
        workpoint = select_exploratory_workpoint(
            curve,
            recall_min=protocol.recall_min,
            fdr_max=protocol.fdr_max,
        )
        floor_summary, _, _ = decompose_official_errors(
            formal_gt,
            predictions,
            threshold=float(config["candidate_floor"]),
            protocol=protocol,
            model_key=args.model_key,
            include_cases=False,
        )
        selected_summary, selected_cases, _ = decompose_official_errors(
            formal_gt,
            predictions,
            threshold=float(workpoint["threshold"]),
            protocol=protocol,
            model_key=args.model_key,
            include_cases=True,
        )
        image_folds: dict[int, int] = {}
        with (args.aggregate / "oof_images.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                image_id = int(row["image_id"])
                if image_id in image_folds:
                    raise ValueError(f"oof_images image_id 重复: {image_id}")
                image_folds[image_id] = int(row["fold"])
        if set(image_folds) != set(formal_gt.image_ids):
            raise ValueError("per-fold ledger 与 formal GT image_id 集合不一致")
        per_fold_metrics: dict[str, Any] = {}
        for fold in range(3):
            fold_images = {
                image_id
                for image_id, source_fold in image_folds.items()
                if source_fold == fold
            }
            fold_gt = {
                image_id: list(formal_gt.boxes.get(image_id, ()))
                for image_id in fold_images
            }
            fold_predictions = {
                image_id: list(predictions.get(image_id, ()))
                for image_id in fold_images
            }
            fold_payload: dict[str, Any] = {
                "images": len(fold_images),
                "ground_truth_objects": sum(map(len, fold_gt.values())),
            }
            for label, threshold in (
                ("candidate_floor", float(config["candidate_floor"])),
                ("exploratory_workpoint", float(workpoint["threshold"])),
            ):
                metrics = evaluate_predictions(
                    fold_gt,
                    filter_predictions_by_score(fold_predictions, threshold),
                    class_names=protocol.class_names,
                    category_mapping=protocol.category_mapping,
                    iou_thresholds=protocol.iou_thresholds,
                )
                fold_payload[label] = {
                    "threshold": threshold,
                    **_metrics_payload(metrics),
                }
            per_fold_metrics[str(fold)] = fold_payload

        threshold_fields = [
            "threshold",
            "detections_kept",
            "overall_recall",
            "overall_fdr",
            "tp",
            "fp",
            "fn",
            "official_gate_passed",
            *[
                f"{name}_{suffix}"
                for name in protocol.class_names
                for suffix in ("recall", "fdr", "tp", "fp", "fn")
            ],
        ]
        error_fields = [
            "model_key",
            "threshold",
            "case_side",
            "reason",
            "image_id",
            "item_uid",
            "category_id",
            "class_name",
            "score",
            "bbox_xyxy",
            "paired_item_uid",
            "paired_category_id",
            "paired_iou",
        ]
        _write_csv(output / "threshold_curve.csv", curve, threshold_fields)
        _write_json(output / "exploratory_workpoint.json", workpoint)
        _write_json(
            output / "candidate_floor_metrics_and_errors.json",
            floor_summary,
        )
        _write_json(
            output / "exploratory_workpoint_metrics_and_errors.json",
            selected_summary,
        )
        _write_csv(
            output / "exploratory_workpoint_error_cases.csv",
            selected_cases,
            error_fields,
        )
        _write_json(output / "per_fold_metrics.json", per_fold_metrics)

        metadata: dict[str, Any] = {
            "analysis_version": "single_model_cv3_oof_analysis_v1",
            "status": "complete_single_model_formal_oof_descriptive_analysis",
            "model_key": args.model_key,
            "counts": {
                "images": len(formal_gt.image_ids),
                "ground_truth_objects": formal_gt.annotation_count,
                "candidate_floor_proposals": int(aggregate["proposal_count"]),
            },
            "input": {
                "aggregate": str(args.aggregate.expanduser().resolve()),
                "aggregate_metadata_sha256": aggregate["metadata_sha256"],
                "predictions_sha256": aggregate["predictions_sha256"],
                "formal_crop_manifest": str(
                    args.formal_crop_manifest.expanduser().resolve()
                ),
                "formal_crop_manifest_sha256": _sha256(
                    args.formal_crop_manifest.expanduser().resolve()
                ),
            },
            "aggregate_cross_file_audit": aggregate[
                "aggregate_cross_file_audit"
            ],
            "relocated_artifacts": aggregate["relocated_artifacts"],
            "curve_parity": parity,
            "candidate_floor": floor_summary,
            "exploratory_workpoint": workpoint,
            "exploratory_workpoint_error_decomposition": selected_summary,
            "per_fold_metrics": per_fold_metrics,
            "upstream_recovery_amendment": aggregate.get(
                "recovery_amendment"
            ),
            "scientific_claim_scope": {
                "official_matching_metrics": True,
                "error_decomposition_is_diagnostic": True,
                "same_oof_selected_threshold_is_final": False,
                "deployment_admission": False,
                "cross_model_complementarity_available": False,
            },
            "artifacts": {},
        }
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name != "analysis_metadata.json":
                metadata["artifacts"][path.name] = {
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
        _write_json(output / "analysis_metadata.json", metadata)
    except (
        csv.Error,
        json.JSONDecodeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"SINGLE_OOF_ANALYSIS_FAIL: {error}", file=sys.stderr)
        return 1

    print(
        "SINGLE_OOF_ANALYSIS_PASS "
        f"model={args.model_key} images={len(formal_gt.image_ids)} "
        f"objects={formal_gt.annotation_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
