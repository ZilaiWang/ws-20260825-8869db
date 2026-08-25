#!/usr/bin/env python3
"""Run R1-6 class-aware NMS after frozen CE+D4 reranking and C2 calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsdet.analysis.aircraft_refinement import (
    AIRCRAFT_CLASS_IDS,
    build_aircraft_variants,
    condition_probabilities,
    reconstruct_selected_crossfit_predictions,
)
from rsdet.analysis.crossfit_thresholds import (
    evaluate_ranking_workpoint,
    load_cv3_aggregate,
)
from rsdet.analysis.oof_detection import (
    decompose_official_errors,
    load_formal_ground_truth,
)
from rsdet.analysis.post_rerank_nms import run_crossfit_post_rerank_nms
from rsdet.analysis.proposal_reranking import (
    apply_frozen_c2,
    load_proposal_manifest,
    sha256_file,
)
from rsdet.evaluation.official_metric import RankingMetrics
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.postprocess.nms import class_aware_nms_predictions
from rsdet.utils.config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/r1_post_rerank_nms_v1.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _require_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA mismatch: {actual} != {expected}")
    return actual


def _ranking_payload(metrics: RankingMetrics) -> dict[str, Any]:
    return {
        "macro_recall": metrics.overall_recall,
        "macro_fdr": metrics.overall_fdr,
        "per_coarse": {
            name: {
                "macro_recall": item.macro_recall,
                "macro_fdr": item.macro_fdr,
                "pooled_recall": item.pooled_recall,
                "pooled_fdr": item.pooled_fdr,
                "fine_count": item.fine_count,
                "fine_ids": item.fine_ids,
            }
            for name, item in metrics.per_coarse.items()
        },
    }


def _metric_snapshot(summary: Mapping[str, Any], ranking: RankingMetrics) -> dict[str, Any]:
    official = summary["official_metrics"]
    return {
        "recall": float(official["overall_recall"]),
        "fdr": float(official["overall_fdr"]),
        "tp": int(official["details"]["tp"]),
        "fp": int(official["details"]["fp"]),
        "fn": int(official["details"]["fn"]),
        **_ranking_payload(ranking),
    }


def _subtract(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: float(candidate[key]) - float(baseline[key])
        for key in ("recall", "fdr", "macro_recall", "macro_fdr")
    } | {
        key: int(candidate[key]) - int(baseline[key])
        for key in ("tp", "fp", "fn")
    }


def _flatten_predictions(
    predictions: Mapping[int, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image_id in sorted(predictions):
        for record in predictions[image_id]:
            rows.append({"image_id": image_id, **dict(record)})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = Path.cwd().resolve()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    if config.get("contract_version") != "r1_post_rerank_nms_v1":
        raise ValueError("unexpected R1-6 contract version")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = config["inputs"]
    paths = {
        name: _resolve(project_root, inputs[name])
        for name in (
            "base_experiment_config",
            "condition_result",
            "inference_manifest",
            "base_logits_dir",
            "bundle_dir",
            "aggregate_dir",
            "formal_crop_manifest",
            "y1_calibration_result",
        )
    }
    input_sha = {
        "config": sha256_file(config_path),
        "condition_result": _require_sha(
            paths["condition_result"],
            inputs["condition_result_sha256"],
            "condition_result",
        ),
        "inference_manifest": _require_sha(
            paths["inference_manifest"],
            inputs["inference_manifest_sha256"],
            "inference_manifest",
        ),
        "formal_crop_manifest": _require_sha(
            paths["formal_crop_manifest"],
            inputs["formal_crop_manifest_sha256"],
            "formal_crop_manifest",
        ),
        "y1_calibration_result": _require_sha(
            paths["y1_calibration_result"],
            inputs["y1_calibration_result_sha256"],
            "y1_calibration_result",
        ),
    }
    input_sha["bundle"] = {
        str(fold): _require_sha(
            paths["bundle_dir"] / f"fold_{fold}_aircraft_bundle.npz",
            inputs["bundle_sha256"][str(fold)],
            f"bundle fold {fold}",
        )
        for fold in (0, 1, 2)
    }
    input_sha["aggregate"] = {
        name: _require_sha(
            paths["aggregate_dir"] / name,
            expected,
            f"aggregate/{name}",
        )
        for name, expected in inputs["aggregate_sha256"].items()
    }

    base_experiment = load_config(paths["base_experiment_config"])
    condition_result = json.loads(paths["condition_result"].read_text(encoding="utf-8"))
    condition_name = str(inputs["condition_name"])
    if condition_result.get("condition") != condition_name:
        raise ValueError("condition result/name mismatch")
    selected = {
        int(item["held_out_fold"]): str(item["selected_variant"])
        for item in condition_result["per_fold"]
    }
    records = load_proposal_manifest(paths["inference_manifest"])
    probabilities = condition_probabilities(
        records,
        paths["base_logits_dir"],
        paths["bundle_dir"],
        view=str(inputs["view"]),
    )
    _, _, image_folds = load_cv3_aggregate(paths["aggregate_dir"], candidate_floor=0.001)
    raw = reconstruct_selected_crossfit_predictions(
        records,
        probabilities,
        build_aircraft_variants(base_experiment["search"]),
        selected,
        image_folds=image_folds,
    )
    calibration = json.loads(paths["y1_calibration_result"].read_text(encoding="utf-8"))
    calibrated = apply_frozen_c2(calibration, raw, image_folds)

    protocol = parse_evaluation_protocol(load_config(_resolve(project_root, config["project_config"])))
    formal = load_formal_ground_truth(
        paths["formal_crop_manifest"],
        expected_sha256=inputs["formal_crop_manifest_sha256"],
        expected_images=int(inputs["expected_images"]),
        expected_annotations=int(inputs["expected_annotations"]),
    )
    baseline_summary, _, _ = decompose_official_errors(
        formal,
        calibrated,
        threshold=0.0,
        protocol=protocol,
        model_key="CE_D4_C2",
        include_cases=False,
    )
    baseline_ranking = evaluate_ranking_workpoint(
        formal.boxes,
        calibrated,
        threshold=0.0,
        protocol=protocol,
        require_complete_taxonomy=True,
    )
    baseline = _metric_snapshot(baseline_summary, baseline_ranking)
    expected = condition_result["merged"]["C2_frozen_after_r1"]
    parity = {
        key: float(baseline[key]) - float(expected[key])
        for key in ("recall", "fdr", "macro_recall", "macro_fdr")
    }
    if max(abs(value) for value in parity.values()) > 1e-12:
        raise RuntimeError(f"frozen CE+D4 parity failed: {parity}")

    postprocess = config["postprocess"]
    target_category_ids = tuple(
        int(value) for value in postprocess["target_category_ids"]
    )
    if target_category_ids != AIRCRAFT_CLASS_IDS:
        raise ValueError("R1-6 target categories must be the frozen 20 aircraft classes")
    selection, crossfit_suppressed = run_crossfit_post_rerank_nms(
        formal.boxes,
        calibrated,
        image_folds,
        iou_thresholds=postprocess["iou_thresholds"],
        maximum_pooled_recall_drop=float(
            postprocess["maximum_selection_pooled_recall_drop"]
        ),
        category_ids=target_category_ids,
        protocol=protocol,
    )
    fixed_iou_threshold = float(postprocess["fixed_deployment_iou_threshold"])
    suppressed = class_aware_nms_predictions(
        calibrated,
        fixed_iou_threshold,
        category_ids=target_category_ids,
    )
    suppressed = {
        image_id: suppressed.get(image_id, []) for image_id in sorted(image_folds)
    }
    crossfit_confirms_fixed = all(
        float(value) == fixed_iou_threshold
        for value in selection["selected_iou_threshold_by_fold"].values()
    )
    crossfit_prediction_parity = all(
        suppressed[image_id] == crossfit_suppressed.get(image_id, [])
        for image_id in image_folds
    )
    candidate_summary, _, _ = decompose_official_errors(
        formal,
        suppressed,
        threshold=0.0,
        protocol=protocol,
        model_key="CE_D4_C2_POST_NMS",
        include_cases=False,
    )
    candidate_ranking = evaluate_ranking_workpoint(
        formal.boxes,
        suppressed,
        threshold=0.0,
        protocol=protocol,
        require_complete_taxonomy=True,
    )
    candidate = _metric_snapshot(candidate_summary, candidate_ranking)
    delta = _subtract(candidate, baseline)

    baseline_dup = int(baseline_summary["fp_counts"]["FP_DUP"])
    candidate_dup = int(candidate_summary["fp_counts"]["FP_DUP"])
    duplicate_reduction_fraction = (
        (baseline_dup - candidate_dup) / baseline_dup if baseline_dup else 0.0
    )
    gates_config = config["decision_gates"]
    gates = {
        "official_gate": candidate["recall"] >= float(gates_config["official_recall_min"])
        and candidate["fdr"] <= float(gates_config["official_fdr_max"]),
        "pooled_recall_safety": delta["recall"]
        >= -float(gates_config["maximum_pooled_recall_drop"]),
        "macro_recall_safety": delta["macro_recall"]
        >= -float(gates_config["maximum_macro_recall_drop"]),
        "pooled_fdr_gain": -delta["fdr"]
        >= float(gates_config["minimum_pooled_fdr_reduction"]),
        "macro_fdr_gain": -delta["macro_fdr"]
        >= float(gates_config["minimum_macro_fdr_reduction"]),
        "duplicate_reduction": duplicate_reduction_fraction
        >= float(gates_config["minimum_fp_dup_reduction_fraction"]),
        "ship_vehicle_exact_parity": all(
            candidate["per_coarse"][name] == baseline["per_coarse"][name]
            for name in ("ship", "vehicle")
        ),
        "crossfit_confirms_fixed_threshold": crossfit_confirms_fixed
        and crossfit_prediction_parity,
    }
    scientific_admission = all(gates.values())
    decision = {
        "contract_version": config["contract_version"],
        "experiment_id": config["experiment_id"],
        "status": "complete",
        "scientific_scope": config["scientific_scope"]["level"],
        "baseline": baseline,
        "candidate": candidate,
        "delta_candidate_minus_baseline": delta,
        "fp_dup": {
            "baseline": baseline_dup,
            "candidate": candidate_dup,
            "reduction": baseline_dup - candidate_dup,
            "reduction_fraction": duplicate_reduction_fraction,
        },
        "gates": gates,
        "scientific_admission": scientific_admission,
        "formal_admission": False,
        "formal_admission_reason": "iterative OOF development; independent test pending",
        "fixed_deployment_iou_threshold": fixed_iou_threshold,
        "crossfit_confirms_fixed_threshold": crossfit_confirms_fixed,
        "crossfit_prediction_parity": crossfit_prediction_parity,
        "next_action": (
            "retain_as_deployment_candidate_and_validate_in_full_pipeline"
            if scientific_admission
            else "reject_post_rerank_nms_workpoint"
        ),
    }
    audit = {
        "contract_version": config["contract_version"],
        "status": "pass",
        "input_sha256": input_sha,
        "selected_variant_by_fold": selected,
        "frozen_result_parity": parity,
        "prediction_count_before": sum(len(items) for items in calibrated.values()),
        "prediction_count_after": sum(len(items) for items in suppressed.values()),
        "image_count": len(image_folds),
    }
    _write_json(output_dir / "input_audit.json", audit)
    _write_json(output_dir / "crossfit_selection.json", selection)
    _write_json(output_dir / "baseline_error_decomposition.json", baseline_summary)
    _write_json(output_dir / "post_nms_error_decomposition.json", candidate_summary)
    _write_json(output_dir / "selected_predictions_xyxy.json", _flatten_predictions(suppressed))
    _write_json(output_dir / "decision.json", decision)
    (output_dir / "R1_POST_RERANK_NMS_PASS").write_text(
        "pass\n" if scientific_admission else "scientific_gate_not_passed\n",
        encoding="utf-8",
    )
    print(
        "R1_POST_RERANK_NMS_COMPLETE "
        f"admission={scientific_admission} recall_delta={delta['recall']:.8f} "
        f"fdr_delta={delta['fdr']:.8f} fp_dup={baseline_dup}->{candidate_dup}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
