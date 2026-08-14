#!/usr/bin/env python3
"""Diagnose one frozen R1 condition without repeating variant/threshold search."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsdet.analysis.aircraft_refinement import (
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
from rsdet.analysis.proposal_reranking import (
    apply_frozen_c2,
    load_proposal_manifest,
    sha256_file,
)
from rsdet.data.xh_dataset import FINE_NAMES
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--condition-result", type=Path, required=True)
    parser.add_argument("--condition-name", required=True)
    parser.add_argument("--view", choices=("identity", "d4"), required=True)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--base-logits-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--formal-crop-manifest", type=Path, required=True)
    parser.add_argument("--y1-calibration-result", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_cases(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fine_confusions(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[int, int]] = Counter()
    for row in cases:
        if row["case_side"] != "ground_truth" or row["reason"] != "FN_CLS":
            continue
        true_id = int(row["category_id"])
        predicted_id = int(row["paired_category_id"])
        counts[(true_id, predicted_id)] += 1
    return [
        {
            "true_category_id": true_id,
            "true_name": FINE_NAMES[true_id],
            "predicted_category_id": predicted_id,
            "predicted_name": FINE_NAMES[predicted_id],
            "count": count,
        }
        for (true_id, predicted_id), count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def _metric_parity(
    observed: Mapping[str, Any], expected: Mapping[str, Any], ranking: Any
) -> dict[str, float]:
    return {
        "recall_delta": float(observed["overall_recall"]) - float(expected["recall"]),
        "fdr_delta": float(observed["overall_fdr"]) - float(expected["fdr"]),
        "macro_recall_delta": float(ranking.overall_recall)
        - float(expected["macro_recall"]),
        "macro_fdr_delta": float(ranking.overall_fdr) - float(expected["macro_fdr"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    destination = args.output_dir.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=True)

    experiment = load_config(args.experiment_config)
    condition_result = json.loads(args.condition_result.read_text(encoding="utf-8"))
    if condition_result.get("condition") != args.condition_name:
        raise ValueError("condition result/name 不一致")
    selected = {
        int(item["held_out_fold"]): str(item["selected_variant"])
        for item in condition_result["per_fold"]
    }
    records = load_proposal_manifest(args.inference_manifest)
    probabilities = condition_probabilities(
        records,
        args.base_logits_dir,
        args.bundle_dir,
        view=args.view,
    )
    _, _, image_folds = load_cv3_aggregate(args.aggregate_dir, candidate_floor=0.001)
    raw = reconstruct_selected_crossfit_predictions(
        records,
        probabilities,
        build_aircraft_variants(experiment["search"]),
        selected,
        image_folds=image_folds,
    )
    calibration = json.loads(args.y1_calibration_result.read_text(encoding="utf-8"))
    calibrated = apply_frozen_c2(calibration, raw, image_folds)

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    formal = load_formal_ground_truth(
        args.formal_crop_manifest,
        expected_sha256=experiment["inputs"]["formal_crop_manifest_sha256"],
        expected_images=int(experiment["inputs"]["expected_images"]),
        expected_annotations=int(experiment["inputs"]["expected_annotations"]),
    )
    summary, cases, _ = decompose_official_errors(
        formal,
        calibrated,
        threshold=0.0,
        protocol=protocol,
        model_key=args.condition_name,
    )
    ranking = evaluate_ranking_workpoint(
        formal.boxes,
        calibrated,
        threshold=0.0,
        protocol=protocol,
        require_complete_taxonomy=True,
    )
    expected = condition_result["merged"]["C2_frozen_after_r1"]
    parity = _metric_parity(summary["official_metrics"], expected, ranking)
    if max(abs(value) for value in parity.values()) > 1e-12:
        raise RuntimeError(f"冻结 condition 重建不一致: {parity}")

    confusions = _fine_confusions(cases)
    summary.update(
        {
            "contract_version": "r1_frozen_condition_diagnostic_v1",
            "status": "complete",
            "condition": args.condition_name,
            "view": args.view,
            "selected_variant_by_fold": selected,
            "official_ranking": {
                "macro_recall": ranking.overall_recall,
                "macro_fdr": ranking.overall_fdr,
            },
            "frozen_result_parity": parity,
            "input_sha256": {
                "experiment_config": sha256_file(args.experiment_config),
                "condition_result": sha256_file(args.condition_result),
                "inference_manifest": sha256_file(args.inference_manifest),
                "formal_crop_manifest": sha256_file(args.formal_crop_manifest),
                "y1_calibration_result": sha256_file(args.y1_calibration_result),
            },
            "fine_confusion_pair_count": len(confusions),
        }
    )
    _write_json(destination / "summary.json", summary)
    _write_json(destination / "fine_confusions.json", confusions)
    _write_cases(destination / "error_cases.csv", cases)
    print(
        "R1_FROZEN_CONDITION_DIAGNOSTIC_PASS "
        f"condition={args.condition_name} fp={summary['conservation']['official_fp']} "
        f"fn={summary['conservation']['official_fn']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
