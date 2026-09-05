#!/usr/bin/env python3
"""Validate and summarize the Plan-15 progressive S1280 full candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--background-result", type=Path, required=True)
    parser.add_argument("--background-runtime", type=Path, required=True)
    parser.add_argument("--timing-summary", type=Path, required=True)
    parser.add_argument("--input-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    training = load_json(args.training_result)
    background = load_json(args.background_result)
    background_runtime = load_json(args.background_runtime)
    timing = load_json(args.timing_summary)
    contract = load_json(args.input_contract)
    with args.results_csv.open(newline="", encoding="utf-8") as handle:
        epoch_rows = list(csv.DictReader(handle))

    if training.get("status") != "complete_candidate":
        raise ValueError("training did not reach complete_candidate")
    if len(epoch_rows) != 40:
        raise ValueError(f"expected exactly 40 adaptation epochs, got {len(epoch_rows)}")
    if [int(row["epoch"].strip()) for row in epoch_rows] != list(range(1, 41)):
        raise ValueError("epoch history is not exactly 1..40")
    for row in epoch_rows:
        losses = [float(value) for key, value in row.items() if "loss" in key]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError(f"missing/nonfinite loss at epoch {row['epoch']}")
    if int(training.get("imgsz", -1)) != 1280:
        raise ValueError("candidate is not the admitted S1280 adaptation")
    if float(contract.get("deployment_threshold", -1.0)) != 0.536:
        raise ValueError("unexpected frozen deployment threshold")
    if timing.get("status") != "cv3_oof_pseudo_inference_complete":
        raise ValueError("timing-only pseudo-10K run is incomplete")
    folds = timing.get("folds")
    if not isinstance(folds, list) or len(folds) != 3:
        raise ValueError("timing summary must contain all three pseudo-10K folds")
    if int(background_runtime.get("image_count", -1)) <= 0:
        raise ValueError("background inference has no images")

    wall_seconds = [
        float(image["wall_seconds"])
        for fold in folds for image in fold["image_timings"]
    ]
    if not wall_seconds or not all(math.isfinite(value) and value > 0 for value in wall_seconds):
        raise ValueError("missing/invalid per-image timing measurements")
    route = background["route"]
    checkpoint = Path(str(training["checkpoint"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if sha256(checkpoint) != training["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA does not match training result")

    payload = {
        "schema_version": "scaleroute_plan15_progressive40_full_validation_v1",
        "status": "complete_pending_submission_decision",
        "scientific_basis": "paired_CV3_P40_selected_before_full_training",
        "external_training_data": False,
        "adaptation_epochs": 40,
        "effective_training_history": "mature_S1024_full_160e_plus_S1280_adaptation_40e",
        "hardware_migration": {
            key: training[key] for key in (
                "resume_checkpoint_sha256", "resume_completed_epochs", "device", "batch",
                "previous_global_batch", "nominal_effective_batch",
                "not_bitwise_equivalent_to_single_gpu", "rng_state",
            ) if key in training
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": training["checkpoint_sha256"],
        "deployment_threshold": contract["deployment_threshold"],
        "background_100mp": route,
        "background_prediction_floor": background_runtime["confidence"],
        "timing_only_hard_pseudo10k": {
            "image_count": len(wall_seconds),
            "mean_wall_seconds": sum(wall_seconds) / len(wall_seconds),
            "max_wall_seconds": max(wall_seconds),
            "measurements": wall_seconds,
            "not_a_scientific_accuracy_evaluation": True,
        },
        "docker_packaging_performed": False,
        "official_submission_performed": False,
        "artifacts": {
            "training_result_sha256": sha256(args.training_result),
            "results_csv_sha256": sha256(args.results_csv),
            "background_result_sha256": sha256(args.background_result),
            "timing_summary_sha256": sha256(args.timing_summary),
            "input_contract_sha256": sha256(args.input_contract),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        "SCALEROUTE_PROGRESSIVE_FULL_VALIDATION_COMPLETE "
        f"checkpoint_sha256={payload['checkpoint_sha256']} "
        f"background_fp_per_100mp={route['false_positives_per_100mp']:.6f} "
        f"timing_mean={payload['timing_only_hard_pseudo10k']['mean_wall_seconds']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
