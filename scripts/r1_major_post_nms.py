#!/usr/bin/env python3
"""Evaluate fixed official-IoU same-fine NMS for ship and vehicle after R1-6."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsdet.analysis.crossfit_thresholds import evaluate_ranking_workpoint
from rsdet.analysis.oof_detection import decompose_official_errors, load_formal_ground_truth
from rsdet.analysis.proposal_reranking import sha256_file
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.postprocess.nms import category_threshold_nms_predictions
from rsdet.utils.config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/r1_major_post_nms_v1.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _require_sha(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA mismatch: {actual} != {expected}")
    return actual


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_predictions(path: Path, image_ids: set[int]) -> dict[int, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("baseline predictions must be a flat JSON list")
    grouped = {image_id: [] for image_id in sorted(image_ids)}
    seen_uids: set[str] = set()
    for row in payload:
        image_id = int(row["image_id"])
        if image_id not in grouped:
            raise ValueError(f"prediction image_id outside ledger: {image_id}")
        record = {key: value for key, value in row.items() if key != "image_id"}
        uid = str(record.get("proposal_uid", ""))
        if not uid or uid in seen_uids:
            raise ValueError(f"missing or duplicate proposal_uid: {uid!r}")
        seen_uids.add(uid)
        grouped[image_id].append(record)
    return grouped


def _flatten(predictions: Mapping[int, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {"image_id": image_id, **dict(record)}
        for image_id in sorted(predictions)
        for record in predictions[image_id]
    ]


def _ranking_payload(ranking: Any) -> dict[str, Any]:
    return {
        "macro_recall": ranking.overall_recall,
        "macro_fdr": ranking.overall_fdr,
        "per_coarse": {
            name: {
                "macro_recall": item.macro_recall,
                "macro_fdr": item.macro_fdr,
                "pooled_recall": item.pooled_recall,
                "pooled_fdr": item.pooled_fdr,
                "fine_count": item.fine_count,
                "fine_ids": item.fine_ids,
            }
            for name, item in ranking.per_coarse.items()
        },
    }


def _snapshot(summary: Mapping[str, Any], ranking: Any) -> dict[str, Any]:
    official = summary["official_metrics"]
    return {
        "recall": float(official["overall_recall"]),
        "fdr": float(official["overall_fdr"]),
        "tp": int(official["details"]["tp"]),
        "fp": int(official["details"]["fp"]),
        "fn": int(official["details"]["fn"]),
        **_ranking_payload(ranking),
    }


def _evaluate(
    formal: Any,
    predictions: Mapping[int, Sequence[Mapping[str, Any]]],
    protocol: Any,
    model_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary, _, _ = decompose_official_errors(
        formal,
        predictions,
        threshold=0.0,
        protocol=protocol,
        model_key=model_key,
        include_cases=False,
    )
    ranking = evaluate_ranking_workpoint(
        formal.boxes,
        predictions,
        threshold=0.0,
        protocol=protocol,
        require_complete_taxonomy=True,
    )
    return summary, _snapshot(summary, ranking)


def _removed_uids(
    before: Mapping[int, Sequence[Mapping[str, Any]]],
    after: Mapping[int, Sequence[Mapping[str, Any]]],
) -> list[str]:
    before_uids = {str(record["proposal_uid"]) for records in before.values() for record in records}
    after_uids = {str(record["proposal_uid"]) for records in after.values() for record in records}
    return sorted(before_uids - after_uids)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path.cwd().resolve()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    if config.get("contract_version") != "r1_major_post_nms_v1":
        raise ValueError("unexpected R1-9 contract")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    inputs = config["inputs"]
    paths = {
        name: (root / inputs[name]).resolve()
        for name in ("baseline_predictions", "formal_crop_manifest", "image_ledger")
    }
    input_sha = {
        "config": sha256_file(config_path),
        **{name: _require_sha(paths[name], inputs[f"{name}_sha256"], name) for name in paths},
    }
    with paths["image_ledger"].open("r", encoding="utf-8-sig", newline="") as handle:
        ledger = {int(row["image_id"]): row for row in csv.DictReader(handle)}
    if len(ledger) != int(inputs["expected_images"]):
        raise ValueError("image ledger count mismatch")
    baseline_predictions = _load_predictions(paths["baseline_predictions"], set(ledger))

    protocol = parse_evaluation_protocol(load_config(config["project_config"]))
    formal = load_formal_ground_truth(
        paths["formal_crop_manifest"],
        expected_sha256=inputs["formal_crop_manifest_sha256"],
        expected_images=int(inputs["expected_images"]),
        expected_annotations=int(inputs["expected_annotations"]),
    )
    baseline_summary, baseline = _evaluate(formal, baseline_predictions, protocol, "R1_6_BASELINE")
    if int(baseline_summary["fp_counts"]["FP_DUP"]) != int(inputs["expected_baseline_fp_dup"]):
        raise ValueError("baseline FP_DUP mismatch")

    thresholds = {
        int(category): float(threshold)
        for category, threshold in config["postprocess"]["category_thresholds"].items()
    }
    variants = {
        "ship_only": {
            category: threshold for category, threshold in thresholds.items() if category < 4
        },
        "vehicle_only": {
            category: threshold for category, threshold in thresholds.items() if category == 24
        },
        "combined": thresholds,
    }
    evaluations: dict[str, Any] = {}
    selected_predictions: dict[int, list[dict[str, Any]]] | None = None
    for name, category_thresholds in variants.items():
        candidate_predictions = category_threshold_nms_predictions(
            baseline_predictions, category_thresholds
        )
        candidate_summary, candidate = _evaluate(
            formal, candidate_predictions, protocol, f"R1_9_{name.upper()}"
        )
        evaluations[name] = {
            "category_thresholds": category_thresholds,
            "metrics": candidate,
            "error_decomposition": candidate_summary,
            "removed_count": len(_removed_uids(baseline_predictions, candidate_predictions)),
        }
        if name == "combined":
            selected_predictions = candidate_predictions

    if selected_predictions is None:
        raise RuntimeError("combined predictions missing")
    candidate = evaluations["combined"]["metrics"]
    candidate_summary = evaluations["combined"]["error_decomposition"]
    delta = {
        key: float(candidate[key]) - float(baseline[key])
        for key in ("recall", "fdr", "macro_recall", "macro_fdr")
    } | {key: int(candidate[key]) - int(baseline[key]) for key in ("tp", "fp", "fn")}
    baseline_dup = int(baseline_summary["fp_counts"]["FP_DUP"])
    candidate_dup = int(candidate_summary["fp_counts"]["FP_DUP"])
    gates_config = config["decision_gates"]
    gates = {
        "exact_tp_fn_recall_parity": candidate["tp"] == baseline["tp"]
        and candidate["fn"] == baseline["fn"]
        and candidate["recall"] == baseline["recall"]
        and candidate["macro_recall"] == baseline["macro_recall"],
        "fp_reduction": candidate["fp"] < baseline["fp"],
        "macro_fdr_noninferiority": candidate["macro_fdr"] <= baseline["macro_fdr"],
        "aircraft_exact_parity": candidate["per_coarse"]["aircraft"]
        == baseline["per_coarse"]["aircraft"],
        "fp_dup_reduction": baseline_dup - candidate_dup
        >= int(gates_config["minimum_fp_dup_reduction"]),
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
        },
        "removed_count": evaluations["combined"]["removed_count"],
        "gates": gates,
        "scientific_admission": scientific_admission,
        "formal_admission": False,
        "formal_admission_reason": "iterative OOF fixed-rule diagnostic",
        "next_action": (
            "retain_as_final_chain_candidate"
            if scientific_admission
            else "stop_ship_vehicle_post_nms"
        ),
    }
    _write_json(output / "input_audit.json", {"status": "pass", "input_sha256": input_sha})
    _write_json(output / "baseline_error_decomposition.json", baseline_summary)
    _write_json(output / "variant_evaluations.json", evaluations)
    _write_json(output / "decision.json", decision)
    _write_json(output / "selected_predictions_xyxy.json", _flatten(selected_predictions))
    _write_json(
        output / "removed_proposal_uids.json",
        _removed_uids(baseline_predictions, selected_predictions),
    )
    print(
        "R1_MAJOR_POST_NMS_COMPLETE "
        f"admission={scientific_admission} tp_delta={delta['tp']} "
        f"fp_delta={delta['fp']} fn_delta={delta['fn']} "
        f"fp_dup={baseline_dup}->{candidate_dup}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
