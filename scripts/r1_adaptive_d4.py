#!/usr/bin/env python3
"""Evaluate fixed confidence-gated D4 using already cached CE bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from rsdet.analysis.aircraft_refinement import (
    adaptive_d4_probabilities,
    build_aircraft_variants,
)
from rsdet.analysis.proposal_reranking import (
    load_proposal_manifest,
    run_reranking_crossfit,
    sha256_file,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

CONTRACT = "r1_adaptive_d4_v1"


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("contract_version") != CONTRACT:
        raise ValueError("adaptive D4 config contract 非法")
    return payload


def _json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _metrics(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return result["merged"]["C2_frozen_after_r1"]


def _delta(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(reference[key])
        for key in ("recall", "fdr", "macro_recall", "macro_fdr")
    }


def evaluate(args: argparse.Namespace, config: dict[str, Any]) -> int:
    destination = args.output_dir.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=True)
    inputs = config["inputs"]
    checks = {
        "inference_manifest": args.inference_manifest,
        "formal_crop_manifest": args.formal_crop_manifest,
        "y1_calibration_result": args.y1_calibration_result,
        "ce_identity_result": args.ce_identity_result,
        "ce_d4_result": args.ce_d4_result,
    }
    for name, path in checks.items():
        actual = sha256_file(path)
        expected = inputs[f"{name}_sha256"]
        if actual != expected:
            raise ValueError(f"{name} SHA 不匹配: {actual}")
    for fold in (0, 1, 2):
        for name, root, expected_key, pattern in (
            (
                "base_logits",
                args.base_logits_dir,
                "base_logits_sha256",
                f"fold_{fold}_logits.npz",
            ),
            (
                "ce_bundle",
                args.ce_bundle_dir,
                "ce_bundle_sha256",
                f"fold_{fold}_aircraft_bundle.npz",
            ),
        ):
            actual = sha256_file(root / pattern)
            expected = inputs[expected_key][str(fold)]
            if actual != expected:
                raise ValueError(f"fold{fold} {name} SHA 不匹配: {actual}")
    records = load_proposal_manifest(args.inference_manifest)
    threshold = float(config["adaptive_d4"]["maximum_identity_probability"])
    probabilities, compute_audit = adaptive_d4_probabilities(
        records,
        args.base_logits_dir,
        args.ce_bundle_dir,
        maximum_identity_probability=threshold,
    )
    project = load_config(config["project_config"])
    protocol = parse_evaluation_protocol(project)
    y1 = json.loads(args.y1_calibration_result.read_text(encoding="utf-8"))
    search = config["search"]
    result, _ = run_reranking_crossfit(
        records=records,
        probabilities=probabilities,
        aggregate_dir=args.aggregate_dir,
        formal_crop_manifest_path=args.formal_crop_manifest,
        y1_calibration_result=y1,
        protocol=protocol,
        variants=build_aircraft_variants(search),
        threshold_start=float(search["threshold_start"]),
        threshold_stop=float(search["threshold_stop"]),
        threshold_step=float(search["threshold_step"]),
        maximum_selection_recall_drop=float(search["maximum_selection_recall_drop"]),
    )
    result["condition"] = f"ce_adaptive_d4_p{threshold:.2f}"
    result["compute_audit"] = compute_audit
    identity = json.loads(args.ce_identity_result.read_text(encoding="utf-8"))
    full_d4 = json.loads(args.ce_d4_result.read_text(encoding="utf-8"))
    if identity.get("condition") != "ce_identity" or full_d4.get("condition") != "ce_d4":
        raise ValueError("R1-1 reference condition 名称不匹配")
    candidate_metrics = _metrics(result)
    identity_metrics = _metrics(identity)
    full_d4_metrics = _metrics(full_d4)
    gates = config["decision_gates"]
    retain = bool(
        candidate_metrics["recall"]
        >= full_d4_metrics["recall"] - float(gates["maximum_recall_gap_vs_full_d4"])
        and candidate_metrics["fdr"]
        <= full_d4_metrics["fdr"] + float(gates["maximum_fdr_gap_vs_full_d4"])
        and compute_audit["view_compute_ratio_vs_full_d4"]
        <= float(gates["maximum_compute_ratio_vs_full_d4"])
    )
    decision = {
        "contract_version": CONTRACT,
        "status": "complete",
        "formal_admission": False,
        "condition": result["condition"],
        "adaptive_gate_passed": retain,
        "compute_audit": compute_audit,
        "metrics": candidate_metrics,
        "delta_vs_ce_identity": _delta(candidate_metrics, identity_metrics),
        "delta_vs_ce_full_d4": _delta(candidate_metrics, full_d4_metrics),
        "next_action": (
            "retain_adaptive_d4_for_10k_runtime_confirmation"
            if retain
            else "retain_full_d4_or_distill_ensemble"
        ),
    }
    _json(destination / "adaptive_d4_result.json", result)
    _json(destination / "decision.json", decision)
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--inference-manifest", type=Path, required=True)
    value.add_argument("--base-logits-dir", type=Path, required=True)
    value.add_argument("--ce-bundle-dir", type=Path, required=True)
    value.add_argument("--aggregate-dir", type=Path, required=True)
    value.add_argument("--formal-crop-manifest", type=Path, required=True)
    value.add_argument("--y1-calibration-result", type=Path, required=True)
    value.add_argument("--ce-identity-result", type=Path, required=True)
    value.add_argument("--ce-d4-result", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return evaluate(args, _load_config(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
