#!/usr/bin/env python3
"""Retrospective recipe-direction check on the fresh paired-trend holdouts.

The two endpoints are predeclared, not chosen after seeing their scores. This
is a benchmark sanity check, never an estimate of prospective hit rate.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from rsdet.evaluation.absolute_score import COARSE_CLASSES, platform_confirmed_score
from rsdet.experiments.paired_trend import OFFICIAL_WEIGHT_SHA, read, safe_path, sha, write
from scripts import run_paired_trend as pipeline
from scripts.compare_candidate_trend import compare


def direction(value: float, deadband: float = 0.5) -> str:
    return "positive" if value > deadband else ("negative" if value < -deadband else "uncertain")


def official_anchor(summary: dict) -> dict:
    scores = {}
    for tag in ("v1.0", "v2.0"):
        raw = summary["submissions"][tag]["metrics"]
        scores[tag] = platform_confirmed_score(
            {c: {"recall": raw["basic_metrics"][c]["recall"],
                 "fdr": raw["basic_metrics"][c]["false_detection_rate"]}
             for c in COARSE_CLASSES}, raw["average_inference_time_sec"],
        )
        if abs(scores[tag]["total_score"] - raw["score_details"]["total_score"]) > 0.0001:
            raise ValueError("official score does not reproduce to displayed precision")
    b, c = scores["v1.0"], scores["v2.0"]
    return {
        "baseline": b, "candidate": c,
        "delta_quality_contribution": sum(c["seven_subscores"][:6]) / 7
        - sum(b["seven_subscores"][:6]) / 7,
        "delta_total_score": c["total_score"] - b["total_score"],
        "per_coarse_delta": {
            label: {"recall_pp": 100 * (c["per_coarse"][label]["recall"] - b["per_coarse"][label]["recall"]),
                    "fdr_pp": 100 * (c["per_coarse"][label]["fdr"] - b["per_coarse"][label]["fdr"])}
            for label in COARSE_CLASSES
        },
    }


def prepare(base: Path, bundle: Path, official: Path, output: Path) -> dict:
    plan = read(base / "plan.json")
    if plan["bundle_sha256"] != sha(bundle / "contract.json"):
        raise ValueError("wrong reference split")
    if (base / "foundation/complete.json").exists():
        raise ValueError("declare anchor audit before foundation completes")
    frozen = pipeline.validate_bundle(bundle, pipeline.PROJECT)
    contract = {
        "version": "paired_anchor_recipe_audit_v1",
        "purpose": "retrospective_two_endpoint_recipe_direction_check",
        "base": str(base.resolve()), "bundle_sha256": sha(bundle / "contract.json"),
        "training_plan_sha256": sha(base / "plan.json"),
        "script_sha256": sha(Path(__file__)),
        "official_source_sha256": sha(official), "official": official_anchor(read(official)),
        "foundation_inference": {**frozen["inference"], "imgsz": 1024},
        "adaptation_inference": frozen["inference"],
        "threshold_selection": "same_model_development_only_max_official_quality",
        "confirmation_policy": "both_declared_anchors_once_even_if_negative; audit_only",
        "deadband": frozen["deadband"],
        "heldout_predictions_observed_when_declaring": False,
        "official_outcomes_already_known": True,
        "same_training_universe": True, "equal_training_compute": False,
        "not_exact_historical_reproduction": [
            "new grouped 70/15/15 split, not full training data",
            "stable single GPU batch12/8, not historical mid-run DDP migration",
            "current common YOLO adapter, not every historical wrapper",
            "own development threshold, not historical transferred thresholds",
        ],
        "automatic_full_or_submission_admission": False,
    }
    write(output / "contract.json", contract)
    return contract


def execute(bundle: Path, data: Path, output: Path, device: str) -> dict:
    c = read(output / "contract.json")
    base = Path(c["base"])
    frozen = pipeline.validate_bundle(bundle, pipeline.PROJECT, data)
    if c["script_sha256"] != sha(Path(__file__)) or c["training_plan_sha256"] != sha(base / "plan.json"):
        raise ValueError("anchor script/plan changed after declaration")
    if c["bundle_sha256"] != sha(bundle / "contract.json"):
        raise ValueError("anchor bundle changed")
    if (output / "foundation").exists() or (output / "result.json").exists():
        raise FileExistsError("partial/completed anchor audit preserved; no implicit rerun")
    plan = read(base / "plan.json")
    parent = base / "foundation/run/weights/last.pt"
    final = base / "adaptation/run/weights/last.pt"
    previous_sha = OFFICIAL_WEIGHT_SHA
    for stage in ("foundation", "adaptation"):
        receipt = read(base / stage / "complete.json")
        checkpoint = base / stage / "run/weights/last.pt"
        results = base / stage / "run/results.csv"
        pipeline.check_epochs(results, plan["recipe"][stage]["epochs"])
        if (receipt["checkpoint_sha256"] != sha(checkpoint)
                or receipt["initial_sha256"] != previous_sha
                or receipt["plan_sha256"] != c["training_plan_sha256"]
                or receipt["results_sha256"] != sha(results)):
            raise ValueError("training stage receipt mismatch")
        previous_sha = sha(checkpoint)
    p40 = base / "evaluation"
    r = read(p40 / "review.json")
    if r["role"] != "baseline" or r["bundle_sha256"] != c["bundle_sha256"]:
        raise ValueError("wrong P40 reference evaluation")
    for rel, digest in r["artifacts"].items():
        if sha(safe_path(p40, rel)) != digest:
            raise ValueError("P40 evaluation artifact tampered")
    for part in ("development", "confirmation"):
        request = read(p40 / part / "inference.json")["request"]
        if request["config"] != c["adaptation_inference"] or request["checkpoint_sha256"] != sha(final):
            raise ValueError("P40 inference does not match declared anchor")
    lineage = read(base / "lineage.json")
    pipeline.validate_lineage(lineage, bundle, final)
    if len(lineage["ancestors"]) != 3 or lineage["ancestors"][1]["checkpoint_sha256"] != sha(parent):
        raise ValueError("P40 is not derived from this foundation")
    foundation_lineage = {
        **lineage, "checkpoint_sha256": sha(parent), "ancestors": lineage["ancestors"][:2],
        "training_recipe": {"foundation": plan["recipe"]["foundation"]},
    }
    pipeline.validate_lineage(foundation_lineage, bundle, parent)
    write(output / "foundation/lineage.json", foundation_lineage)
    comparisons = {}
    threshold = None
    for part in ("development", "confirmation"):
        gt = bundle / f"{part}_gt.json"
        pred = pipeline.infer(parent, gt, data, output / "foundation" / part,
                              c["foundation_inference"], device)
        if part == "development":
            threshold, selection = pipeline.select_threshold(gt, pred, frozen["threshold_grid"])
            write(output / "foundation/threshold.json", {**selection, "threshold": threshold,
                  "checkpoint_sha256": sha(parent), "bundle_sha256": c["bundle_sha256"]})
        measured = pipeline.metrics(gt, pred, threshold)
        write(output / "foundation" / f"{part}_metrics.json", measured)
        comparisons[part] = compare(measured, read(p40 / f"{part}_metrics.json"), deadband=c["deadband"])
    actual = c["official"]
    agreement = {}
    for part, result in comparisons.items():
        agreement[part] = {
            "quality_direction_matches_known_official_pair": direction(result["delta_quality_contribution"])
            == direction(actual["delta_quality_contribution"]),
            "local_quality_direction": direction(result["delta_quality_contribution"]),
            "per_coarse_sign_matches": {
                label: {key: direction(result["per_coarse_delta"][label][key], 1e-8)
                        == direction(actual["per_coarse_delta"][label][key], 1e-8)
                        for key in ("recall_pp", "fdr_pp")}
                for label in COARSE_CLASSES
            },
        }
    result = {
        "status": "complete_recipe_anchor_audit", "comparisons": comparisons,
        "known_official_pair": actual, "direction_checks": agreement,
        "foundation_threshold": threshold, "p40_threshold": r["threshold"],
        "retrospective_comparisons": 1, "prospective_comparisons": 0,
        "hit_rate_estimate": None, "official_score_forecast": False,
        "automatic_full_or_submission_admission": False,
        "interpretation": "one known recipe pair can falsify, but cannot validate general trend reliability",
        "local_total_delta": None, "timing_note": "quality-only; no matched foundation 100MP timing",
        "artifacts": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*.json"))},
    }
    write(output / "result.json", result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "run"))
    p.add_argument("--bundle", type=Path, default=ROOT / "data/splits/paired_trend_v1")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--official", type=Path, default=ROOT / "reports/experiments/formal_attempt2_20260903/analysis_summary.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if a.command == "prepare":
        result = prepare(a.base, a.bundle, a.official, a.output)
    elif a.execute:
        if str(a.base.resolve()) != read(a.output / "contract.json")["base"]:
            raise ValueError("base differs from declaration")
        result = execute(a.bundle, a.data_root, a.output, a.device)
    else:
        p.error("run requires --execute")
    print(result.get("status", "anchor_pair_declared_before_holdout_predictions"))


if __name__ == "__main__":
    main()
