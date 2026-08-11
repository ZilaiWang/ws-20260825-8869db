#!/usr/bin/env python3
"""Run the formal C0-C3 YOLO score-calibration screen on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.postprocess.yolo_calibration import run_yolo_calibration_crossfit
from rsdet.utils.config import load_config
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="y1_crossfit_calibration")


def build_decision(result: dict[str, object]) -> dict[str, object]:
    """Apply the pre-declared scientific admission rules."""

    merged = result["merged_held_out"]
    c1 = merged["C1_coarse"]
    c2 = merged["C2_prior"]
    c3 = merged["C3_fractal_proxy"]

    def all_folds_improve_fdr(candidate: dict[str, object]) -> bool:
        directions = candidate["fold_direction_vs_c0"]
        return len(directions) == 3 and all(item["fdr_delta"] < 0.0 for item in directions)

    c1_admitted = bool(
        c1["official_gate_passed"]
        and c1["delta_vs_c0"]["recall"] >= -0.005
        and c1["delta_vs_c0"]["fdr"] <= -0.005
    )
    c2_admitted = bool(
        c2["official_gate_passed"]
        and c2["delta_vs_c0"]["recall"] >= -0.005
        and c2["delta_vs_c0"]["fdr"] <= -0.01
        and c2["delta_vs_c0"]["overall_macro_fdr"] <= -0.01
        and all_folds_improve_fdr(c2)
    )
    c3_incremental_macro_fdr = (
        c3["official_ranking"]["overall_macro_fdr"] - c2["official_ranking"]["overall_macro_fdr"]
    )
    c3_incremental_recall = c3["recall"] - c2["recall"]
    c3_admitted = bool(
        c2_admitted
        and c3["official_gate_passed"]
        and (
            c3_incremental_macro_fdr <= -0.002
            or (c3_incremental_recall >= 0.002 and c3_incremental_macro_fdr <= 0.0)
        )
    )
    selected = "C3_fractal_proxy" if c3_admitted else "C2_prior" if c2_admitted else "C0_global"
    return {
        "contract_version": "y1_calibration_decision_v1",
        "status": "complete",
        "selected_method": selected,
        "operational_admission": selected != "C0_global",
        "methods": {
            "C0_global": {"admitted": True, "role": "formal_reference"},
            "C1_coarse": {"admitted": c1_admitted},
            "C2_prior": {
                "admitted": c2_admitted,
                "requires_exact_fracal_claim": False,
                "scope": "post_nms_scalar_calibration",
            },
            "C3_fractal_proxy": {
                "admitted": c3_admitted,
                "incremental_macro_fdr_vs_c2": c3_incremental_macro_fdr,
                "incremental_recall_vs_c2": c3_incremental_recall,
                "exact_fracal_admission": False,
            },
        },
        "rules": {
            "maximum_recall_drop": 0.005,
            "minimum_pooled_fdr_gain": 0.01,
            "minimum_macro_fdr_gain": 0.01,
            "require_all_fold_fdr_direction": True,
            "C3_minimum_incremental_macro_fdr_gain": 0.002,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Y1 C0-C3 strict cross-fit calibration")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/y1_crossfit_calibration.yaml"),
    )
    parser.add_argument(
        "--aggregate-dir",
        type=Path,
        default=None,
        help="覆盖配置中的 OOF aggregate，用于同协议评估新模型",
    )
    parser.add_argument(
        "--formal-crop-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    destination = args.output_dir.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        logger.error("输出目录非空，禁止覆盖: %s", destination)
        return 1
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Y1 配置顶层必须是对象")
        project_config = Path(config["project_config"])
        protocol = parse_evaluation_protocol(load_config(project_config))
        if protocol.eval_version != "official_eval_v1":
            raise ValueError(f"eval_version 必须是 official_eval_v1: {protocol.eval_version}")
        grid = config["grid"]
        result = run_yolo_calibration_crossfit(
            aggregate_dir=args.aggregate_dir or config["aggregate_dir"],
            formal_crop_manifest_path=(args.formal_crop_manifest or config["formal_crop_manifest"]),
            protocol=protocol,
            expected_images=int(config["expected_images"]),
            expected_annotations=int(config["expected_annotations"]),
            candidate_floor=float(config["candidate_floor"]),
            threshold_start=float(grid["threshold_start"]),
            threshold_stop=float(grid["threshold_stop"]),
            threshold_step=float(grid["threshold_step"]),
            betas=tuple(float(value) for value in grid["betas"]),
            spatial_lambdas=tuple(float(value) for value in grid["spatial_lambdas"]),
            max_grid_size=int(grid["max_grid_size"]),
        )
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as error:
        logger.error("Y1 失败: %s", error)
        return 1
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "calibration_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        method: {
            "recall": payload["recall"],
            "fdr": payload["fdr"],
            "macro_recall": payload["official_ranking"]["overall_macro_recall"],
            "macro_fdr": payload["official_ranking"]["overall_macro_fdr"],
            "gate": payload["official_gate_passed"],
            "delta_vs_c0": payload["delta_vs_c0"],
        }
        for method, payload in result["merged_held_out"].items()
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    decision = build_decision(result)
    (destination / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for method, payload in summary.items():
        logger.info(
            "%s Recall=%.4f FDR=%.4f macroR=%.4f macroFDR=%.4f gate=%s",
            method,
            payload["recall"],
            payload["fdr"],
            payload["macro_recall"],
            payload["macro_fdr"],
            payload["gate"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
