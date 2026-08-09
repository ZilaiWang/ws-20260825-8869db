#!/usr/bin/env python3
"""N0-1：M1 正式 CV3 OOF 的 strict cross-fit 阈值基线。

用法（CPU 即可）：
    PYTHONPATH=src python scripts/n0_1_crossfit_thresholds.py \
        --aggregate outputs/M1-CV3-OOF-return-no-checkpoints-extracted-20260725/M1-CV3-OOF-aggregate \
        --formal-crop-manifest outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv \
        --output-dir outputs/N0-CROSSFIT-M1

规则：
- 每个 held-out fold 只用另外两折选全局阈值（网格同 M1 同 OOF 探索）。
- 内部目标：Recall >= 0.85 且 FDR <= 0.20 的最优工作点；同时报告
  FDR <= 0.17 的达成情况。
- 合并三份 held-out 评估得到无偏 baseline。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rsdet.analysis.crossfit_thresholds import run_crossfit
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="n0_1_crossfit")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="N0-1 strict cross-fit 阈值基线",
    )
    parser.add_argument(
        "--aggregate",
        type=Path,
        required=True,
        help="M1 OOF aggregate 目录（含 predictions_oof_low.json）",
    )
    parser.add_argument(
        "--formal-crop-manifest",
        type=Path,
        required=True,
        help="formal crop manifest CSV（GT 来源）",
    )
    parser.add_argument(
        "--project-config",
        type=Path,
        default=Path("configs/project.yaml"),
        help="冻结评估协议配置",
    )
    parser.add_argument(
        "--expected-images",
        type=int,
        default=4481,
        help="formal GT 图像数",
    )
    parser.add_argument(
        "--expected-annotations",
        type=int,
        default=20933,
        help="formal GT 对象数",
    )
    parser.add_argument(
        "--candidate-floor",
        type=float,
        default=0.001,
        help="OOF 聚合低阈值（校验）",
    )
    parser.add_argument(
        "--threshold-start",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--threshold-stop",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--internal-recall-min",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--internal-fdr-max",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--internal-fdr-strict",
        type=float,
        default=0.17,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出目录（非空禁止覆盖）",
    )
    return parser.parse_args(argv)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """运行 N0-1。"""
    args = parse_args(argv)
    try:
        protocol = parse_evaluation_protocol(load_config(args.project_config))
        if protocol.eval_version != "official_eval_v1":
            logger.error("只接受冻结 official_eval_v1，实际 %s", protocol.eval_version)
            return 1
        result = run_crossfit(
            aggregate_dir=args.aggregate,
            formal_crop_manifest_path=args.formal_crop_manifest,
            protocol=protocol,
            expected_images=args.expected_images,
            expected_annotations=args.expected_annotations,
            candidate_floor=args.candidate_floor,
            threshold_start=args.threshold_start,
            threshold_stop=args.threshold_stop,
            threshold_step=args.threshold_step,
            internal_recall_min=args.internal_recall_min,
            internal_fdr_max=args.internal_fdr_max,
            internal_fdr_strict=args.internal_fdr_strict,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        logger.error("N0-1 运行失败: %s", error)
        return 1

    destination = Path(args.output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        logger.error("输出目录非空，禁止覆盖: %s", destination)
        return 1
    destination.mkdir(parents=True, exist_ok=True)

    merged = result["merged_held_out"]
    logger.info("=== N0-1 cross-fit 结果 ===")
    logger.info(
        "合并 held-out: Recall %.4f / FDR %.4f (gate %s)",
        merged["recall"],
        merged["fdr"],
        merged["official_gate_passed"],
    )
    logger.info(
        "内部目标 FDR<=0.17: %s",
        merged["internal_fdr_passed"],
    )
    for per_fold in result["per_fold"]:
        logger.info(
            "fold %s: 选阈 %.4f → held-out Recall %.4f / FDR %.4f (gate %s)",
            per_fold["held_out_fold"],
            per_fold["selected_threshold"],
            per_fold["held_out_recall"],
            per_fold["held_out_fdr"],
            per_fold["held_out_gate_passed"],
        )
    logger.info(
        "阈值离散度: mean %.4f / std %.4f / spread %.4f",
        result["threshold_dispersion"]["mean"],
        result["threshold_dispersion"]["std"],
        result["threshold_dispersion"]["spread"],
    )

    _write_json(destination / "crossfit_result.json", result)
    logger.info("结果已保存: %s", destination / "crossfit_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
