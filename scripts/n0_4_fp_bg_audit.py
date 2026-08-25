#!/usr/bin/env python3
"""N0-4：FP_BG 人工语义审计抽检包生成。

用法（CPU 即可）：
    PYTHONPATH=src python scripts/n0_4_fp_bg_audit.py \
        --manifest outputs/N0-EVIDENCE-M1/pred_oof_evidence.json \
        --output-dir outputs/N0-FP-BG-AUDIT \
        --max-per-stratum 10 \
        --repeat-control-fraction 0.20 \
        --seed 42

产出：
- audit_samples.csv：人工标注表（label/labeler 空，待 B 盲审填写）
- audit_samples.json：抽检包（含汇总与样本明细）
- audit_review_guide.md：人工标注协议速查

分层：三大类 × 三折 × 分数分位（低/中/高），每层最多 max-per-stratum 条。
每批混入 repeat-control-fraction 比例的盲重复卡测人工一致性。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rsdet.analysis.fp_bg_audit import (
    audit_samples_to_csv,
    audit_samples_to_json,
    sample_fp_bg_audit,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="n0_4_fp_bg")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="N0-4 FP_BG 人工语义审计抽检包",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--max-per-stratum", type=int, default=10)
    parser.add_argument("--repeat-control-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """运行 N0-4。"""
    args = parse_args(argv)
    try:
        protocol = parse_evaluation_protocol(load_config(args.project_config))
        if protocol.eval_version != "official_eval_v1":
            logger.error("只接受冻结 official_eval_v1")
            return 1
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        category_mapping = {
            int(category_id): name for category_id, name in protocol.category_mapping.items()
        }
        samples, summary = sample_fp_bg_audit(
            manifest,
            category_mapping=category_mapping,
            max_per_stratum=args.max_per_stratum,
            repeat_control_fraction=args.repeat_control_fraction,
            seed=args.seed,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        logger.error("N0-4 运行失败: %s", error)
        return 1

    destination = Path(args.output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        logger.error("输出目录非空，禁止覆盖: %s", destination)
        return 1
    destination.mkdir(parents=True, exist_ok=True)

    csv_path = destination / "audit_samples.csv"
    json_path = destination / "audit_samples.json"
    audit_samples_to_csv(samples, csv_path)
    audit_samples_to_json(samples, json_path, summary=summary)

    logger.info("=== N0-4 FP_BG 抽检 ===")
    logger.info(
        "池大小 %d → 正卡 %d + 盲重复卡 %d = %d",
        summary["pool_size"],
        summary["sampled_positives"],
        summary["repeat_controls"],
        summary["total_samples"],
    )
    logger.info("分层单元数: %d（种子 %d）", summary["strata_sampled"], args.seed)
    logger.info("人工标注表: %s", csv_path)
    logger.info("抽检包: %s", json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
