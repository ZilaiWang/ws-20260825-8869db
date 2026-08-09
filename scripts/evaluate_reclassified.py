#!/usr/bin/env python3
"""N2-2：重分类后的消融评估（cross-fit 阈值）。

用法（CPU 即可）：
    PYTHONPATH=src python scripts/evaluate_reclassified.py \
        --aggregate outputs/M1-CV3-OOF-return-no-checkpoints-extracted-20260725/M1-CV3-OOF-aggregate \
        --formal-crop-manifest outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv \
        --reclassified outputs/N2-RECLASS/fold0.json --fold 0 \
        --reclassified outputs/N2-RECLASS/fold1.json --fold 1 \
        --reclassified outputs/N2-RECLASS/fold2.json --fold 2 \
        --mode reclassify \
        --output-dir outputs/N2-EVAL-RECLASSIFY

产出：cross-fit 工作点 + 与 M1 基线对照。供 N2-3 准入门槛判断。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from rsdet.analysis.crossfit_thresholds import (
    load_cv3_aggregate,
    load_gt_from_formal_crop_manifest,
    run_crossfit,
)
from rsdet.analysis.proposal_reclassification import (
    MODE_BACKGROUND,
    MODE_JOINT,
    MODE_RECLASSIFY,
    build_reclassified_predictions,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="n2_eval")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="N2-2 重分类消融评估")
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--formal-crop-manifest", type=Path, required=True)
    parser.add_argument("--reclassified", action="append", required=True)
    parser.add_argument("--fold", action="append", type=int, required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(MODE_RECLASSIFY, MODE_BACKGROUND, MODE_JOINT),
    )
    parser.add_argument(
        "--project-config", type=Path, default=Path("configs/project.yaml")
    )
    parser.add_argument("--expected-images", type=int, default=4481)
    parser.add_argument("--expected-annotations", type=int, default=20933)
    parser.add_argument("--candidate-floor", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """运行 N2-2 消融评估。"""
    args = parse_args(argv)
    if len(args.reclassified) != len(args.fold):
        logger.error("--reclassified 与 --fold 数量必须一致")
        return 1
    try:
        protocol = parse_evaluation_protocol(load_config(args.project_config))
        if protocol.eval_version != "official_eval_v1":
            logger.error("只接受冻结 official_eval_v1")
            return 1
        metadata, predictions, image_folds = load_cv3_aggregate(
            args.aggregate,
            candidate_floor=args.candidate_floor,
        )

        # 融合重分类结果（逐折加载，按 fold 合并到 override_predictions）。
        merged = dict(predictions)
        for path, fold in zip(args.reclassified, args.fold):
            reclassified = json.loads(Path(path).read_text(encoding="utf-8"))
            fold_predictions = {
                image_id: records
                for image_id, records in predictions.items()
                if image_folds.get(image_id) == fold
            }
            updated = build_reclassified_predictions(
                oof_predictions=fold_predictions,
                reclassified=reclassified,
                mode=args.mode,
            )
            for image_id, records in updated.items():
                merged[image_id] = records

        result = run_crossfit(
            aggregate_dir=args.aggregate,
            formal_crop_manifest_path=args.formal_crop_manifest,
            protocol=protocol,
            expected_images=args.expected_images,
            expected_annotations=args.expected_annotations,
            candidate_floor=args.candidate_floor,
            override_predictions=merged,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        logger.error("评估失败: %s", error)
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        logger.error("输出目录非空，禁止覆盖: %s", output_dir)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    result["mode"] = args.mode
    result["analysis"] = f"N2-2_{args.mode}"
    (output_dir / "crossfit_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("已保存: %s", output_dir / "crossfit_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
