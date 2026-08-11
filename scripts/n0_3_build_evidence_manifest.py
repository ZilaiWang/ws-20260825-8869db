#!/usr/bin/env python3
"""N0-3：构建 Pred-OOF 对象证据 manifest（统一对象证据层）。

用法（CPU 即可）：
    PYTHONPATH=src python scripts/n0_3_build_evidence_manifest.py \
        --aggregate outputs/M1-CV3-OOF-return-no-checkpoints-extracted-20260725/M1-CV3-OOF-aggregate \
        --formal-crop-manifest outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv \
        --groups data/splits/cv3_airport_proxy_k60_v2_groups.json \
        --image-group-map outputs/N0-IMG2GROUP.json \
        --threshold 0.051 \
        --output-dir outputs/N0-EVIDENCE-M1

产出：
- pred_oof_evidence.json：统一对象证据 manifest（records + 三种视图 + summary）
- manifest.sha256：不可变校验和（下游模块按此引用）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rsdet.analysis.crossfit_thresholds import (
    load_cv3_aggregate,
    load_gt_from_formal_crop_manifest,
)
from rsdet.analysis.object_evidence import (
    build_object_evidence_manifest,
    manifest_sha256,
    write_manifest_json,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="n0_3_evidence")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="N0-3 Pred-OOF 对象证据 manifest",
    )
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--formal-crop-manifest", type=Path, required=True)
    parser.add_argument(
        "--groups",
        type=Path,
        default=None,
        help="cv3 划分来源组 JSON（{group_id: fold}，可选）",
    )
    parser.add_argument(
        "--image-group-map",
        type=Path,
        default=None,
        help="image_id->group_id 映射 JSON（可选，用于 source_group 填充）",
    )
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.051,
        help="工作点 score 阈值（建议取 cross-fit 均值或逐折阈值）",
    )
    parser.add_argument("--expected-images", type=int, default=4481)
    parser.add_argument("--expected-annotations", type=int, default=20933)
    parser.add_argument("--candidate-floor", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _load_image_groups(path: Path) -> dict[int, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {int(key): str(value) for key, value in data.items()}
    if isinstance(data, list):
        return {
            int(item["image_id"]): str(item["group_id"])
            for item in data
            if "image_id" in item and "group_id" in item
        }
    raise ValueError("image-group-map 必须是对象或列表")


def main(argv: list[str] | None = None) -> int:
    """运行 N0-3。"""
    args = parse_args(argv)
    try:
        protocol = parse_evaluation_protocol(load_config(args.project_config))
        if protocol.eval_version != "official_eval_v1":
            logger.error("只接受冻结 official_eval_v1")
            return 1

        metadata, predictions, image_folds = load_cv3_aggregate(
            args.aggregate,
            candidate_floor=args.candidate_floor,
        )
        gt_boxes = load_gt_from_formal_crop_manifest(
            args.formal_crop_manifest,
            expected_images=args.expected_images,
            expected_annotations=args.expected_annotations,
        )
        checkpoint_sha256 = {}
        for fold in metadata.get("folds", []):
            fold_index = int(fold["fold"])
            fold_ckpt = str(fold["checkpoint_sha256"])
            for image_id, fold_id in image_folds.items():
                if fold_id == fold_index:
                    checkpoint_sha256[image_id] = fold_ckpt

        image_groups = None
        if args.image_group_map is not None:
            image_groups = _load_image_groups(args.image_group_map)
            if len(image_groups) != args.expected_images:
                logger.warning(
                    "来源组映射覆盖 %d 张图（预期 %d）",
                    len(image_groups),
                    args.expected_images,
                )

        manifest = build_object_evidence_manifest(
            gt_boxes=gt_boxes,
            predictions=predictions,
            protocol=protocol,
            threshold=args.threshold,
            image_folds=image_folds,
            checkpoint_sha256=checkpoint_sha256,
            image_groups=image_groups,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        logger.error("N0-3 运行失败: %s", error)
        return 1

    destination = Path(args.output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        logger.error("输出目录非空，禁止覆盖: %s", destination)
        return 1
    destination.mkdir(parents=True, exist_ok=True)

    manifest_path = destination / "pred_oof_evidence.json"
    write_manifest_json(manifest, manifest_path)
    sha = manifest_sha256(manifest_path)
    (destination / "manifest.sha256").write_text(sha + "\n", encoding="utf-8")

    summary = manifest["summary"]
    logger.info("=== N0-3 对象证据 manifest ===")
    logger.info(
        "候选总数: %d | TP %d | FP %d",
        summary["total_candidates"],
        summary["official_tp"],
        summary["official_fp"],
    )
    logger.info("FP 类型: %s", summary["fp_by_type"])
    logger.info("视图计数: %s", summary["view_counts"])
    logger.info(
        "官方指标: Recall %.4f / FDR %.4f", summary["official_recall"], summary["official_fdr"]
    )
    logger.info("manifest SHA256: %s", sha)
    logger.info("已保存: %s", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
