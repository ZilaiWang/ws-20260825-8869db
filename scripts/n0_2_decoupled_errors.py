#!/usr/bin/env python3
"""N0-2：M1 正式 CV3 OOF 的定位 / 分类错误解耦分析。

用法（CPU 即可）：
    PYTHONPATH=src python scripts/n0_2_decoupled_errors.py \
        --aggregate outputs/M1-CV3-OOF-return-no-checkpoints-extracted-20260725/M1-CV3-OOF-aggregate \
        --formal-crop-manifest outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv \
        --groups data/splits/cv3_airport_proxy_k60_v2_groups.json \
        --threshold 0.051 \
        --output-dir outputs/N0-DECOUPLED-M1

产出：
- oracle 定位召回（R_loc@oracle-class，预测细类免费）
- 已定位对象上的细类准确率（Acc_fine@localized）
- 按来源组 / 尺寸档 / 细类分层
- source-group bootstrap 95% 区间
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
from rsdet.analysis.decoupled_errors import analyze_decoupled_errors
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="n0_2_decoupled")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="N0-2 定位/分类错误解耦",
    )
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--formal-crop-manifest", type=Path, required=True)
    parser.add_argument(
        "--groups",
        type=Path,
        default=None,
        help="来源组 JSON（可选；用于分层与 bootstrap）。支持两种结构："
        "{group_id: fold} 或 {image_id: group_id}；若为前者需配合 "
        "--group-id-key 与 --image-group-map-file 使用",
    )
    parser.add_argument(
        "--image-group-map-file",
        type=Path,
        default=None,
        help="可选的 image_id->group_id 映射 JSON（若 groups 只含 group->fold）",
    )
    parser.add_argument(
        "--project-config",
        type=Path,
        default=Path("configs/project.yaml"),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.051,
        help="用于解耦评估的预测 score 阈值（建议取 cross-fit 均值）",
    )
    parser.add_argument("--expected-images", type=int, default=4481)
    parser.add_argument("--expected-annotations", type=int, default=20933)
    parser.add_argument("--candidate-floor", type=float, default=0.001)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """运行 N0-2。"""
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

        group_of_image: dict[int, str] | None = None
        if args.groups is not None:
            groups_data = json.loads(args.groups.read_text(encoding="utf-8"))
            if isinstance(groups_data, dict) and "groups" in groups_data:
                # cv3_airport_proxy_k60_v2_groups.json: {group_id: fold}
                if args.image_group_map_file is not None:
                    map_data = json.loads(args.image_group_map_file.read_text(encoding="utf-8"))
                    if isinstance(map_data, dict):
                        raw_map = {int(key): str(value) for key, value in map_data.items()}
                    elif isinstance(map_data, list):
                        raw_map = {
                            int(item["image_id"]): str(item["group_id"])
                            for item in map_data
                            if "image_id" in item and "group_id" in item
                        }
                    else:
                        raise ValueError("image-group-map 文件必须是对象或列表")
                    group_of_image = raw_map
                else:
                    raise ValueError(
                        "cv3 groups 文件只含 group->fold，需配合 "
                        "--image-group-map-file 提供 image->group 映射"
                    )
            elif isinstance(groups_data, dict):
                raw_map = {int(key): str(value) for key, value in groups_data.items()}
                group_of_image = raw_map
            elif isinstance(groups_data, list):
                group_of_image = {
                    int(item["image_id"]): str(item["group_id"])
                    for item in groups_data
                    if "image_id" in item and "group_id" in item
                }
            else:
                raise ValueError("groups 文件格式必须是对象或列表")
            if len(group_of_image) != args.expected_images:
                logger.warning(
                    "来源组映射覆盖 %d 张图（预期 %d），缺失图将被标记 unknown",
                    len(group_of_image),
                    args.expected_images,
                )

        result = analyze_decoupled_errors(
            gt_boxes=gt_boxes,
            predictions=predictions,
            protocol=protocol,
            threshold=args.threshold,
            group_of_image=group_of_image,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        logger.error("N0-2 运行失败: %s", error)
        return 1

    destination = Path(args.output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        logger.error("输出目录非空，禁止覆盖: %s", destination)
        return 1
    destination.mkdir(parents=True, exist_ok=True)

    logger.info("=== N0-2 定位/分类解耦 ===")
    logger.info("oracle 定位召回 (R_loc@oracle-class): %.4f", result["oracle_localization_recall"])
    logger.info(
        "已定位对象细类准确率 (Acc_fine@localized): %.4f", result["localized_fine_accuracy"]
    )
    if result["source_group_bootstrap"] is not None:
        boot = result["source_group_bootstrap"]
        logger.info(
            "source-group bootstrap 95%%: [%.4f, %.4f]（%d 组）",
            boot["ci_low"],
            boot["ci_high"],
            boot["n_groups"],
        )
    for scope, counts in sorted(result["stratified"].items()):
        if scope.startswith("group_"):
            continue
        logger.info(
            "  %s: n=%d recall=%.4f",
            scope,
            counts["n_objects"],
            counts["recall"],
        )

    _write_json(destination / "decoupled_result.json", result)
    logger.info("结果已保存: %s", destination / "decoupled_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
