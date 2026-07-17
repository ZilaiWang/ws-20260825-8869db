#!/usr/bin/env python3
"""P0-2 探索性对象 crop manifest 生成入口。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rsdet.analysis.crop_manifest import run_crop_manifest_analysis
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="rsdet.crop_manifest")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成保守防泄漏、可复现的探索性对象 crop manifest",
    )
    parser.add_argument(
        "--bbox-statistics",
        type=Path,
        required=True,
        help="dataset_audit/machine_readable/bbox_statistics.csv",
    )
    parser.add_argument(
        "--image-stats",
        type=Path,
        required=True,
        help="dataset_audit/machine_readable/image_stats.csv",
    )
    parser.add_argument(
        "--split-candidates",
        type=Path,
        required=True,
        help="dataset_audit/machine_readable/proposed_group_split.csv",
    )
    parser.add_argument(
        "--domain-assignments",
        type=Path,
        required=True,
        help="dataset_audit/machine_readable/domain_cluster_assignments.csv",
    )
    parser.add_argument(
        "--near-duplicates",
        type=Path,
        required=True,
        help="dataset_audit/machine_readable/near_duplicate_groups.csv",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="包含 images/train 和 images/val 的原始数据根目录",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/analysis/exploratory_crop_manifest.yaml"),
        help="crop 几何、防泄漏和渲染契约",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/P0-2-exploratory-crop-manifest"),
        help="manifest、审计表、QA 预览和报告输出目录",
    )
    parser.add_argument(
        "--skip-source-checksums",
        action="store_true",
        help="仅跳过源图 SHA-256 重算；仍校验全部文件存在",
    )
    parser.add_argument(
        "--skip-previews",
        action="store_true",
        help="跳过每类一例的几何 QA 联系表",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="跳过无依赖 SVG 图表，不影响数值结果",
    )
    parser.add_argument("--verbose", action="store_true", help="显示详细错误堆栈")
    return parser.parse_args(argv)


def _add_file_log(output_dir: Path) -> logging.FileHandler:
    output_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return handler


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    handler = _add_file_log(args.output_dir)
    try:
        logger.info("开始 P0-2 探索性 crop manifest 生成")
        logger.info("数据根目录: %s", args.data_root)
        logger.info("配置: %s", args.config)
        metadata = run_crop_manifest_analysis(
            bbox_statistics_path=args.bbox_statistics,
            image_stats_path=args.image_stats,
            split_candidates_path=args.split_candidates,
            domain_assignments_path=args.domain_assignments,
            near_duplicates_path=args.near_duplicates,
            data_root=args.data_root,
            config_path=args.config,
            output_dir=args.output_dir,
            command=[sys.executable, __file__, *(argv if argv is not None else sys.argv[1:])],
            repo_root=Path(__file__).resolve().parent.parent,
            verify_source_checksums=not args.skip_source_checksums,
            generate_previews=not args.skip_previews,
            write_charts=not args.skip_figures,
        )
        validation = metadata["validation"]
        logger.info(
            "完成: %d 张源图，%d 个对象，%d 行 manifest，%d 个 leakage group",
            validation["n_source_images"],
            validation["n_annotations"],
            validation["n_manifest_rows"],
            metadata["summary"]["leakage_groups"],
        )
        logger.info(
            "划分修复: main split 移动 %d 张，fold 移动 %d 张",
            validation["main_split_moved_images"],
            validation["fold_moved_images"],
        )
        logger.info("完整报告: %s", args.output_dir / "report.md")
        return 0
    except Exception:
        if args.verbose:
            logger.exception("生成失败")
        else:
            logger.error("生成失败；使用 --verbose 查看详细堆栈", exc_info=False)
        return 1
    finally:
        logger.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    sys.exit(main())
