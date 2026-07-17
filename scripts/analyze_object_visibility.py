#!/usr/bin/env python3
"""对象尺度与 feature-grid 几何可见性审计入口。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rsdet.analysis.object_visibility import run_visibility_analysis
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="rsdet.object_visibility")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="计算完整 bbox 在 tile/crop 预处理后的连续 feature-grid 跨度",
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
        "--config",
        type=Path,
        default=Path("configs/analysis/object_visibility.yaml"),
        help="场景、表征网格和预注册阈值配置",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/X-CROP-00-token-visibility"),
        help="数值产物、图表和报告输出目录",
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
        logger.info("开始对象几何可见性审计")
        logger.info("bbox 统计: %s", args.bbox_statistics)
        logger.info("图像统计: %s", args.image_stats)
        logger.info("场景配置: %s", args.config)
        metadata = run_visibility_analysis(
            bbox_path=args.bbox_statistics,
            image_stats_path=args.image_stats,
            config_path=args.config,
            output_dir=args.output_dir,
            command=[sys.executable, __file__, *(argv if argv is not None else sys.argv[1:])],
            repo_root=Path(__file__).resolve().parent.parent,
            write_charts=not args.skip_figures,
        )
        validation = metadata["validation"]
        logger.info(
            "完成: %d 个 bbox x %d 个场景 = %d 行明细，边界风险框 %d 个",
            validation["n_boxes"],
            metadata["scenario_count"],
            metadata["detail_row_count"],
            validation["n_source_edge_risk"],
        )
        logger.info("完整报告: %s", args.output_dir / "report.md")
        return 0
    except Exception:
        if args.verbose:
            logger.exception("分析失败")
        else:
            logger.error("分析失败；使用 --verbose 查看详细堆栈", exc_info=False)
        return 1
    finally:
        logger.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    sys.exit(main())
