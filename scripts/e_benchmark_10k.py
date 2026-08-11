#!/usr/bin/env python3
"""E 的 10K pipeline 多参数测速。

用法:
  python scripts/e_benchmark_10k.py --warmup 2 --repeats 10

输出每组 (tile_size, overlap, batch_size) 的 p50/p95 耗时、tile 数、检测数、显存。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import rsdet.pipeline.mock_model  # noqa: F401 — 注册 "mock" 检测器
from rsdet.models.base import BaseDetector
from rsdet.models.registry import build_model
from rsdet.pipeline.large_image import PipelineConfig, PipelineTiming, run_pipeline
from rsdet.tiling.synthetic import generate_synthetic_scene
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="e_benchmark_10k")


def _build_detector(model_name: str, **kwargs: Any) -> BaseDetector:
    return build_model(model_name, {"init_args": kwargs})


def _tile_metadata_for_mock(scene) -> Any:
    """动态计算每个 tile 包含哪些合成目标（不依赖 obj.tile_ids）。

    这样可以支持 benchmark 中用同一场景测试不同 tile_size/overlap 组合。
    """
    from rsdet.contracts import TileRecord

    def _fn(tile: TileRecord) -> Dict[str, Any]:
        gt_boxes: List[Dict[str, Any]] = []
        tx1, ty1 = float(tile.x_offset), float(tile.y_offset)
        tx2, ty2 = tx1 + tile.width, ty1 + tile.height
        for obj in scene.objects:
            gx1, gy1, gx2, gy2 = obj.bbox
            # 检查对象是否与 tile 有重叠（至少 30% 可见）
            ix1 = max(gx1, tx1)
            iy1 = max(gy1, ty1)
            ix2 = min(gx2, tx2)
            iy2 = min(gy2, ty2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            area = (gx2 - gx1) * (gy2 - gy1)
            if inter / area <= 0.3:
                continue
            # 转到 tile 局部坐标并裁剪
            lx1 = max(0.0, gx1 - tx1)
            ly1 = max(0.0, gy1 - ty1)
            lx2 = min(float(tile.width), gx2 - tx1)
            ly2 = min(float(tile.height), gy2 - ty1)
            if lx2 <= lx1 or ly2 <= ly1:
                continue
            gt_boxes.append(
                {
                    "bbox": [lx1, ly1, lx2, ly2],
                    "category_id": obj.category_id,
                    "score": 1.0,
                }
            )
        return {"gt_boxes": gt_boxes}

    return _fn


def run_single(
    detector: BaseDetector,
    scene,
    config: PipelineConfig,
) -> PipelineTiming:
    """跑一次完整 pipeline 并返回计时。"""
    _, timing = run_pipeline(
        scene.image,
        detector,
        config=config,
        tile_metadata_fn=_tile_metadata_for_mock(scene),
    )
    return timing


def benchmark_configs(
    detector: BaseDetector,
    scene,
    configs: List[PipelineConfig],
    warmup: int = 2,
    repeats: int = 10,
) -> List[Dict[str, Any]]:
    """对多组参数依次测速。"""
    results: List[Dict[str, Any]] = []

    for idx, config in enumerate(configs):
        logger.info(
            "测速 [%d/%d]: tile=%d overlap=%d batch=%d",
            idx + 1,
            len(configs),
            config.tile_size,
            config.overlap,
            config.batch_size,
        )

        # 预热
        for w in range(warmup):
            logger.debug("  预热 %d/%d", w + 1, warmup)
            run_single(detector, scene, config)

        # 正式计时
        timings: List[PipelineTiming] = []
        for r in range(repeats):
            logger.debug("  第 %d/%d 次", r + 1, repeats)
            timings.append(run_single(detector, scene, config))

        pipeline_ts = [t.pipeline_s for t in timings]
        model_ts = [t.model_only_s for t in timings]
        tiling_ts = [t.tiling_s for t in timings]
        fusion_ts = [t.fusion_s for t in timings]

        def _p(arr):
            return round(float(np.percentile(arr, 50)), 4)

        def _p95(arr):
            return round(float(np.percentile(arr, 95)), 4)

        results.append(
            {
                "config": {
                    "tile_size": config.tile_size,
                    "overlap": config.overlap,
                    "batch_size": config.batch_size,
                },
                "pipeline_p50": _p(pipeline_ts),
                "pipeline_p95": _p95(pipeline_ts),
                "model_only_p50": _p(model_ts),
                "model_only_p95": _p95(model_ts),
                "tiling_p50": _p(tiling_ts),
                "fusion_p50": _p(fusion_ts),
                "n_tiles": timings[0].n_tiles,
                "n_detections": timings[0].n_detections,
                "peak_vram_gb": max(t.peak_vram_gb for t in timings),
            }
        )

        logger.info(
            "  → pipeline p50=%.4fs  p95=%.4fs  tiles=%d",
            results[-1]["pipeline_p50"],
            results[-1]["pipeline_p95"],
            results[-1]["n_tiles"],
        )

    return results


def parse_configs(raw: List[str]) -> List[PipelineConfig]:
    """解析 'tile,overlap,batch' 格式的配置列表。"""
    configs = []
    for entry in raw:
        parts = entry.replace(" ", "").split(",")
        if len(parts) != 3:
            raise ValueError(f"配置格式错误，应为 'tile,overlap,batch': {entry}")
        configs.append(
            PipelineConfig(
                tile_size=int(parts[0]),
                overlap=int(parts[1]),
                batch_size=int(parts[2]),
            )
        )
    return configs


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E: 10K pipeline 多参数测速",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/e_benchmark_10k.py --warmup 2 --repeats 10
  python scripts/e_benchmark_10k.py --configs "1024,128,16" "1024,256,8" "2048,256,16"
        """,
    )
    parser.add_argument("--warmup", type=int, default=2, help="每组预热次数")
    parser.add_argument("--repeats", type=int, default=10, help="每组计时重复次数")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=["1024,128,16", "1024,256,8", "2048,256,16"],
        help="测速参数组: 'tile,overlap,batch'",
    )
    parser.add_argument("--model", default="mock", help="注册的模型名")
    parser.add_argument("--image-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="outputs/e_benchmark/result.json",
        help="测速结果 JSON",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    configs = parse_configs(args.configs)
    logger.info("测速配置: %d 组, warmup=%d, repeats=%d", len(configs), args.warmup, args.repeats)

    logger.info("生成合成图 ...")
    scene = generate_synthetic_scene(
        image_size=args.image_size,
        tile_size=max(c.tile_size for c in configs),
        overlap=max(c.overlap for c in configs),
        seed=args.seed,
    )
    logger.info("合成图: %d 目标", len(scene.objects))

    detector = _build_detector(args.model)
    detector.eval()

    results = benchmark_configs(
        detector,
        scene,
        configs,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("测速结果已保存: %s", output_path)

    # 打印汇总表
    print("\n" + "=" * 70)
    print(
        f"{'tile':>6} {'ovlp':>5} {'batch':>6}  {'p50(s)':>8} {'p95(s)':>8} {'tiles':>6} {'dets':>6}"
    )
    print("-" * 70)
    for r in results:
        c = r["config"]
        print(
            f"{c['tile_size']:>6} {c['overlap']:>5} {c['batch_size']:>6}  "
            f"{r['pipeline_p50']:>8.4f} {r['pipeline_p95']:>8.4f} "
            f"{r['n_tiles']:>6} {r['n_detections']:>6}"
        )
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
