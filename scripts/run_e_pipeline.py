#!/usr/bin/env python3
"""E 的大图推理 CLI 入口。

模式:
    synthetic  — 生成合成 10K 图 → 跑 pipeline → 输出 COCO JSON → (可选) 评测
    image      — 读真实大图 → 跑 pipeline → 输出 COCO JSON
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import rsdet.pipeline.mock_model  # noqa: F401 — 注册 "mock" 检测器
from rsdet.contracts import Prediction, TileRecord
from rsdet.models.base import BaseDetector
from rsdet.models.registry import build_model, list_models
from rsdet.pipeline.large_image import PipelineConfig, run_pipeline
from rsdet.tiling.synthetic import SyntheticObject, generate_synthetic_scene
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="run_e_pipeline")


def _build_detector(model_name: str, **kwargs: Any) -> BaseDetector:
    available = list(list_models().keys())
    if model_name not in available:
        logger.error("模型 '%s' 未注册。可用: %s", model_name, available)
        sys.exit(1)
    return build_model(model_name, {"init_args": kwargs})


def _tile_metadata_for_mock(
    scene_objects: List[SyntheticObject],
) -> Any:
    """返回一个闭包：对每个 tile，查找落在其中的真值框（局部坐标）。"""

    def _fn(tile: TileRecord) -> Dict[str, Any]:
        gt_boxes: List[Dict[str, Any]] = []
        for obj in scene_objects:
            if tile.tile_id not in obj.tile_ids:
                continue
            gx1, gy1, gx2, gy2 = obj.bbox
            # 转到 tile 局部坐标并裁剪到边界内
            lx1 = max(0.0, gx1 - tile.x_offset)
            ly1 = max(0.0, gy1 - tile.y_offset)
            lx2 = min(float(tile.width), gx2 - tile.x_offset)
            ly2 = min(float(tile.height), gy2 - tile.y_offset)
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


def _coco_from_prediction(
    prediction: Prediction,
    image_id: int = 0,
) -> List[Dict[str, Any]]:
    """将单个 Prediction 转为 COCO detection 列表。"""
    records: List[Dict[str, Any]] = []
    for box, score, label in zip(prediction.boxes_xyxy, prediction.scores, prediction.labels):
        x1, y1, x2, y2 = [float(v) for v in box]
        records.append(
            {
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            }
        )
    return records


def mode_synthetic(args: argparse.Namespace) -> int:
    """合成模式：生成图 → pipeline → COCO → 评测。"""
    logger.info(
        "生成合成图: %dx%d, tile=%d, overlap=%d, seed=%d",
        args.image_size,
        args.image_size,
        args.tile_size,
        args.overlap,
        args.seed,
    )
    scene = generate_synthetic_scene(
        image_size=args.image_size,
        tile_size=args.tile_size,
        overlap=args.overlap,
        num_ships=args.num_ships,
        num_aircraft=args.num_aircraft,
        num_vehicles=args.num_vehicles,
        seed=args.seed,
    )
    logger.info(
        "合成图生成完成: %d 目标, 分布在 %d tiles 中",
        len(scene.objects),
        len({tid for obj in scene.objects for tid in obj.tile_ids}),
    )

    detector = _build_detector(args.model)
    detector.eval()

    config = PipelineConfig(
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        score_threshold=args.score_threshold,
    )

    logger.info("开始 pipeline ...")
    prediction, timing = run_pipeline(
        scene.image,
        detector,
        config=config,
        parent_image_id=scene.image_id,
        tile_metadata_fn=_tile_metadata_for_mock(scene.objects),
    )

    logger.info("Pipeline 完成: %s", json.dumps(timing.to_dict(), indent=2))
    logger.info("检测数: %d", timing.n_detections)

    # 输出 COCO JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = _coco_from_prediction(prediction, image_id=scene.image_id)
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("预测已保存: %s", output_path)

    # 输出 GT
    if args.save_gt:
        gt_path = output_path.parent / "gt.json"
        scene.save_gt(gt_path)
        logger.info("GT 已保存: %s", gt_path)

    # 可选评测
    if args.evaluate:
        gt_path = output_path.parent / "gt.json"
        if not gt_path.exists():
            scene.save_gt(gt_path)
        _run_evaluation(gt_path, output_path, args)

    return 0


def mode_image(args: argparse.Namespace) -> int:
    """真实图像模式：读图 → pipeline → COCO。"""
    try:
        from PIL import Image
    except ImportError:
        logger.error("需要 Pillow 才能读图: pip install Pillow")
        return 1

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error("图像不存在: %s", image_path)
        return 1

    t0 = time.perf_counter()
    pil_img = Image.open(image_path).convert("RGB")
    image = np.array(pil_img, dtype=np.uint8)
    disk_read_s = time.perf_counter() - t0

    detector = _build_detector(args.model)
    detector.eval()

    config = PipelineConfig(
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        score_threshold=args.score_threshold,
    )

    logger.info("开始 pipeline (image 模式) ...")
    prediction, timing = run_pipeline(
        image,
        detector,
        config=config,
        parent_image_id=0,
    )

    logger.info("读图耗时: %.3fs", disk_read_s)
    logger.info("Pipeline 完成: %s", json.dumps(timing.to_dict(), indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = _coco_from_prediction(prediction, image_id=0)
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("预测已保存: %s", output_path)
    return 0


def _run_evaluation(
    gt_path: Path,
    pred_path: Path,
    args: argparse.Namespace,
) -> None:
    """调用 A 的评测器。"""
    from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
    from rsdet.evaluation.official_metric import evaluate_predictions
    from rsdet.evaluation.protocol import parse_evaluation_protocol
    from rsdet.utils.config import load_config

    project_config = load_config(args.project_config)
    protocol = parse_evaluation_protocol(project_config)
    gt_boxes = load_coco_ground_truth(gt_path)
    pred_boxes = load_coco_predictions(pred_path)
    result = evaluate_predictions(
        gt_boxes,
        pred_boxes,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )

    passed = result.recall >= protocol.recall_min and result.fdr <= protocol.fdr_max
    logger.info("Overall Recall: %.4f", result.recall)
    logger.info("Overall FDR:    %.4f", result.fdr)
    logger.info("硬指标: %s", "PASS" if passed else "FAIL")
    for name, metrics in result.per_class.items():
        logger.info(
            "%s: Recall=%.4f, FDR=%.4f, TP=%d, FP=%d, FN=%d",
            name,
            metrics.recall,
            metrics.fdr,
            metrics.tp,
            metrics.fp,
            metrics.fn,
        )

    eval_path = args.output.replace(".json", "_eval.json")
    Path(eval_path).parent.mkdir(parents=True, exist_ok=True)
    Path(eval_path).write_text(
        json.dumps(
            {
                "overall_recall": result.recall,
                "overall_fdr": result.fdr,
                "per_class": {
                    name: {
                        "recall": m.recall,
                        "fdr": m.fdr,
                        "tp": m.tp,
                        "fp": m.fp,
                        "fn": m.fn,
                    }
                    for name, m in result.per_class.items()
                },
                "passed": passed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("评测结果已保存: %s", eval_path)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E: 大图推理 pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 合成模式（mock 模型 + 评测）
  python scripts/run_e_pipeline.py --mode synthetic --evaluate

  # 真实图像模式
  python scripts/run_e_pipeline.py --mode image --image path/to/big.png
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "image"],
        default="synthetic",
        help="运行模式",
    )
    parser.add_argument(
        "--model",
        default="mock",
        help="注册的模型名 (mock / dummy / 后续 M1/M3 adapter)",
    )

    # 图像参数
    parser.add_argument("--image", default=None, help="真实图像路径 (image 模式)")
    parser.add_argument("--image-size", type=int, default=10000, help="合成图边长 (synthetic 模式)")

    # Pipeline 参数
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--score-threshold", type=float, default=0.0)

    # 合成目标参数
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-ships", type=int, default=10)
    parser.add_argument("--num-aircraft", type=int, default=30)
    parser.add_argument("--num-vehicles", type=int, default=5)

    # 输出
    parser.add_argument(
        "--output",
        default="outputs/e_pipeline/predictions.json",
        help="预测 COCO JSON 输出路径",
    )
    parser.add_argument("--save-gt", action="store_true", help="同时保存 GT JSON")
    parser.add_argument("--evaluate", action="store_true", help="跑完自动评测")

    # 配置
    parser.add_argument(
        "--project-config",
        default="configs/project.yaml",
        help="项目配置文件",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.mode == "synthetic":
        return mode_synthetic(args)
    elif args.mode == "image":
        return mode_image(args)
    else:
        logger.error("未知模式: %s", args.mode)
        return 1


if __name__ == "__main__":
    sys.exit(main())
