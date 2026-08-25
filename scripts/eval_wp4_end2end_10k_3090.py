#!/usr/bin/env python3
"""E 主线 3/4：真实 M1 权重 + 3090 的 10K 端到端耗时实测（工程 smoke）。

对齐 docs/server/E_10K_PIPELINE_TASK.md 的冻结工作点与计时口径：

- 图像：synthetic.py 生成 10K 合成图（官方不提供真实 10K；synthetic 只能算
  工程 smoke，不能宣称官方时延，image_source_type="synthetic"）
- 冻结几何：tile 1280 / overlap 256 / stride 1024 → 恰好 100 tiles / batch 8
- 模型：M1Wrapper(yolo26s best.pt) @ CUDA；conf=0.001 / iou=0.7 / max_det=500
- 融合：fusion="global"（E 主线 3 全局聚合），fine_nms_iou=0.55 / coarse_nms_iou=0.85
- 硬门禁：每次 measured run 的 wall_after_read <= 20.0s（非均值）
- 计时：perf_counter + torch.cuda.synchronize（GPU 阶段前后）
- 输出：e2e_result.json（含 timing 拆分、聚合统计、权重 SHA、GPU）

用法：
    PYTHONPATH=src python scripts/eval_wp4_end2end_10k_3090.py \\
        --weights /root/autodl-tmp/M1/fold_0_best.pt \\
        --output outputs/e_wp3/e2e_result_fold0_3090.json
    PYTHONPATH=src python scripts/eval_wp4_end2end_10k_3090.py
        # 无 --weights = mock 检测器，仅验证脚本链路
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import rsdet.pipeline.mock_model  # noqa: F401  注册 mock
from rsdet.pipeline.large_image import PipelineConfig, run_pipeline
from rsdet.pipeline.m1_wrapper import M1Wrapper
from rsdet.models.registry import build_model
from rsdet.predictions import (
    predictions_to_coco_records,
    validate_coco_prediction_records,
)
from rsdet.tiling.synthetic import generate_synthetic_scene

HARD_GATE_S = 20.0


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=str, default="", help="真实 M1 best.pt；空=mock")
    ap.add_argument("--image-path", type=str, default="",
                    help="真实大图路径（PNG/TIF/JPG）。给定时加载真实图；否则生成合成图")
    ap.add_argument("--image-source-type", type=str, default="",
                    help="覆盖 image_source_type；默认：有图=real_project_proxy，无图=synthetic")
    ap.add_argument("--image-size", type=int, default=10000)
    ap.add_argument("--tile-size", type=int, default=1280)
    ap.add_argument("--overlap", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--runs", type=int, default=5, help="measured runs")
    ap.add_argument("--warmup", type=int, default=1, help="warmup runs（不计入）")
    ap.add_argument("--output", type=str, default="outputs/e_wp3/e2e_result.json")
    ap.add_argument("--coco", type=str, default="",
                    help="导出 COCO detection JSON 到该路径并跑统一校验（默认不导出）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    torch.cuda.synchronize()

    if args.image_path:
        from PIL import Image as PILImage

        img_path = Path(args.image_path)
        if not img_path.exists():
            raise SystemExit(f"图像不存在: {args.image_path}")
        with PILImage.open(img_path) as handle:
            image = np.asarray(handle.convert("RGB"))
        image = np.ascontiguousarray(image)
        image_source_type = args.image_source_type or "real_project_proxy"
        image_sha256 = sha256_file(str(img_path))
        n_gt = None
        print(f"[e2e] 真实图 {image.shape[1]}x{image.shape[0]} RGB, "
              f"sha256={image_sha256[:16]}… source={image_source_type}")
    else:
        scene = generate_synthetic_scene(
            image_size=args.image_size,
            tile_size=args.tile_size,
            overlap=args.overlap,
            num_ships=8,
            num_aircraft=20,
            num_vehicles=4,
            seed=args.seed,
        )
        image = scene.image
        image_source_type = args.image_source_type or "synthetic"
        image_sha256 = None
        n_gt = len(scene.objects)
        print(f"[e2e] 10K 图 {image.shape[1]}x{image.shape[0]} RGB, GT 目标 {n_gt}")

    if args.weights:
        if not Path(args.weights).exists():
            raise SystemExit(f"权重不存在: {args.weights}")
        detector = M1Wrapper(
            weights_path=args.weights, imgsz=1024, conf=0.001, iou=0.7, max_det=500
        )
        detector.load(args.weights)
        detector.to("cuda")
        detector.eval()
        model_key = "M1"
    else:
        detector = build_model("mock", {"init_args": {}})
        detector.eval()
        model_key = "mock"

    config = PipelineConfig(
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        fusion="global",
        score_threshold=0.0,
        fine_nms_iou=0.55,
        coarse_nms_iou=0.85,
        cluster_eps=50.0,
        merge_iou=0.3,
        nms_iou=0.5,
    )

    measured: list[dict[str, Any]] = []
    for i in range(args.warmup + args.runs):
        t0 = time.perf_counter()
        prediction, timing, objects = run_pipeline(
            image, detector, config=config, collect_objects=True
        )
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        row = {
            "run_index": i,
            "warmup": i < args.warmup,
            "model_key": model_key,
            "wall_after_read_s": round(wall, 4),
            "pipeline_s": round(timing.pipeline_s, 4),
            "model_only_s": round(timing.model_only_s, 4),
            "tiling_s": round(timing.tiling_s, 4),
            "fusion_s": round(timing.fusion_s, 4),
            "n_tiles": timing.n_tiles,
            "n_output_boxes": len(prediction.boxes_xyxy),
            "n_objects": len(objects),
            "peak_vram_gb": round(timing.peak_vram_gb, 3),
            "under_20s": wall <= HARD_GATE_S,
        }
        if not row["warmup"]:
            measured.append(row)
        print(
            f"[e2e] run {i}: {'warmup' if row['warmup'] else 'MEAS'} "
            f"wall={wall:6.2f}s pipeline={timing.pipeline_s:6.2f}s "
            f"model={timing.model_only_s:6.2f}s tiling={timing.tiling_s:6.2f}s "
            f"fusion={timing.fusion_s:6.2f}s tiles={timing.n_tiles} "
            f"objs={len(objects)} vram={timing.peak_vram_gb:.2f}GiB"
        )

    walls = sorted(r["wall_after_read_s"] for r in measured)

    # COCO detection JSON 导出 + 统一校验（分工 E 要求：输出字段/坐标格式/类别ID/分数可通过统一校验）
    coco_summary: dict[str, Any] | None = None
    if args.coco:
        # 用最后一次 measured run 的融合结果导出（全局坐标已恢复、重复已去重）
        records = predictions_to_coco_records(
            [prediction],
            allowed_category_ids=range(25),
        )
        coco_path = Path(args.coco)
        coco_path.parent.mkdir(parents=True, exist_ok=True)
        coco_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        check = validate_coco_prediction_records(
            records,
            allowed_category_ids=range(25),
            image_sizes={prediction.image_id: (int(image.shape[1]), int(image.shape[0]))},
        )
        coco_summary = {
            "path": str(coco_path),
            "detections": check["detections"],
            "images_with_predictions": check["images_with_predictions"],
            "categories_with_predictions": check["categories_with_predictions"],
            "validation": "passed",
        }
        print(f"[e2e] COCO 导出+统一校验通过: {coco_path} "
              f"detections={check['detections']} categories={check['categories_with_predictions']}")

    payload = {
        "contract_version": "e_10k_engineering_smoke_v1",
        "engineering_checkpoint_only": True,
        "image_source_type": image_source_type,
        "image_sha256": image_sha256,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "n_gt": n_gt,
        "hard_gate_total_after_read_seconds": HARD_GATE_S,
        "geometry": {
            "image_size": args.image_size,
            "tile_size": args.tile_size,
            "overlap": args.overlap,
            "stride": args.tile_size - args.overlap,
            "batch_size": args.batch_size,
            "fine_nms_iou": 0.55,
            "coarse_nms_iou": 0.85,
            "conf_threshold": 0.001,
        },
        "model": {"key": model_key, "weights": args.weights or None},
        "weights_sha256": (
            sha256_file(args.weights) if args.weights and Path(args.weights).exists() else None
        ),
        "timing_method": "perf_counter_with_torch_cuda_synchronize",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "warmup_runs": args.warmup,
        "measured_runs": len(measured),
        "samples": measured,
        "summary": {
            "total_after_read_s": {
                "p50": round(walls[len(walls) // 2], 4),
                "p95": round(float(np.percentile(walls, 95)), 4),
                "max": round(walls[-1], 4),
            },
            "all_under_20s": all(r["under_20s"] for r in measured),
            "max_peak_vram_gb": round(max(r["peak_vram_gb"] for r in measured), 3),
        },
        "coco_export": coco_summary,
        "note": (
            "工程 smoke：合成 10K + 工程 best.pt，仅验证管道与耗时预算；"
            "官方时延结论需 real_official 10K + 正式 last.pt + 独占 GPU（见 E_10K_PIPELINE_TASK.md）"
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[e2e] 结果已写: {args.output}")
    print(f"[e2e] total_after_read p50={payload['summary']['total_after_read_s']['p50']}s "
          f"p95={payload['summary']['total_after_read_s']['p95']}s "
          f"max={payload['summary']['total_after_read_s']['max']}s "
          f"all_under_20s={payload['summary']['all_under_20s']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
