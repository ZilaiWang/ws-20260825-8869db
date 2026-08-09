#!/usr/bin/env python3
"""Benchmark the complete BHC-DETR pipeline after image reading."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rsdet.engine.inference import predict_image
from rsdet.evaluation.runtime import RuntimeBreakdown
from rsdet.models.registry import build_model
from rsdet.predictions import predictions_to_coco_records
from rsdet.utils.config import load_config
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="benchmark")


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return dict(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BHC-DETR 10K 完整流水线测速")
    parser.add_argument("--config", type=Path, required=True, help="推理配置")
    parser.add_argument("--image", type=Path, default=None, help="真实 10000x10000 图像")
    parser.add_argument("--warmup", type=int, default=3, help="预热次数")
    parser.add_argument("--runs", type=int, default=10, help="正式重复次数")
    parser.add_argument(
        "--image-size",
        type=int,
        default=10000,
        help="未提供 --image 时的零像素工程代理尺寸",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="覆盖 checkpoint")
    parser.add_argument("--device", type=str, default=None, help="覆盖设备")
    parser.add_argument("--output", type=Path, default=None, help="测速 JSON")
    parser.add_argument(
        "--official-hardware-claim",
        action="store_true",
        help="仅在确认设备满足评分方案的 RTX 3090/等效硬件条件时使用",
    )
    return parser.parse_args(argv)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.is_file() or args.warmup < 0 or args.runs <= 0:
        logger.error("配置、warmup 或 runs 非法")
        return 1
    try:
        config = load_config(args.config)
        model_config = _mapping(config.get("model"), "model")
        adapter = str(model_config.pop("adapter", "bhcdetr"))
        if adapter != "bhcdetr":
            raise ValueError("测速只允许论文模型 bhcdetr")
        # Pop constructor-external fields even when a CLI override wins; an
        # ``override or pop(...)`` expression would short-circuit and leak the
        # YAML checkpoint into BHCDetrDetector.__init__.
        configured_checkpoint = model_config.pop("checkpoint", None)
        checkpoint = args.checkpoint or configured_checkpoint or config.get(
            "checkpoint"
        )
        if not checkpoint:
            raise ValueError("缺少 checkpoint")
        detector = build_model(adapter, {"init_args": model_config})
        detector.load(str(checkpoint))
        device = args.device or str(config.get("device", "cuda:0"))
        detector.to(device)
        detector.eval()
        if args.official_hardware_claim and not str(device).startswith("cuda"):
            raise ValueError("--official-hardware-claim requires a CUDA device")
        if args.image is not None:
            with Image.open(args.image) as source:
                image = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
            image_source = str(args.image)
        else:
            if args.image_size <= 0:
                raise ValueError("--image-size 必须 > 0")
            image = np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)
            image_source = "synthetic_zero_proxy"
        tiling = _mapping(config.get("tiling", {}), "tiling")
        batch_size = int(config.get("batch_size", 1))
        try:
            import torch
        except (ImportError, OSError):
            torch = None

        def synchronize() -> None:
            if torch is not None and str(device).startswith("cuda"):
                torch.cuda.synchronize(device)

        hardware_name = "cpu"
        if torch is not None and str(device).startswith("cuda"):
            hardware_name = torch.cuda.get_device_name(torch.device(device))

        measured: list[dict[str, float]] = []
        for run_index in range(args.warmup + args.runs):
            synchronize()
            wall_started = time.perf_counter()
            runtime = RuntimeBreakdown()
            prediction = predict_image(
                detector,
                image_id=1,
                image=image,
                batch_size=batch_size,
                tiling_config=tiling,
                runtime=runtime,
            )
            synchronize()
            started = time.perf_counter()
            json.dumps(
                predictions_to_coco_records([prediction], allowed_category_ids=range(25)),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            runtime.serialization += time.perf_counter() - started
            # CUDA kernels are asynchronous.  Synchronize before closing the
            # end-to-end wall clock so the competition gate cannot undercount
            # pending GPU work that per-stage CPU timers did not observe.
            synchronize()
            wall_total = time.perf_counter() - wall_started
            if run_index >= args.warmup:
                record = runtime.to_dict()
                record["instrumented_total"] = record["total"]
                record["unattributed_or_gpu_sync"] = max(
                    0.0, wall_total - record["instrumented_total"]
                )
                record["total"] = wall_total
                measured.append(record)
        totals = [item["total"] for item in measured]
        summary = {
            "model": "bhcdetr",
            "paper": "arXiv:2512.24074v1",
            "image_source": image_source,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "device": str(device),
            "hardware_name": hardware_name,
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
            "timing_scope": "after image read through result serialization",
            "samples": measured,
            "total_seconds": {
                "mean": statistics.fmean(totals),
                "p50": _percentile(totals, 0.50),
                "p95": _percentile(totals, 0.95),
                "maximum": max(totals),
            },
            "competition_gate": {
                "threshold_seconds": 20.0,
                "p95_passed": _percentile(totals, 0.95) <= 20.0,
                "official_hardware_claim": bool(args.official_hardware_claim),
                "official_gate_passed": (
                    _percentile(totals, 0.95) <= 20.0
                    and bool(args.official_hardware_claim)
                ),
            },
        }
        output = args.output or Path(
            str(config.get("benchmark_output", "outputs/bhcdetr_benchmark.json"))
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except (ImportError, OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        logger.error("测速失败: %s", error)
        return 1
    logger.info("BHC-DETR p50=%.3fs p95=%.3fs -> %s", summary["total_seconds"]["p50"], summary["total_seconds"]["p95"], output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
