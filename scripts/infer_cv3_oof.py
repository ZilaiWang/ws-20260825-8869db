#!/usr/bin/env python3
"""CV3 OOF 正式推理引擎（ultralytics 后端）。

补齐 model 仓库 ``infer.py`` 的契约：读 ``materialize_cv3_oof_config.py``
生成的 ``resolved_infer.yaml``，用 ``UltralyticsDetector`` 对本折 val 图像
做低阈值整图推理，输出：

- ``output_json``（COCO detection 列表，bbox 为 xywh）；
- ``<output_json>.runtime.json``（``rsdet_inference_runtime_v2`` 血缘）。

严格遵循 ``_validate_inference_config`` / ``_validate_runtime_provenance``
契约。tiling 首轮禁用（整图推理）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from rsdet.contracts import InferenceSample
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="infer_cv3_oof")

RUNTIME_SCHEMA_VERSION = "rsdet_inference_runtime_v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"推理配置必须是 YAML 对象: {path}")
    return payload


def _resolve(path: str | Path, base: Path) -> Path:
    result = Path(str(path)).expanduser()
    if not result.is_absolute():
        result = base.parent / result
    return result.resolve()


def _load_val_records(split_view: Path) -> list[dict[str, Any]]:
    view = json.loads(split_view.read_text(encoding="utf-8"))
    samples = view.get("samples")
    if not isinstance(samples, list):
        raise ValueError("split_view.json 缺少 samples")
    return [s for s in samples if s.get("split") == "val"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = _load_yaml(args.config)
    model_cfg = config["model"]
    input_cfg = config["input"]
    data_root = _resolve(input_cfg["data_root"], args.config)
    split_view = _resolve(input_cfg["manifest"], args.config)
    checkpoint = _resolve(model_cfg["checkpoint"], args.config)
    output_json = _resolve(config["output_json"], args.config)

    val_records = _load_val_records(split_view)
    logger.info("val 图像数: %d", len(val_records))

    from rsdet.models.ultralytics_adapter import UltralyticsDetector

    detector = UltralyticsDetector(
        family=str(model_cfg["family"]),
        imgsz=int(model_cfg["imgsz"]),
        confidence=float(model_cfg["confidence"]),
        iou=float(model_cfg["iou"]),
        max_detections=int(model_cfg["max_detections"]),
        half=bool(model_cfg.get("half", True)),
        agnostic_nms=bool(model_cfg.get("agnostic_nms", False)),
        score_transform=model_cfg.get("score_transform"),
    )
    detector.load(str(checkpoint))
    detector.to(str(config["device"]))
    detector.eval()

    predictions: list[dict[str, Any]] = []
    for record in val_records:
        image_id = int(record["image_id"])
        image_path = data_root / record["relative_path"]
        with Image.open(image_path) as img:
            rgb = np.asarray(img.convert("RGB"))
        sample = InferenceSample(
            image_id=image_id,
            image=rgb,
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
        )
        prediction = detector.predict([sample])[0]
        for box, score, label in zip(
            prediction.boxes_xyxy, prediction.scores, prediction.labels
        ):
            x1, y1, x2, y2 = [float(v) for v in box]
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    runtime = {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "images": len(val_records),
        "artifacts": {
            "config": {"path": str(args.config.resolve()), "sha256": _sha256_file(args.config)},
            "checkpoint": {"path": str(checkpoint), "sha256": _sha256_file(checkpoint)},
            "predictions": {"path": str(output_json), "sha256": _sha256_file(output_json)},
        },
    }
    runtime_path = output_json.with_name(output_json.stem + ".runtime.json")
    runtime_path.write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    logger.info("推理完成: %d 条预测 -> %s", len(predictions), output_json)
    print(f"INFER_OOF_PASS proposals={len(predictions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
