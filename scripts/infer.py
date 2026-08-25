#!/usr/bin/env python3
"""End-to-end BHC-DETR inference, tiling, fusion and COCO export."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from rsdet.contracts import Prediction
from rsdet.data.xh_dataset import FINE_NAMES, coarse_name
from rsdet.engine.inference import predict_image
from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_predictions, evaluate_ranking_metrics
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.evaluation.runtime import RuntimeBreakdown
from rsdet.models.registry import build_model
from rsdet.predictions import write_coco_predictions
from rsdet.utils.config import load_config
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="infer")


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} 必须是对象")
    return dict(value)


def _safe_image_path(root: Path, relative_path: str) -> Path:
    relative = Path(str(relative_path).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"不安全的图像相对路径: {relative_path}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise FileNotFoundError(path)
    return path


def _manifest_records(
    path: Path,
    *,
    split: str,
    held_out_fold: int | None = None,
) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            raw = list(csv.DictReader(handle))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping) and isinstance(payload.get("samples"), list):
            raw = payload["samples"]
        elif isinstance(payload, Mapping) and isinstance(payload.get("images"), list):
            raw = [
                {
                    "image_id": item.get("id"),
                    "relative_path": item.get("file_name"),
                }
                for item in payload["images"]
            ]
        elif isinstance(payload, list):
            raw = payload
        else:
            raise ValueError("manifest 必须包含 samples/images 列表")
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise TypeError(f"manifest sample {index} 不是对象")
        item_split = str(item.get("split", "")).strip().lower()
        item_fold = str(item.get("fold", "")).strip()
        if item_split:
            selected = item_split == split.lower()
        elif item_fold:
            if held_out_fold is None:
                raise ValueError("fold manifest requires input.held_out_fold")
            if split.lower() not in {"train", "val"}:
                raise ValueError("fold manifest input.split must be train or val")
            try:
                fold = int(item_fold)
            except ValueError as error:
                raise ValueError(f"manifest sample {index} has an invalid fold") from error
            is_validation = fold == held_out_fold
            selected = is_validation if split.lower() == "val" else not is_validation
        else:
            # Materialized CSV views and COCO image lists may omit split data.
            selected = True
        if not selected:
            continue
        image_id = int(item.get("image_id", index + 1))
        relative_path = item.get("relative_path", item.get("file_name"))
        if image_id <= 0 or image_id in seen or not relative_path:
            raise ValueError(f"manifest sample {index} 的 image_id/path 非法")
        result.append({"image_id": image_id, "relative_path": str(relative_path)})
        seen.add(image_id)
    return result


def _input_records(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    input_config = _mapping(config.get("input"), "input")
    split = str(input_config.get("split", "test"))
    data_root = Path(str(input_config.get("data_root", ""))).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"input.data_root 不存在: {data_root}")
    manifest_value = str(input_config.get("manifest", "")).strip()
    if manifest_value:
        manifest = Path(manifest_value).expanduser()
        held_out_value = input_config.get("held_out_fold")
        held_out_fold = None if held_out_value is None else int(held_out_value)
        records = _manifest_records(
            manifest,
            split=split,
            held_out_fold=held_out_fold,
        )
        for record in records:
            record["path"] = _safe_image_path(data_root, record["relative_path"])
        return records
    image_dir_value = str(input_config.get("image_dir", f"images/{split}"))
    image_dir = Path(image_dir_value).expanduser()
    if not image_dir.is_absolute():
        image_dir = data_root / image_dir
    if not image_dir.is_dir():
        raise FileNotFoundError(f"input.image_dir 不存在: {image_dir}")
    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    return [
        {"image_id": index, "relative_path": path.name, "path": path}
        for index, path in enumerate(paths, start=1)
    ]


def _filter_scores(
    prediction: Prediction,
    coarse_thresholds: Mapping[str, Any],
    fine_thresholds: Mapping[str, Any],
) -> Prediction:
    """Apply fine-class overrides after tile fusion, as required for threshold sweeps."""

    keep: list[int] = []
    for index, (score, label) in enumerate(zip(prediction.scores, prediction.labels)):
        label_id = int(label)
        fine_name = FINE_NAMES[label_id]
        override = fine_thresholds.get(
            fine_name,
            fine_thresholds.get(str(label_id), fine_thresholds.get(label_id)),
        )
        threshold = (
            float(override)
            if override is not None
            else float(coarse_thresholds.get(coarse_name(label_id), 0.0))
        )
        if float(score) >= threshold:
            keep.append(index)
    return Prediction(
        prediction.image_id,
        [prediction.boxes_xyxy[index] for index in keep],
        [prediction.scores[index] for index in keep],
        [prediction.labels[index] for index in keep],
    )


def _automatic_evaluation(
    config: Mapping[str, Any],
    prediction_path: Path,
) -> tuple[Path, dict[str, Any]] | None:
    evaluation = config.get("evaluation")
    if evaluation in (None, {}):
        return None
    evaluation_config = _mapping(evaluation, "evaluation")
    gt_path = Path(str(evaluation_config.get("gt", ""))).expanduser()
    if not gt_path.is_file():
        raise FileNotFoundError(f"evaluation.gt 不存在: {gt_path}")
    project_path = Path(
        str(evaluation_config.get("project_config", "configs/project.yaml"))
    ).expanduser()
    protocol = parse_evaluation_protocol(load_config(project_path))
    ground_truth = load_coco_ground_truth(gt_path)
    predictions = load_coco_predictions(prediction_path)
    pooled = evaluate_predictions(
        ground_truth,
        predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    ranking = evaluate_ranking_metrics(
        ground_truth,
        predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=bool(evaluation_config.get("require_complete_taxonomy", True)),
    )
    metrics: dict[str, Any] = {
        "protocol_versions": {
            "contract_version": protocol.contract_version,
            "eval_version": protocol.eval_version,
            "ranking_version": protocol.ranking_version,
        },
        "overall_recall": pooled.recall,
        "overall_fdr": pooled.fdr,
        "detection_gate": {
            "recall_min": protocol.recall_min,
            "fdr_max": protocol.fdr_max,
            "passed": pooled.recall >= protocol.recall_min and pooled.fdr <= protocol.fdr_max,
        },
        "details": pooled.details,
        "per_class": {
            name: {
                "recall": item.recall,
                "fdr": item.fdr,
                "tp": item.tp,
                "fp": item.fp,
                "fn": item.fn,
            }
            for name, item in pooled.per_class.items()
        },
        "official_ranking": {
            "overall_recall": ranking.overall_recall,
            "overall_fdr": ranking.overall_fdr,
            "per_coarse": {
                name: {
                    "macro_recall": item.macro_recall,
                    "macro_fdr": item.macro_fdr,
                    "pooled_recall": item.pooled_recall,
                    "pooled_fdr": item.pooled_fdr,
                    "fine_count": item.fine_count,
                    "fine_ids": item.fine_ids,
                }
                for name, item in ranking.per_coarse.items()
            },
        },
    }
    output_value = evaluation_config.get("metrics_output")
    output_path = (
        Path(str(output_value)).expanduser()
        if output_value
        else prediction_path.with_name(prediction_path.stem + "_metrics.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path, metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BHC-DETR 小图/10K 滑窗推理")
    parser.add_argument("--config", type=Path, required=True, help="推理 YAML")
    parser.add_argument("--checkpoint", type=Path, default=None, help="覆盖 checkpoint")
    parser.add_argument("--device", type=str, default=None, help="覆盖设备")
    parser.add_argument("--output", type=Path, default=None, help="覆盖 COCO JSON 输出")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.is_file():
        logger.error("配置文件不存在: %s", args.config)
        return 1
    try:
        config = load_config(args.config)
        model_config = _mapping(config.get("model"), "model")
        adapter = str(model_config.pop("adapter", "bhcdetr"))
        if adapter != "bhcdetr":
            raise ValueError("当前可执行主线只允许 model.adapter=bhcdetr")
        # Always remove config-only fields before forwarding model_config to
        # the detector constructor.  Using ``override or pop(...)`` leaves the
        # YAML checkpoint behind whenever --checkpoint is supplied because of
        # Python's short-circuit evaluation.
        configured_checkpoint = model_config.pop("checkpoint", None)
        checkpoint_value = args.checkpoint or configured_checkpoint or config.get("checkpoint")
        if not checkpoint_value:
            raise ValueError("未指定 BHC-DETR checkpoint")
        detector = build_model(adapter, {"init_args": model_config})
        detector.load(str(checkpoint_value))
        detector.to(args.device or str(config.get("device", "cuda:0")))
        detector.eval()
        records = _input_records(config)
        if not records:
            raise ValueError("输入清单没有图像")
        tiling = _mapping(config.get("tiling", {}), "tiling")
        batch_size = int(config.get("batch_size", 1))
        if batch_size <= 0:
            raise ValueError("batch_size 必须 > 0")
        coarse_thresholds = _mapping(config.get("score_thresholds", {}), "score_thresholds")
        fine_thresholds = _mapping(config.get("fine_score_thresholds", {}), "fine_score_thresholds")
        predictions: list[Prediction] = []
        runtime_records: list[dict[str, Any]] = []
        for record in records:
            with Image.open(record["path"]) as source:
                image = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
            runtime = RuntimeBreakdown()
            prediction = predict_image(
                detector,
                image_id=int(record["image_id"]),
                image=image,
                batch_size=batch_size,
                tiling_config=tiling,
                runtime=runtime,
            )
            predictions.append(_filter_scores(prediction, coarse_thresholds, fine_thresholds))
            runtime_records.append(
                {
                    "image_id": int(record["image_id"]),
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    **runtime.to_dict(),
                }
            )
        output_path = args.output or Path(
            str(config.get("output_json", "outputs/bhcdetr_predictions.json"))
        )
        serialization_started = time.perf_counter()
        write_coco_predictions(output_path, predictions, allowed_category_ids=range(25))
        serialization = time.perf_counter() - serialization_started
        if runtime_records:
            runtime_records[-1]["serialization"] += serialization
            runtime_records[-1]["total"] += serialization
        runtime_path = output_path.with_name(output_path.stem + "_runtime.json")
        runtime_path.write_text(
            json.dumps(runtime_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        evaluated = _automatic_evaluation(config, output_path)
    except (
        ImportError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
    ) as error:
        logger.error("BHC-DETR 推理失败: %s", error)
        return 1
    logger.info("预测已写入: %s；时延记录: %s", output_path, runtime_path)
    if evaluated is not None:
        logger.info("自动评估已写入: %s", evaluated[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
