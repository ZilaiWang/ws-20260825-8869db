#!/usr/bin/env python3
"""Produce raw, phase-level 10K runtime records with a real detector adapter.

Production usage intentionally runs this script with ``xh-202625-model/src``
first on ``PYTHONPATH``.  It therefore exercises C's existing model adapter,
tiler, coordinate utilities, and NMS implementation without copying that
repository into the project repository.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image

from rsdet.contracts import InferenceSample, Prediction
from rsdet.data.xh_dataset import coarse_name
from rsdet.models.registry import build_model
from rsdet.postprocess.nms import nms
from rsdet.predictions import predictions_to_coco_records, validate_prediction
from rsdet.tiling.coordinates import clip_bbox, tile_to_full
from rsdet.tiling.slicer import generate_tiles

PHASES = (
    "image_read",
    "tiling",
    "preprocess",
    "model",
    "tile_postprocess",
    "coordinate_restore",
    "fusion",
    "serialization",
)
IMAGE_SOURCE_TYPES = {
    "real_official",
    "real_project_proxy",
    "synthetic",
    "stitched",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} 必须是对象")
    return dict(value)


def _sha256_field(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} 必须是 SHA256 字符串")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} 不是合法 SHA256")
    return normalized


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} 必须是正整数")
    return value


def _validate_benchmark_contract(value: Any) -> dict[str, Any]:
    contract = _mapping(value, "benchmark contract")
    if contract.get("contract_version") != "runtime_10k_benchmark_v1":
        raise ValueError("未知 benchmark contract")
    source_type = contract.get("image_source_type")
    if not isinstance(source_type, str) or source_type not in IMAGE_SOURCE_TYPES:
        raise ValueError("benchmark contract image_source_type 非法")
    model_key = contract.get("model_key")
    if not isinstance(model_key, str) or model_key.strip().upper() not in {"M1", "M3"}:
        raise ValueError("benchmark contract model_key 只允许 M1/M3")
    contract["model_key"] = model_key.strip().upper()
    for field in (
        "image_manifest_sha256",
        "checkpoint_sha256",
        "checkpoint_provenance_sha256",
        "config_sha256",
        "hardware_sha256",
    ):
        contract[field] = _sha256_field(contract.get(field), f"benchmark contract {field}")
    if not isinstance(contract.get("engineering_checkpoint_only"), bool):
        raise ValueError("benchmark contract engineering_checkpoint_only 必须是布尔值")
    if contract.get("cuda_synchronized") is not True:
        raise ValueError("benchmark contract 必须声明 cuda_synchronized=true")
    if contract.get("timing_method") != "perf_counter_with_torch_cuda_synchronize":
        raise ValueError("benchmark contract timing_method 与采集器实现不一致")
    for field in (
        "tile_size",
        "expected_tile_count",
        "warmup_runs",
        "minimum_measured_runs",
    ):
        contract[field] = _positive_integer(contract.get(field), f"benchmark contract {field}")
    overlap = contract.get("overlap")
    if (
        isinstance(overlap, bool)
        or not isinstance(overlap, int)
        or not 0 <= overlap < contract["tile_size"]
    ):
        raise ValueError("benchmark contract overlap 必须是 [0,tile_size) 内的整数")
    return contract


def _safe_path(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"图像路径必须是安全相对路径: {relative_path}")
    result = (root / relative).resolve()
    if not result.is_relative_to(root) or not result.is_file():
        raise FileNotFoundError(f"图像不存在或越出 data root: {relative_path}")
    return result


def _resolve_config_path(value: Any, config_path: Path, field: str) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} 不能为空")
    result = Path(text).expanduser()
    if not result.is_absolute():
        result = config_path.parent / result
    return result.resolve()


def _load_images(
    manifest_path: Path,
    *,
    data_root: Path,
    expected_width: int,
    expected_height: int,
    required_count: int,
    source_type: str,
) -> list[dict[str, Any]]:
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or document.get("version") != "e_10k_image_manifest_v1":
        raise ValueError("未知 image manifest version")
    samples = document.get("samples") if isinstance(document, Mapping) else None
    if not isinstance(samples, list):
        raise ValueError("image manifest 必须包含 samples 列表")
    result: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_shas: set[str] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"samples[{index}] 必须是对象")
        image_id = int(sample["image_id"])
        if image_id in seen_ids:
            raise ValueError(f"image_id 重复: {image_id}")
        if int(sample["width"]) != expected_width or int(sample["height"]) != expected_height:
            raise ValueError(f"image_id={image_id} 不是冻结的 10K 尺寸")
        if str(sample.get("image_source_type", "")) != source_type:
            raise ValueError(f"image_id={image_id} 来源类型与 benchmark contract 不一致")
        image_path = _safe_path(data_root, str(sample["relative_path"]))
        expected_sha = str(sample["sha256"]).strip().lower()
        actual_sha = _sha256(image_path)
        if actual_sha != expected_sha:
            raise ValueError(f"image_id={image_id} 内容 SHA 不匹配")
        if actual_sha in seen_shas:
            raise ValueError("正式 measured 输入不得用同一图像内容重复充数")
        result.append(
            {
                "image_id": image_id,
                "path": image_path,
                "sha256": actual_sha,
            }
        )
        seen_ids.add(image_id)
        seen_shas.add(actual_sha)
    if len(result) < required_count:
        raise ValueError(f"需要至少 {required_count} 张不同内容的 10K 图，实际 {len(result)} 张")
    return result


def _validate_checkpoint_provenance(
    provenance_path: Path,
    *,
    contract: Mapping[str, Any],
) -> Path:
    if _sha256(provenance_path) != str(contract.get("checkpoint_provenance_sha256", "")):
        raise ValueError("checkpoint provenance SHA 与 benchmark contract 不一致")
    payload = _mapping(
        json.loads(provenance_path.read_text(encoding="utf-8")),
        "checkpoint provenance",
    )
    if payload.get("contract_version") != "checkpoint_provenance_v1":
        raise ValueError("未知 checkpoint provenance contract")
    if payload.get("status") != "checkpoint_lineage_verified":
        raise ValueError("checkpoint provenance 尚未通过 lineage 门禁")
    if payload.get("engineering_checkpoint_only") is not contract.get(
        "engineering_checkpoint_only"
    ):
        raise ValueError("checkpoint 科学用途声明与 benchmark contract 不一致")
    if (
        str(payload.get("model_key", "")).strip().upper()
        != str(contract.get("model_key", "")).strip().upper()
    ):
        raise ValueError("checkpoint provenance model_key 不一致")

    checkpoint = _resolve_config_path(
        payload.get("checkpoint"), provenance_path, "checkpoint provenance.checkpoint"
    )
    checkpoint_sha = str(payload.get("checkpoint_sha256", "")).strip().lower()
    if (
        not checkpoint.is_file()
        or checkpoint_sha != str(contract.get("checkpoint_sha256", "")).lower()
        or _sha256(checkpoint) != checkpoint_sha
    ):
        raise ValueError("checkpoint provenance 指向的 checkpoint 实体/SHA 不一致")

    fold_metadata_path = _resolve_config_path(
        payload.get("fold_metadata"),
        provenance_path,
        "checkpoint provenance.fold_metadata",
    )
    oof_metadata_path = _resolve_config_path(
        payload.get("oof_metadata"),
        provenance_path,
        "checkpoint provenance.oof_metadata",
    )
    for path, expected_sha, label in (
        (
            fold_metadata_path,
            payload.get("fold_metadata_sha256"),
            "fold metadata",
        ),
        (
            oof_metadata_path,
            payload.get("oof_metadata_sha256"),
            "OOF metadata",
        ),
    ):
        if not path.is_file() or _sha256(path) != str(expected_sha):
            raise ValueError(f"checkpoint provenance {label} 实体/SHA 不一致")

    fold_metadata = _mapping(
        json.loads(fold_metadata_path.read_text(encoding="utf-8")),
        "fold metadata",
    )
    source_fold_value = payload.get("source_fold")
    if isinstance(source_fold_value, bool) or not isinstance(source_fold_value, int):
        raise ValueError("checkpoint provenance source_fold 必须是非负整数")
    source_fold = source_fold_value
    if source_fold < 0:
        raise ValueError("checkpoint provenance source_fold 必须是非负整数")
    held_out_fold = fold_metadata.get("held_out_fold")
    if isinstance(held_out_fold, bool) or not isinstance(held_out_fold, int):
        raise ValueError("fold metadata held_out_fold 必须是整数")
    artifacts = _mapping(fold_metadata.get("artifacts"), "fold metadata.artifacts")
    if (
        fold_metadata.get("status") != "fold_delivery_complete"
        or str(fold_metadata.get("model_key", "")).upper() != str(contract["model_key"]).upper()
        or held_out_fold != source_fold
        or _resolve_config_path(
            artifacts.get("checkpoint"),
            fold_metadata_path,
            "fold metadata.artifacts.checkpoint",
        )
        != checkpoint
        or str(artifacts.get("checkpoint_sha256", "")).lower() != checkpoint_sha
    ):
        raise ValueError("fold metadata 未闭环到所选 checkpoint")

    oof_metadata = _mapping(
        json.loads(oof_metadata_path.read_text(encoding="utf-8")),
        "OOF metadata",
    )
    folds = oof_metadata.get("folds")
    if (
        oof_metadata.get("status") != "complete_downstream_ready"
        or oof_metadata.get("downstream_admission") is not True
        or str(oof_metadata.get("model_key", "")).upper() != str(contract["model_key"]).upper()
        or not isinstance(folds, list)
    ):
        raise ValueError("OOF metadata 尚未完成正式 aggregate")
    matching = [
        fold
        for fold in folds
        if (
            isinstance(fold, Mapping)
            and not isinstance(fold.get("fold"), bool)
            and isinstance(fold.get("fold"), int)
            and fold["fold"] == source_fold
        )
    ]
    if (
        len(matching) != 1
        or str(matching[0].get("checkpoint_sha256", "")).lower() != checkpoint_sha
        or str(matching[0].get("metadata_sha256", "")).lower() != _sha256(fold_metadata_path)
    ):
        raise ValueError("OOF metadata 未闭环到所选 fold/checkpoint")
    return checkpoint


def _cuda_sync(torch: Any, device: str) -> None:
    if not device.startswith("cuda"):
        raise ValueError("正式 10K GPU 测速必须使用 CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    torch.cuda.synchronize(device=device)


def _require_real_adapter(adapter: str, detector: Any) -> None:
    """Reject dummy or custom adapters for the frozen M1/M3 benchmark."""

    if adapter != "ultralytics":
        raise ValueError("正式 M1/M3 10K 采集只允许 adapter=ultralytics")
    detector_type = type(detector)
    if (
        detector_type.__module__ != "rsdet.models.ultralytics_adapter"
        or detector_type.__name__ != "UltralyticsDetector"
    ):
        raise RuntimeError(
            "ultralytics 注册项不是冻结的真实 UltralyticsDetector；"
            "请确认 xh-202625-model/src 位于 PYTHONPATH 第一位"
        )


def _timed(function: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = function()
    return result, time.perf_counter() - started


def _predict_adapter_batches(
    detector: Any,
    samples: Sequence[InferenceSample],
    *,
    batch_size: int,
) -> list[Prediction]:
    """Batch adapter calls without folding public contract validation into model time."""

    if batch_size <= 0:
        raise ValueError("batch_size 必须 > 0")
    outputs: list[Prediction] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        batch_outputs = list(detector.predict(batch))
        if len(batch_outputs) != len(batch):
            raise ValueError(
                f"模型返回 {len(batch_outputs)} 个预测，当前 batch 有 {len(batch)} 个输入"
            )
        outputs.extend(batch_outputs)
    return outputs


def _expected_tile_count(
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
) -> int:
    if width <= 0 or height <= 0 or tile_size <= 0:
        raise ValueError("width/height/tile_size 必须为正数")
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap 必须位于 [0,tile_size)")
    stride = tile_size - overlap

    def count_axis(length: int) -> int:
        if length <= tile_size:
            return 1
        return math.ceil((length - tile_size) / stride) + 1

    return count_axis(width) * count_axis(height)


def _restore_predictions(
    predictions: Sequence[Prediction],
    tiles: Sequence[Any],
    *,
    width: int,
    height: int,
) -> tuple[list[list[float]], list[float], list[int]]:
    boxes: list[list[float]] = []
    scores: list[float] = []
    labels: list[int] = []
    for prediction, tile in zip(predictions, tiles):
        if prediction.image_id != tile.tile_id:
            raise ValueError("tile prediction ID 与 tile record 不一致")
        for box, score, label in zip(
            prediction.boxes_xyxy,
            prediction.scores,
            prediction.labels,
        ):
            restored = tile_to_full(box, tile.x_offset, tile.y_offset)
            clipped = [float(value) for value in clip_bbox(restored, width, height)]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            boxes.append(clipped)
            scores.append(float(score))
            labels.append(int(label))
    return boxes, scores, labels


def _grouped_keep(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    threshold: float,
    coarse: bool,
) -> list[int]:
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[coarse_name(label) if coarse else label].append(index)
    keep: list[int] = []
    for indices in groups.values():
        local = nms(
            [[float(value) for value in boxes[index]] for index in indices],
            [float(scores[index]) for index in indices],
            threshold,
        )
        keep.extend(indices[position] for position in local)
    return sorted(keep, key=lambda index: (-scores[index], index))


def _fuse(
    image_id: int,
    boxes: list[list[float]],
    scores: list[float],
    labels: list[int],
    *,
    fine_iou: float,
    coarse_iou: float | None,
    maximum: int,
) -> Prediction:
    if not boxes:
        return Prediction(image_id, [], [], [])
    keep = _grouped_keep(
        boxes,
        scores,
        labels,
        threshold=fine_iou,
        coarse=False,
    )
    boxes = [boxes[index] for index in keep]
    scores = [scores[index] for index in keep]
    labels = [labels[index] for index in keep]
    if coarse_iou is not None:
        keep = _grouped_keep(
            boxes,
            scores,
            labels,
            threshold=coarse_iou,
            coarse=True,
        )
        boxes = [boxes[index] for index in keep]
        scores = [scores[index] for index in keep]
        labels = [labels[index] for index in keep]
    return Prediction(
        image_id,
        boxes[:maximum],
        scores[:maximum],
        labels[:maximum],
    )


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集真实 10K 完整流水线分段时间")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark-contract", type=Path, required=True)
    parser.add_argument("--checkpoint-provenance", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-width", type=int, default=10000)
    parser.add_argument("--expected-height", type=int, default=10000)
    return parser.parse_args(argv)


def _run_capture(
    args: argparse.Namespace,
    *,
    final_output_dir: Path,
) -> dict[str, Any]:
    import torch

    if (args.expected_width, args.expected_height) != (10000, 10000):
        raise ValueError("正式 E-10K 采集只接受冻结的 10000x10000 尺寸")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = _mapping(yaml.safe_load(args.config.read_text(encoding="utf-8")), "config")
    contract = _validate_benchmark_contract(
        json.loads(args.benchmark_contract.read_text(encoding="utf-8"))
    )
    if _sha256(args.image_manifest) != str(contract["image_manifest_sha256"]):
        raise ValueError("image manifest SHA 与 benchmark contract 不一致")
    if _sha256(args.config) != str(contract["config_sha256"]):
        raise ValueError("config SHA 与 benchmark contract 不一致")
    provenance_checkpoint = _validate_checkpoint_provenance(
        args.checkpoint_provenance,
        contract=contract,
    )
    model_config = _mapping(config.get("model"), "model")
    model_key = str(contract.get("model_key", "")).strip().upper()
    expected_families = {"M1": "yolo", "M3": "rtdetr"}
    if model_key not in expected_families:
        raise ValueError("benchmark contract model_key 只允许 M1/M3")
    if str(model_config.get("family", "")).strip().lower() != expected_families[model_key]:
        raise ValueError("config model.family 与 benchmark contract model_key 不一致")
    checkpoint = Path(str(model_config.pop("checkpoint", ""))).expanduser().resolve()
    if checkpoint != provenance_checkpoint:
        raise ValueError("config checkpoint 与已验收 checkpoint provenance 不一致")
    if not checkpoint.is_file() or _sha256(checkpoint) != str(contract["checkpoint_sha256"]):
        raise ValueError("checkpoint 路径/SHA 与 benchmark contract 不一致")
    input_config = _mapping(config.get("input"), "input")
    data_root = args.data_root.resolve()
    if (
        _resolve_config_path(input_config.get("data_root"), args.config, "input.data_root")
        != data_root
    ):
        raise ValueError("config input.data_root 与命令行 data root 不一致")
    if (
        _resolve_config_path(input_config.get("manifest"), args.config, "input.manifest")
        != args.image_manifest.resolve()
    ):
        raise ValueError("config input.manifest 与命令行 image manifest 不一致")
    expected_output = final_output_dir.resolve() / "predictions_10k_low.json"
    if (
        _resolve_config_path(config.get("output_json"), args.config, "output_json")
        != expected_output
    ):
        raise ValueError("config output_json 与本次 capture 输出目录不一致")
    device = str(config.get("device", "cuda:0"))
    adapter = str(model_config.pop("adapter", "ultralytics"))
    detector = build_model(adapter, {"init_args": model_config})
    _require_real_adapter(adapter, detector)
    detector.load(str(checkpoint))
    detector.to(device)
    detector.eval()

    tiling = _mapping(config.get("tiling"), "tiling")
    tile_size = int(tiling["tile_size"])
    overlap = int(tiling["overlap"])
    if tile_size != int(contract["tile_size"]) or overlap != int(contract["overlap"]):
        raise ValueError("config tiling 与 benchmark contract 不一致")
    computed_tile_count = _expected_tile_count(
        args.expected_width,
        args.expected_height,
        tile_size,
        overlap,
    )
    if int(contract.get("expected_tile_count", -1)) != computed_tile_count:
        raise ValueError(
            "benchmark contract expected_tile_count 与冻结几何不一致: "
            f"declared={contract.get('expected_tile_count')}, "
            f"computed={computed_tile_count}"
        )
    threshold = float(model_config.get("confidence", math.nan))
    thresholds = _mapping(config.get("score_thresholds"), "score_thresholds")
    if not math.isclose(threshold, 0.001, abs_tol=1e-12) or any(
        not math.isclose(float(thresholds.get(name, math.nan)), 0.001, abs_tol=1e-12)
        for name in ("ship", "aircraft", "vehicle")
    ):
        raise ValueError("正式 10K 候选阈值必须统一为 0.001")
    if config.get("fine_score_thresholds", {}) not in ({}, None):
        raise ValueError("正式 10K 低阈值采集禁止细类独立阈值")
    warmup_runs = int(contract["warmup_runs"])
    measured_runs = int(contract["minimum_measured_runs"])
    images = _load_images(
        args.image_manifest,
        data_root=data_root,
        expected_width=args.expected_width,
        expected_height=args.expected_height,
        required_count=measured_runs,
        source_type=str(contract["image_source_type"]),
    )
    scheduled_images = [images[index % len(images)] for index in range(warmup_runs)] + images[
        :measured_runs
    ]
    runtime_path = args.output_dir / "runtime_samples.jsonl"
    all_measured_predictions: list[dict[str, Any]] = []
    measured_fused_count = 0
    for run_index, image_record in enumerate(scheduled_images):
        phases = {phase: 0.0 for phase in PHASES}
        started = time.perf_counter()
        with Image.open(image_record["path"]) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        phases["image_read"] = time.perf_counter() - started
        if image.shape != (args.expected_height, args.expected_width, 3):
            raise ValueError("读入图像形状不是冻结的 HxWx3")

        raw_tiles, phases["tiling"] = _timed(
            lambda: generate_tiles(
                args.expected_width,
                args.expected_height,
                tile_size,
                overlap,
            )
        )
        tiles = [replace(tile, parent_image_id=int(image_record["image_id"])) for tile in raw_tiles]
        if len(tiles) != computed_tile_count:
            raise ValueError(f"实际 tile_count={len(tiles)}，冻结值={computed_tile_count}")

        def make_samples() -> list[InferenceSample]:
            return [
                InferenceSample(
                    tile.tile_id,
                    np.ascontiguousarray(
                        image[
                            tile.y_offset : tile.y_offset + tile.height,
                            tile.x_offset : tile.x_offset + tile.width,
                        ]
                    ),
                    tile.width,
                    tile.height,
                    {
                        "parent_image_id": image_record["image_id"],
                        "x_offset": tile.x_offset,
                        "y_offset": tile.y_offset,
                    },
                )
                for tile in tiles
            ]

        samples, phases["preprocess"] = _timed(make_samples)
        torch.cuda.reset_peak_memory_stats(device=device)
        _cuda_sync(torch, device)
        started = time.perf_counter()
        tile_predictions = _predict_adapter_batches(
            detector,
            samples,
            batch_size=int(config.get("batch_size", 1)),
        )
        _cuda_sync(torch, device)
        phases["model"] = time.perf_counter() - started
        peak_vram_mib = torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)
        if len(tile_predictions) != len(tiles):
            raise ValueError("模型输出 tile 数与输入 tile 数不一致")

        started = time.perf_counter()
        for sample, prediction in zip(samples, tile_predictions):
            validate_prediction(
                prediction,
                expected_image_id=sample.image_id,
                allowed_category_ids=range(25),
                image_size=(sample.width, sample.height),
            )
        raw_proposal_count = sum(len(prediction.scores) for prediction in tile_predictions)
        phases["tile_postprocess"] = time.perf_counter() - started

        started = time.perf_counter()
        boxes, scores, labels = _restore_predictions(
            tile_predictions,
            tiles,
            width=args.expected_width,
            height=args.expected_height,
        )
        phases["coordinate_restore"] = time.perf_counter() - started

        started = time.perf_counter()
        fused = _fuse(
            int(image_record["image_id"]),
            boxes,
            scores,
            labels,
            fine_iou=float(tiling.get("fine_nms_iou", 0.55)),
            coarse_iou=(
                None if tiling.get("coarse_nms_iou") is None else float(tiling["coarse_nms_iou"])
            ),
            maximum=int(tiling.get("max_detections", 2000)),
        )
        phases["fusion"] = time.perf_counter() - started
        fused_count = len(fused.scores)

        started = time.perf_counter()
        output_records = predictions_to_coco_records([fused], allowed_category_ids=range(25))
        output_path = args.output_dir / f"run_{run_index:03d}_predictions.json"
        output_path.write_text(
            json.dumps(output_records, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        phases["serialization"] = time.perf_counter() - started
        if run_index >= warmup_runs:
            all_measured_predictions.extend(output_records)
            measured_fused_count += fused_count

        record = {
            "run_index": run_index,
            "image_id": int(image_record["image_id"]),
            "warmup": run_index < warmup_runs,
            "width": args.expected_width,
            "height": args.expected_height,
            "image_content_sha256": image_record["sha256"],
            "image_source_type": contract["image_source_type"],
            "model_key": contract["model_key"],
            "checkpoint_sha256": contract["checkpoint_sha256"],
            "config_sha256": contract["config_sha256"],
            "timing_method": contract["timing_method"],
            "cuda_synchronized": True,
            "tile_size": tile_size,
            "overlap": overlap,
            "tile_count": len(tiles),
            "raw_proposal_count": raw_proposal_count,
            "fused_proposal_count": fused_count,
            "peak_vram_mib": peak_vram_mib,
            "phases": phases,
            "total_after_read": sum(
                value for phase, value in phases.items() if phase != "image_read"
            ),
        }
        _append_jsonl(runtime_path, record)

    aggregate_path = args.output_dir / "predictions_10k_low.json"
    if len(all_measured_predictions) != measured_fused_count:
        raise RuntimeError("measured prediction 聚合数量与逐次 fused proposal 数不一致")
    aggregate_path.write_text(
        json.dumps(
            all_measured_predictions,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return {
        "runtime_jsonl": str(runtime_path),
        "predictions": str(aggregate_path),
        "runs": warmup_runs + measured_runs,
        "measured_runs": measured_runs,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Capture in a sibling staging directory, then publish atomically."""

    final_output_dir = args.output_dir.resolve()
    if final_output_dir.exists() and (
        not final_output_dir.is_dir() or any(final_output_dir.iterdir())
    ):
        raise FileExistsError(f"输出目录非空或不是目录，禁止混入旧测速: {final_output_dir}")
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            dir=final_output_dir.parent,
            prefix=f".{final_output_dir.name}.staging.",
        )
    )
    staged_args = copy.copy(args)
    staged_args.output_dir = staging_dir
    published = False
    try:
        result = _run_capture(
            staged_args,
            final_output_dir=final_output_dir,
        )
        if final_output_dir.exists():
            final_output_dir.rmdir()
        staging_dir.replace(final_output_dir)
        published = True
        result["runtime_jsonl"] = str(final_output_dir / "runtime_samples.jsonl")
        result["predictions"] = str(final_output_dir / "predictions_10k_low.json")
        return result
    finally:
        if not published:
            shutil.rmtree(staging_dir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except (
        ImportError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as error:
        print(f"BENCHMARK_10K_FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "BENCHMARK_10K_CAPTURE_PASS "
        f"runs={result['runs']} measured={result['measured_runs']} "
        f"runtime={result['runtime_jsonl']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
