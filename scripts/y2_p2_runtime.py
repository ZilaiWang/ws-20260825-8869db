#!/usr/bin/env python3
"""Formal YOLO26-s P2 training and low-threshold held-out inference.

This entry point is deliberately independent of the repository's current
BHC-DETR main runner.  It emits the same framework-neutral CV3 artifacts as
M1, so :mod:`rsdet.experiments.cv3_oof` remains the only final authority.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from PIL import Image

from rsdet.experiments.cv3_oof import (
    FORMAL_INFERENCE_CONTRACTS,
    FORMAL_TRAINING_CONTRACTS,
    INFERENCE_RUNTIME_SCHEMA_VERSION,
    atomic_write_json,
    sha256_file,
)
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="y2_p2_runtime")
EXPECTED_ULTRALYTICS_VERSION = "8.4.103"
EXPECTED_ARCHITECTURE = "yolo26s-p2.yaml"
EXPECTED_STRIDES = (4.0, 8.0, 16.0, 32.0)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} 必须是对象")
    return dict(value)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("配置顶层必须是对象")
    return dict(payload)


def _resolve(value: Any, *, relative_to: Path, field: str) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} 不能为空")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _safe_source(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"不安全的图像路径: {relative_path}")
    source = (root / relative).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise FileNotFoundError(source)
    return source


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as error:
        raise ValueError(f"图像路径不含 images 目录: {image_path}") from error
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_split_view(
    manifest_path: Path,
    *,
    data_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Load and fully validate one materialized CV3 split view."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("samples"), list):
        raise ValueError("split view 必须包含 samples 列表")
    held_out_fold = int(payload.get("held_out_fold", -1))
    if held_out_fold not in (0, 1, 2):
        raise ValueError("split view held_out_fold 必须是 0/1/2")
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index, raw in enumerate(payload["samples"]):
        if not isinstance(raw, Mapping):
            raise ValueError(f"samples[{index}] 必须是对象")
        image_id = int(raw["image_id"])
        if image_id <= 0 or image_id in seen:
            raise ValueError(f"samples[{index}] image_id 非法或重复")
        split = str(raw.get("split", ""))
        if split not in {"train", "val"}:
            raise ValueError(f"samples[{index}].split 必须是 train/val")
        image_path = _safe_source(data_root, str(raw["relative_path"]))
        label_path = _label_path(image_path)
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        item = {
            "image_id": image_id,
            "relative_path": str(raw["relative_path"]),
            "image_path": image_path,
            "label_path": label_path,
        }
        (train if split == "train" else val).append(item)
        seen.add(image_id)
    if len(seen) != 4481 or not train or not val:
        raise ValueError(
            f"split view 必须覆盖 4481 图且两集非空: "
            f"all={len(seen)}, train={len(train)}, val={len(val)}"
        )
    if len(val) != int(payload.get("val_images", -1)):
        raise ValueError("split view val_images 计数不一致")
    return train, val, held_out_fold


def materialize_ultralytics_dataset(
    *,
    train_records: Sequence[Mapping[str, Any]],
    val_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Path:
    """Write deterministic absolute-path lists for Ultralytics."""

    output_dir.mkdir(parents=True, exist_ok=False)
    train_txt = output_dir / "train.txt"
    val_txt = output_dir / "val.txt"
    train_txt.write_text(
        "\n".join(str(item["image_path"]) for item in train_records) + "\n",
        encoding="utf-8",
    )
    val_txt.write_text(
        "\n".join(str(item["image_path"]) for item in val_records) + "\n",
        encoding="utf-8",
    )
    dataset = {
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
        "nc": 25,
        "names": [str(index) for index in range(25)],
    }
    dataset_path = output_dir / "dataset.yaml"
    dataset_path.write_text(
        yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return dataset_path


def _versions(torch_module: Any) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch_module.__version__),
        "cuda_runtime": str(torch_module.version.cuda),
        "cudnn": int(torch_module.backends.cudnn.version() or 0),
        "ultralytics": importlib.metadata.version("ultralytics"),
        "gpu": (
            torch_module.cuda.get_device_name(0)
            if torch_module.cuda.is_available()
            else "unavailable"
        ),
    }


def audit_p2_architecture(model: Any) -> dict[str, Any]:
    """Fail closed unless the instantiated detector exposes P2/P3/P4/P5."""

    torch_model = getattr(model, "model", None)
    layers = getattr(torch_model, "model", None)
    if layers is None or len(layers) == 0:
        raise ValueError("无法读取 Ultralytics 模型层")
    head = layers[-1]
    stride_value = getattr(head, "stride", None)
    if stride_value is None:
        raise ValueError("Detect head 缺少 stride")
    strides = tuple(float(value) for value in stride_value.detach().cpu().tolist())
    if strides != EXPECTED_STRIDES:
        raise ValueError(f"P2 Detect strides 错误: {strides} != {EXPECTED_STRIDES}")
    sources = tuple(int(value) for value in getattr(head, "f", ()))
    if len(sources) != 4:
        raise ValueError(f"P2 Detect 必须有 4 个输入，实际 {sources}")
    parameters = sum(int(parameter.numel()) for parameter in torch_model.parameters())
    return {
        "status": "pass",
        "architecture": EXPECTED_ARCHITECTURE,
        "detect_strides": list(strides),
        "detect_sources": list(sources),
        "parameter_count": parameters,
    }


def _parameter_snapshot(torch_model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone() for name, parameter in torch_model.named_parameters()
    }


def _transfer_audit(before: Mapping[str, Any], torch_model: Any) -> dict[str, Any]:
    changed: list[str] = []
    changed_numel = 0
    total_numel = 0
    for name, parameter in torch_model.named_parameters():
        total_numel += int(parameter.numel())
        previous = before.get(name)
        if previous is None:
            continue
        current = parameter.detach().cpu()
        if previous.shape == current.shape and not current.equal(previous):
            changed.append(name)
            changed_numel += int(parameter.numel())
    if not changed:
        raise ValueError("yolo26s.pt 未向 P2 结构迁移任何参数")
    return {
        "changed_parameter_tensor_count": len(changed),
        "changed_parameter_numel": changed_numel,
        "model_parameter_numel_before_quality_module": total_numel,
        "changed_fraction": changed_numel / total_numel,
        "changed_backbone_tensor_count": sum(
            name.startswith(tuple(f"model.{index}." for index in range(11))) for name in changed
        ),
        "changed_parameter_names": changed,
    }


def _validate_ultralytics_version() -> None:
    actual = importlib.metadata.version("ultralytics")
    if actual != EXPECTED_ULTRALYTICS_VERSION:
        raise ValueError(f"ultralytics 版本必须是 {EXPECTED_ULTRALYTICS_VERSION}，实际 {actual}")


def _validate_y2_gate(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        raise FileNotFoundError("Y3 必须提供已通过的 Y2 decision.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Y2 decision 顶层必须是对象")
    if payload.get("contract_version") != "y2_p2_formal_decision_v1":
        raise ValueError("Y2 decision contract_version 不一致")
    if payload.get("p2_structure_admission") is not True:
        raise ValueError("Y2 P2 未准入，Y3 必须停止")
    if payload.get("quality_stage_admission") is not True:
        raise ValueError("Y2 未授权质量改进阶段")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def run_train(config_path: Path, *, gate_decision: Path | None = None) -> int:
    config = _load_yaml(config_path)
    output_dir = _resolve(
        config.get("output_dir"), relative_to=config_path.parent, field="output_dir"
    )
    run_dir = output_dir / "runs" / "foundation"
    summary_path = output_dir / "train_summary.json"
    if run_dir.exists() or summary_path.exists():
        raise FileExistsError("训练产物已存在，禁止覆盖或隐式 resume")
    model_config = _mapping(config.get("model"), "model")
    if model_config.get("architecture") != EXPECTED_ARCHITECTURE:
        raise ValueError(f"model.architecture 必须是 {EXPECTED_ARCHITECTURE}")
    quality_module = str(model_config.get("quality_module", "")).strip()
    if quality_module not in {"", "ibs_p2_pair_v1"}:
        raise ValueError(f"未冻结的 P2 质量模块: {quality_module}")
    experiment_key = "Y3" if quality_module else "P2"
    gate_artifact = _validate_y2_gate(gate_decision) if quality_module else None
    train_config = _mapping(config.get("train"), "train")
    if train_config.get("checkpoint_selection") != "last":
        raise ValueError("正式 P2 只允许 fixed last checkpoint")
    if train_config.get("device") != FORMAL_TRAINING_CONTRACTS[experiment_key]["device"]:
        raise ValueError("训练 device 与冻结合同不一致")
    train_args = _mapping(train_config.get("args"), "train.args")
    if train_args != FORMAL_TRAINING_CONTRACTS[experiment_key]["train_args"]:
        raise ValueError("train.args 与冻结合同不一致")
    stages = train_config.get("stages")
    if not isinstance(stages, list) or len(stages) != 1 or not isinstance(stages[0], Mapping):
        raise ValueError("正式 P2 只允许一个 foundation stage")
    stage = dict(stages[0])
    if stage != {
        "name": "foundation",
        "epochs": 160,
        "balanced": False,
        "args": FORMAL_TRAINING_CONTRACTS[experiment_key]["stage_args"],
    }:
        raise ValueError("foundation stage 与 P2 冻结合同不一致")
    data = _mapping(config.get("data"), "data")
    data_root = _resolve(data.get("root"), relative_to=config_path.parent, field="data.root")
    manifest = _resolve(data.get("manifest"), relative_to=config_path.parent, field="data.manifest")
    pretrained = _resolve(
        model_config.get("weights"), relative_to=config_path.parent, field="model.weights"
    )
    if not pretrained.is_file():
        raise FileNotFoundError(pretrained)
    train_records, val_records, held_out_fold = load_split_view(
        manifest,
        data_root=data_root,
    )
    prepared = output_dir / "prepared_data"
    dataset_yaml = materialize_ultralytics_dataset(
        train_records=train_records,
        val_records=val_records,
        output_dir=prepared,
    )

    _validate_ultralytics_version()
    import torch
    from ultralytics import YOLO

    model = YOLO(EXPECTED_ARCHITECTURE)
    before_transfer = _parameter_snapshot(model.model)
    model.load(str(pretrained))
    transfer_audit = _transfer_audit(before_transfer, model.model)
    quality_audit = None
    trainer = None
    if quality_module:
        from rsdet.models.ibs_sampling import (
            build_ibs_detection_trainer,
            inject_p2_ibs_pair,
        )

        quality_audit = inject_p2_ibs_pair(model.model)
        trainer = build_ibs_detection_trainer()
    architecture_audit = audit_p2_architecture(model)
    architecture_audit.update(
        {
            "held_out_fold": held_out_fold,
            "pretrained_weight": str(pretrained),
            "pretrained_weight_sha256": sha256_file(pretrained),
            "experiment_key": experiment_key,
            "quality_module": quality_audit,
            "admission_gate": gate_artifact,
            "pretrained_transfer": transfer_audit,
        }
    )
    atomic_write_json(output_dir / "architecture_audit.json", architecture_audit)
    arguments = {
        **train_args,
        **FORMAL_TRAINING_CONTRACTS[experiment_key]["stage_args"],
        "epochs": 160,
        "device": train_config["device"],
        "seed": int(config["seed"]),
        "data": str(dataset_yaml),
        "project": str((output_dir / "runs").resolve()),
        "name": "foundation",
        "exist_ok": False,
    }
    started = time.time()
    model.train(trainer=trainer, **arguments)
    elapsed = time.time() - started
    last = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    if not last.is_file() or not results_csv.is_file():
        raise FileNotFoundError("训练完成但 last.pt/results.csv 不完整")
    with results_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 160:
        raise ValueError(f"正式 P2 必须完成 160 epoch，results.csv={len(rows)} 行")
    summary = {
        "dry_run": False,
        "seed": int(config["seed"]),
        "model_family": "yolo",
        "model_key": experiment_key,
        "quality_module": quality_audit,
        "admission_gate": gate_artifact,
        "checkpoint_selection": "last",
        "initial_weights": str(pretrained),
        "held_out_fold": held_out_fold,
        "elapsed_seconds": elapsed,
        "environment": _versions(torch),
        "stages": [
            {
                "name": "foundation",
                "balanced": False,
                "input_weights": str(pretrained),
                "arguments": arguments,
                "checkpoint_selection": "last",
                "last": str(last.resolve()),
                "selected_checkpoint": str(last.resolve()),
                "completed_epochs": len(rows),
            }
        ],
    }
    atomic_write_json(summary_path, summary)
    (output_dir / "environment.txt").write_text(
        json.dumps(_versions(torch), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Y2 P2 fold%d training complete: %s", held_out_fold, last)
    return 0


def _result_records(result: Any, image_id: int) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().tolist()
    scores = boxes.conf.detach().cpu().tolist()
    labels = boxes.cls.detach().cpu().tolist()
    height, width = [int(value) for value in result.orig_shape]
    output: list[dict[str, Any]] = []
    for box, score, label in zip(xyxy, scores, labels):
        x0 = min(float(width), max(0.0, float(box[0])))
        y0 = min(float(height), max(0.0, float(box[1])))
        x1 = min(float(width), max(0.0, float(box[2])))
        y1 = min(float(height), max(0.0, float(box[3])))
        category_id = int(label)
        confidence = float(score)
        if (
            not all(math.isfinite(value) for value in (x0, y0, x1, y1, confidence))
            or x1 <= x0
            or y1 <= y0
            or not 0 <= category_id < 25
            or confidence < 0.001 - 1e-12
        ):
            raise ValueError(f"非法 P2 预测: image_id={image_id}")
        output.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": confidence,
            }
        )
    return output


def run_infer(config_path: Path) -> int:
    config = _load_yaml(config_path)
    model_config = _mapping(config.get("model"), "model")
    actual_model = dict(model_config)
    checkpoint = _resolve(
        actual_model.pop("checkpoint", None),
        relative_to=config_path.parent,
        field="model.checkpoint",
    )
    input_config = _mapping(config.get("input"), "input")
    if input_config.get("split") != "val":
        raise ValueError("正式 OOF 只允许 split=val")
    data_root = _resolve(
        input_config.get("data_root"), relative_to=config_path.parent, field="input.data_root"
    )
    manifest = _resolve(
        input_config.get("manifest"), relative_to=config_path.parent, field="input.manifest"
    )
    _, val_records, held_out_fold = load_split_view(manifest, data_root=data_root)
    output = _resolve(
        config.get("output_json"), relative_to=config_path.parent, field="output_json"
    )
    runtime_path = output.with_suffix(".runtime.json")
    if output.exists() or runtime_path.exists():
        raise FileExistsError("推理产物已存在，禁止覆盖")

    _validate_ultralytics_version()
    import torch
    from ultralytics import YOLO

    # Import top-level classes before torch checkpoint deserialization.
    from rsdet.models.ibs_sampling import IBSDown, IBSUp

    model = YOLO(str(checkpoint))
    layers = model.model.model
    has_ibs_pair = isinstance(layers[17], IBSUp) and isinstance(layers[20], IBSDown)
    experiment_key = "Y3" if has_ibs_pair else "P2"
    contract = FORMAL_INFERENCE_CONTRACTS[experiment_key]
    if actual_model != contract["model"]:
        raise ValueError("推理 model 与冻结合同不一致")
    for field in (
        "device",
        "batch_size",
        "tiling",
        "score_thresholds",
        "fine_score_thresholds",
        "evaluation",
    ):
        if config.get(field) != contract[field]:
            raise ValueError(f"推理 {field} 与冻结合同不一致")
    if audit_p2_architecture(model)["detect_strides"] != list(EXPECTED_STRIDES):
        raise ValueError("推理 checkpoint 不是冻结 P2 结构")
    batch_size = int(config["batch_size"])
    predictions: list[dict[str, Any]] = []
    started = time.time()
    per_batch_seconds: list[float] = []
    for offset in range(0, len(val_records), batch_size):
        batch = val_records[offset : offset + batch_size]
        before = time.time()
        results = list(
            model.predict(
                source=[str(item["image_path"]) for item in batch],
                imgsz=int(model_config["imgsz"]),
                conf=float(model_config["confidence"]),
                iou=float(model_config["iou"]),
                max_det=int(model_config["max_detections"]),
                device=str(config["device"]),
                half=bool(model_config["half"]),
                agnostic_nms=bool(model_config["agnostic_nms"]),
                verbose=False,
                stream=False,
            )
        )
        per_batch_seconds.append(time.time() - before)
        if len(results) != len(batch):
            raise ValueError("推理结果数与输入图像数不一致")
        for item, result in zip(batch, results):
            with Image.open(item["image_path"]) as source_image:
                expected_size = source_image.size[::-1]
            if tuple(int(value) for value in result.orig_shape) != expected_size:
                raise ValueError(f"image_id={item['image_id']} 推理原图尺寸不一致")
            records = _result_records(result, int(item["image_id"]))
            if len(records) > int(model_config["max_detections"]):
                raise ValueError("单图预测超过 max_detections")
            predictions.extend(records)
    elapsed = time.time() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(predictions, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    runtime = {
        "schema_version": INFERENCE_RUNTIME_SCHEMA_VERSION,
        "status": "complete",
        "model_key": experiment_key,
        "quality_module": "ibs_p2_pair_v1" if has_ibs_pair else None,
        "held_out_fold": held_out_fold,
        "images": len(val_records),
        "proposal_count": len(predictions),
        "elapsed_seconds": elapsed,
        "batch_seconds": per_batch_seconds,
        "environment": _versions(torch),
        "artifacts": {
            "config": {"path": str(config_path.resolve()), "sha256": sha256_file(config_path)},
            "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
            "predictions": {"path": str(output), "sha256": sha256_file(output)},
        },
    }
    atomic_write_json(runtime_path, runtime)
    logger.info(
        "Y2 P2 fold%d inference complete: images=%d proposals=%d",
        held_out_fold,
        len(val_records),
        len(predictions),
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Y2 formal YOLO26-s P2 runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--gate-decision", type=Path, default=None)
    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "train":
            gate = args.gate_decision
            return run_train(
                args.config.expanduser().resolve(),
                gate_decision=None if gate is None else gate.expanduser().resolve(),
            )
        return run_infer(args.config.expanduser().resolve())
    except (ImportError, OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        logger.exception("Y2 %s 失败: %s", args.command, error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
