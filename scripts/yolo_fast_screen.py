#!/usr/bin/env python3
"""Paired, fold-local fast screening for YOLO architecture ideas.

The screen is deliberately exploratory.  It trains a 40-epoch control and a
40-epoch candidate on the same fold, emits low-threshold predictions, and
uses official matching only to decide whether a candidate deserves a formal
three-fold run.  It never replaces the formal CV3 contract.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from PIL import Image

from rsdet.analysis.oof_detection import (
    build_threshold_curve,
    load_formal_ground_truth,
    select_exploratory_workpoint,
)
from rsdet.evaluation.coco_metric import load_coco_predictions
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.experiments.cv3_oof import (
    atomic_write_json,
    build_fold_view_document,
    load_cv3_manifest,
    sha256_file,
)
from rsdet.utils.config import load_config
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="yolo_fast_screen")

CONTRACT_VERSION = "yolo_fast_screen_v1"
CV3_SHA = "27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331"
FORMAL_CROP_SHA = "a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128"
WEIGHT_SHA = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
EXPECTED_ULTRALYTICS = "8.4.103"
SCREEN_EPOCHS = 40
SCREEN_FOLDS = (0, 1)
SCREEN_ARGS = {
    "imgsz": 1024,
    "batch": 12,
    "workers": 8,
    "optimizer": "AdamW",
    "lr0": 0.002,
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "cos_lr": True,
    "amp": True,
    "deterministic": True,
    "patience": 0,
    "val": False,
    "plots": False,
    "close_mosaic": 5,
}
CANDIDATES = {
    "M1S": {
        "architecture": "yolo26s.yaml",
        "strides": (8.0, 16.0, 32.0),
        "sources": (16, 19, 22),
        "parameter_count": 10_009_784,
    },
    "Y2S": {
        "architecture": "yolo26s-p2.yaml",
        "strides": (4.0, 8.0, 16.0, 32.0),
        "sources": (19, 22, 25, 28),
        "parameter_count": 9_765_856,
    },
}


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} 必须是对象")
    return dict(value)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(payload, "配置顶层")


def _resolve(value: Any, *, relative_to: Path, field: str) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} 不能为空")
    path = Path(text).expanduser()
    return (path if path.is_absolute() else relative_to / path).resolve()


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
        raise ValueError(f"图像路径不含 images: {image_path}") from error
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_split_view(
    path: Path, *, data_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("samples"), list):
        raise ValueError("split view 必须包含 samples")
    fold = int(payload.get("held_out_fold", -1))
    if fold not in SCREEN_FOLDS:
        raise ValueError(f"快筛只允许 fold {SCREEN_FOLDS}")
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in payload["samples"]:
        item = _mapping(raw, "split sample")
        image_id = int(item["image_id"])
        if image_id in seen:
            raise ValueError(f"重复 image_id={image_id}")
        source = _safe_source(data_root, str(item["relative_path"]))
        label = _label_path(source)
        if not label.is_file():
            raise FileNotFoundError(label)
        record = {"image_id": image_id, "image_path": source, "label_path": label}
        split = str(item.get("split"))
        if split == "train":
            train.append(record)
        elif split == "val":
            val.append(record)
        else:
            raise ValueError("split 必须是 train/val")
        seen.add(image_id)
    if len(seen) != 4481 or not train or not val:
        raise ValueError("快筛 split view 必须完整覆盖 4481 图")
    return train, val, fold


def materialize_dataset(
    train: Sequence[Mapping[str, Any]],
    val: Sequence[Mapping[str, Any]],
    output: Path,
) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    train_txt = output / "train.txt"
    val_txt = output / "val.txt"
    train_txt.write_text("\n".join(str(x["image_path"]) for x in train) + "\n")
    val_txt.write_text("\n".join(str(x["image_path"]) for x in val) + "\n")
    dataset = output / "dataset.yaml"
    dataset.write_text(
        yaml.safe_dump(
            {
                "train": str(train_txt.resolve()),
                "val": str(val_txt.resolve()),
                "nc": 25,
                "names": [str(index) for index in range(25)],
            },
            sort_keys=False,
        )
    )
    return dataset


def _environment(torch_module: Any) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": str(torch_module.__version__),
        "cuda_runtime": str(torch_module.version.cuda),
        "cudnn": int(torch_module.backends.cudnn.version() or 0),
        "ultralytics": importlib.metadata.version("ultralytics"),
        "gpu": torch_module.cuda.get_device_name(0),
    }


def _audit_architecture(
    model: Any, key: str, *, check_parameter_count: bool = True
) -> dict[str, Any]:
    spec = CANDIDATES[key]
    torch_model = model.model
    head = torch_model.model[-1]
    strides = tuple(float(value) for value in head.stride.detach().cpu().tolist())
    sources = tuple(int(value) for value in head.f)
    parameters = sum(int(parameter.numel()) for parameter in torch_model.parameters())
    if (
        strides != spec["strides"]
        or sources != spec["sources"]
        or (check_parameter_count and parameters != spec["parameter_count"])
    ):
        raise ValueError(
            f"{key} 构图不符合冻结规格: strides={strides}, sources={sources}, params={parameters}"
        )
    return {
        "status": "pass",
        "candidate_key": key,
        "architecture": spec["architecture"],
        "detect_strides": list(strides),
        "detect_sources": list(sources),
        "parameter_count": parameters,
    }


def _snapshot(model: Any) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


def _transfer(before: Mapping[str, Any], model: Any) -> dict[str, Any]:
    changed = []
    changed_numel = 0
    total = sum(int(value.numel()) for value in model.parameters())
    for name, value in model.named_parameters():
        old = before.get(name)
        if old is not None and old.shape == value.shape and not old.equal(value.detach().cpu()):
            changed.append(name)
            changed_numel += int(value.numel())
    if not changed:
        raise ValueError("预训练权重未迁移到候选模型")
    return {
        "changed_parameter_tensor_count": len(changed),
        "changed_parameter_numel": changed_numel,
        "changed_fraction": changed_numel / total,
    }


def prepare_view(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    source, samples = load_cv3_manifest(
        args.cv3_manifest,
        expected_sha256=CV3_SHA,
        expected_image_count=4481,
    )
    view = build_fold_view_document(
        source,
        samples,
        held_out_fold=args.fold,
        source_manifest_sha256=CV3_SHA,
    )
    atomic_write_json(output, view)
    print(f"YOLO_FAST_SCREEN_PREPARE_PASS fold={args.fold} output={output}")
    return 0


def train(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser().resolve()
    config = _load_yaml(config_path)
    if config.get("screening_contract") != CONTRACT_VERSION:
        raise ValueError("不是冻结快筛配置")
    key = str(config.get("candidate_key", "")).upper()
    if key not in CANDIDATES:
        raise ValueError(f"candidate_key 必须是 {sorted(CANDIDATES)}")
    model_config = _mapping(config.get("model"), "model")
    if model_config.get("architecture") != CANDIDATES[key]["architecture"]:
        raise ValueError("候选键与 architecture 不一致")
    pretrained = _resolve(
        model_config.get("weights"), relative_to=config_path.parent, field="model.weights"
    )
    if sha256_file(pretrained) != WEIGHT_SHA:
        raise ValueError("预训练权重 SHA 不一致")
    train_config = _mapping(config.get("train"), "train")
    if (
        train_config.get("device") != "cuda:0"
        or train_config.get("checkpoint_selection") != "last"
        or int(train_config.get("epochs", -1)) != SCREEN_EPOCHS
        or _mapping(train_config.get("args"), "train.args") != SCREEN_ARGS
    ):
        raise ValueError("训练参数不符合 yolo_fast_screen_v1")
    output = _resolve(config.get("output_dir"), relative_to=config_path.parent, field="output")
    run_dir = output / "runs" / "screen"
    if run_dir.exists() or (output / "train_summary.json").exists():
        raise FileExistsError("快筛训练产物已存在，禁止覆盖/resume")
    data = _mapping(config.get("data"), "data")
    data_root = _resolve(data.get("root"), relative_to=config_path.parent, field="data.root")
    split = _resolve(data.get("manifest"), relative_to=config_path.parent, field="data.manifest")
    train_records, val_records, fold = load_split_view(split, data_root=data_root)
    dataset = materialize_dataset(train_records, val_records, output / "prepared_data")
    if importlib.metadata.version("ultralytics") != EXPECTED_ULTRALYTICS:
        raise ValueError("Ultralytics 版本不一致")
    import torch
    from ultralytics import YOLO

    model = YOLO(str(CANDIDATES[key]["architecture"]))
    before = _snapshot(model.model)
    architecture = _audit_architecture(model, key)
    model.load(str(pretrained))
    architecture["pretrained_transfer"] = _transfer(before, model.model)
    atomic_write_json(output / "architecture_audit.json", architecture)
    started = time.time()
    model.train(
        **SCREEN_ARGS,
        epochs=SCREEN_EPOCHS,
        device="cuda:0",
        seed=int(config["seed"]),
        data=str(dataset),
        project=str((output / "runs").resolve()),
        name="screen",
        exist_ok=False,
    )
    last = run_dir / "weights" / "last.pt"
    results = run_dir / "results.csv"
    if not last.is_file() or not results.is_file():
        raise FileNotFoundError("快筛训练产物不完整")
    with results.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != SCREEN_EPOCHS:
        raise ValueError(f"快筛必须恰好 {SCREEN_EPOCHS} epoch")
    atomic_write_json(
        output / "train_summary.json",
        {
            "contract_version": CONTRACT_VERSION,
            "status": "complete_screening_only",
            "formal_admission": False,
            "candidate_key": key,
            "held_out_fold": fold,
            "epochs": SCREEN_EPOCHS,
            "seed": int(config["seed"]),
            "elapsed_seconds": time.time() - started,
            "checkpoint": {"path": str(last), "sha256": sha256_file(last)},
            "environment": _environment(torch),
        },
    )
    return 0


def _clip_prediction_box(box: Sequence[float], *, width: int, height: int) -> list[float] | None:
    x0, y0 = max(0.0, float(box[0])), max(0.0, float(box[1]))
    x1, y1 = min(float(width), float(box[2])), min(float(height), float(box[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def _prediction_records(result: Any, image_id: int) -> tuple[list[dict[str, Any]], int]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], 0
    output = []
    filtered_degenerate = 0
    for box, score, label in zip(
        boxes.xyxy.detach().cpu().tolist(),
        boxes.conf.detach().cpu().tolist(),
        boxes.cls.detach().cpu().tolist(),
    ):
        height, width = result.orig_shape
        clipped = _clip_prediction_box(box, width=width, height=height)
        if clipped is None:
            filtered_degenerate += 1
            continue
        output.append(
            {
                "image_id": image_id,
                "category_id": int(label),
                "bbox": clipped,
                "score": float(score),
            }
        )
    return output, filtered_degenerate


def infer(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser().resolve()
    config = _load_yaml(config_path)
    if config.get("screening_contract") != CONTRACT_VERSION:
        raise ValueError("不是冻结快筛推理配置")
    key = str(config.get("candidate_key", "")).upper()
    if key not in CANDIDATES:
        raise ValueError("非法 candidate_key")
    model_config = _mapping(config.get("model"), "model")
    checkpoint = _resolve(
        model_config.get("checkpoint"), relative_to=config_path.parent, field="checkpoint"
    )
    expected_model = {
        "family": "yolo",
        "imgsz": 1024,
        "confidence": 0.001,
        "iou": 0.70,
        "max_detections": 500,
        "half": True,
        "agnostic_nms": False,
    }
    actual_model = dict(model_config)
    actual_model.pop("checkpoint", None)
    if actual_model != expected_model or config.get("device") != "cuda:0":
        raise ValueError("推理参数不符合快筛合同")
    if int(config.get("batch_size", -1)) != 8:
        raise ValueError("快筛 batch_size 必须为 8")
    input_config = _mapping(config.get("input"), "input")
    if input_config.get("split") != "val":
        raise ValueError("快筛只允许 held-out val")
    data_root = _resolve(
        input_config.get("data_root"), relative_to=config_path.parent, field="data_root"
    )
    split = _resolve(input_config.get("manifest"), relative_to=config_path.parent, field="manifest")
    _, val_records, fold = load_split_view(split, data_root=data_root)
    output = _resolve(config.get("output_json"), relative_to=config_path.parent, field="output")
    runtime_path = output.with_suffix(".runtime.json")
    if output.exists() or runtime_path.exists():
        raise FileExistsError("快筛推理产物已存在")
    if importlib.metadata.version("ultralytics") != EXPECTED_ULTRALYTICS:
        raise ValueError("Ultralytics 版本不一致")
    import torch
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    _audit_architecture(model, key, check_parameter_count=False)
    predictions: list[dict[str, Any]] = []
    per_batch = []
    filtered_degenerate_count = 0
    started = time.time()
    batch_size = int(config["batch_size"])
    for offset in range(0, len(val_records), batch_size):
        batch = val_records[offset : offset + batch_size]
        before = time.time()
        results = model.predict(
            source=[str(item["image_path"]) for item in batch],
            imgsz=1024,
            conf=0.001,
            iou=0.70,
            max_det=500,
            device="cuda:0",
            half=True,
            agnostic_nms=False,
            verbose=False,
            stream=False,
        )
        per_batch.append(time.time() - before)
        for item, result in zip(batch, results, strict=True):
            with Image.open(item["image_path"]) as image:
                if tuple(result.orig_shape) != image.size[::-1]:
                    raise ValueError("推理图像尺寸不一致")
            records, filtered = _prediction_records(result, int(item["image_id"]))
            predictions.extend(records)
            filtered_degenerate_count += filtered
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(predictions, separators=(",", ":")) + "\n")
    atomic_write_json(
        runtime_path,
        {
            "contract_version": CONTRACT_VERSION,
            "status": "complete_screening_only",
            "formal_admission": False,
            "candidate_key": key,
            "held_out_fold": fold,
            "images": len(val_records),
            "proposal_count": len(predictions),
            "filtered_degenerate_count": filtered_degenerate_count,
            "elapsed_seconds": time.time() - started,
            "batch_seconds": per_batch,
            "checkpoint_sha256": sha256_file(checkpoint),
            "predictions_sha256": sha256_file(output),
            "environment": _environment(torch),
        },
    )
    return 0


def _macro(row: Mapping[str, Any], names: Sequence[str], suffix: str) -> float:
    return sum(float(row[f"{name}_{suffix}"]) for name in names) / len(names)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _vehicle_no_candidate(
    gt: Mapping[int, Sequence[Mapping[str, Any]]],
    pred: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    vehicle_id: int,
) -> dict[str, int]:
    total = 0
    no_candidate = 0
    for image_id, records in gt.items():
        candidates = [
            item for item in pred.get(image_id, ()) if int(item["category_id"]) == vehicle_id
        ]
        for item in records:
            if int(item["category_id"]) != vehicle_id:
                continue
            total += 1
            if not any(
                _iou(item["bbox_xyxy"], candidate["bbox_xyxy"]) >= 0.35 for candidate in candidates
            ):
                no_candidate += 1
    return {
        "vehicle_gt": total,
        "no_candidate": no_candidate,
        "candidate_covered": total - no_candidate,
    }


def build_screen_decision(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    deltas = {
        key: float(candidate[key]) - float(baseline[key])
        for key in (
            "overall_recall",
            "overall_fdr",
            "macro_recall",
            "macro_fdr",
            "vehicle_recall",
            "vehicle_fdr",
            "vehicle_floor_recall",
        )
    }
    no_candidate_reduction = int(baseline["vehicle_no_candidate"]) - int(
        candidate["vehicle_no_candidate"]
    )
    checks = {
        "overall_recall_safety": deltas["overall_recall"] >= -0.015,
        "overall_fdr_safety": deltas["overall_fdr"] <= 0.02,
        "macro_recall_safety": deltas["macro_recall"] >= -0.02,
        "vehicle_signal": (
            deltas["vehicle_recall"] >= 0.02
            or deltas["vehicle_floor_recall"] >= 0.02
            or no_candidate_reduction >= 5
        ),
    }
    viable = all(checks.values())
    strong = viable and (
        deltas["vehicle_recall"] >= 0.03
        or (deltas["vehicle_floor_recall"] >= 0.03 and no_candidate_reduction >= 5)
    )
    if strong:
        next_action = "promising_for_formal_cv3"
    elif viable:
        next_action = "promising_for_second_screen_fold"
    else:
        next_action = "stop_candidate"
    return {
        "contract_version": CONTRACT_VERSION,
        "scientific_scope": "screening_only_not_formal_admission",
        "status": "complete",
        "next_action": next_action,
        "formal_admission": False,
        "checks": checks,
        "deltas_candidate_minus_control": deltas,
        "vehicle_no_candidate_reduction": no_candidate_reduction,
    }


def evaluate(args: argparse.Namespace) -> int:
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    formal = load_formal_ground_truth(args.formal_crop, expected_sha256=FORMAL_CROP_SHA)
    source, samples = load_cv3_manifest(args.cv3_manifest, expected_sha256=CV3_SHA)
    del source
    image_ids = {sample.image_id for sample in samples if sample.fold == args.fold}
    gt = {image_id: list(formal.boxes.get(image_id, ())) for image_id in image_ids}
    baseline_pred = load_coco_predictions(args.baseline_predictions)
    candidate_pred = load_coco_predictions(args.candidate_predictions)
    for name, records in (("baseline", baseline_pred), ("candidate", candidate_pred)):
        if not set(records) <= image_ids:
            raise ValueError(f"{name} 含 held-out fold 外 image_id")
    thresholds = [round(0.001 + 0.005 * index, 6) for index in range(61)]
    results: dict[str, Any] = {}
    vehicle_ids = [key for key, value in protocol.category_mapping.items() if value == "vehicle"]
    if len(vehicle_ids) != 1:
        raise ValueError("必须恰好有一个 vehicle category_id")
    for name, predictions in (("baseline", baseline_pred), ("candidate", candidate_pred)):
        curve, parity = build_threshold_curve(
            gt,
            predictions,
            thresholds=thresholds,
            protocol=protocol,
        )
        workpoint = select_exploratory_workpoint(
            curve,
            recall_min=protocol.recall_min,
            fdr_max=protocol.fdr_max,
        )
        row = workpoint["metrics"]
        floor = curve[0]
        coverage = _vehicle_no_candidate(gt, predictions, vehicle_id=vehicle_ids[0])
        results[name] = {
            "threshold": float(workpoint["threshold"]),
            "overall_recall": float(row["overall_recall"]),
            "overall_fdr": float(row["overall_fdr"]),
            "macro_recall": _macro(row, protocol.class_names, "recall"),
            "macro_fdr": _macro(row, protocol.class_names, "fdr"),
            "vehicle_recall": float(row["vehicle_recall"]),
            "vehicle_fdr": float(row["vehicle_fdr"]),
            "vehicle_floor_recall": float(floor["vehicle_recall"]),
            "vehicle_no_candidate": coverage["no_candidate"],
            "vehicle_gt": coverage["vehicle_gt"],
            "official_gate_at_exploratory_workpoint": bool(workpoint["official_gate_passed"]),
            "threshold_curve": curve,
            "parity": parity,
        }
    decision = build_screen_decision(results["baseline"], results["candidate"])
    payload = {
        "contract_version": CONTRACT_VERSION,
        "status": "complete_screening_only",
        "formal_admission": False,
        "held_out_fold": args.fold,
        "paired_control_required": True,
        "same_fold_threshold_selection": True,
        "results": results,
        "decision": decision,
        "artifacts": {
            "baseline_predictions_sha256": sha256_file(args.baseline_predictions),
            "candidate_predictions_sha256": sha256_file(args.candidate_predictions),
            "formal_crop_sha256": sha256_file(args.formal_crop),
            "cv3_manifest_sha256": sha256_file(args.cv3_manifest),
        },
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    atomic_write_json(output, payload)
    print(f"YOLO_FAST_SCREEN_EVALUATE_PASS next_action={decision['next_action']}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO paired fast-screen pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--cv3-manifest", type=Path, required=True)
    prepare_parser.add_argument("--fold", type=int, choices=SCREEN_FOLDS, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    infer_parser = sub.add_parser("infer")
    infer_parser.add_argument("--config", type=Path, required=True)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--project-config", type=Path, required=True)
    eval_parser.add_argument("--cv3-manifest", type=Path, required=True)
    eval_parser.add_argument("--formal-crop", type=Path, required=True)
    eval_parser.add_argument("--fold", type=int, choices=SCREEN_FOLDS, required=True)
    eval_parser.add_argument("--baseline-predictions", type=Path, required=True)
    eval_parser.add_argument("--candidate-predictions", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return {"prepare": prepare_view, "train": train, "infer": infer, "evaluate": evaluate}[
            args.command
        ](args)
    except (ImportError, OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        logger.exception("快筛 %s 失败: %s", args.command, error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
