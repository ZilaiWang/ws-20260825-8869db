"""CV3 fold views and framework-independent OOF delivery audits.

The model repository currently consumes a ``split=train/val`` manifest, while
the authoritative project split uses ``fold=0/1/2``.  This module bridges the
two contracts without modifying or copying source images.  It also validates
that every OOF prediction was produced by the model whose held-out fold
contains that image.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from rsdet.predictions import (
    load_coco_prediction_records,
    validate_coco_prediction_records,
)

OOF_CONTRACT_VERSION = "cv3_oof_v1"
INFERENCE_RUNTIME_SCHEMA_VERSION = "rsdet_inference_runtime_v2"
EXPECTED_FOLD_COUNT = 3
EXPECTED_IMAGE_COUNT = 4481
EXPECTED_CATEGORY_IDS = tuple(range(25))
FORMAL_CROP_MANIFEST_VERSION = "formal_crop_manifest_v2"
FORMAL_CROP_MANIFEST_SHA256 = "a3bed44fa6166fa7ee67555ea81ba96024652c0bba81a67da056c78d3e484128"
FORMAL_DETECTION_DATA_LOCK_SHA256 = (
    "03a8d8b5c56062ea0be46434bbcb6de333ba97eb9648f487d86a4ad162e0e77a"
)
FORMAL_DETECTION_DATA_LOCK_VERSION = "formal_detection_data_lock_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORMAL_TRAINING_CONTRACTS: dict[str, dict[str, Any]] = {
    "M1": {
        "device": "cuda:0",
        "checkpoint_selection": "last",
        "train_args": {
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
        },
        "stage_args": {"close_mosaic": 20},
    },
    "M3": {
        "device": "cuda:0",
        "checkpoint_selection": "last",
        "train_args": {
            "imgsz": 1024,
            "batch": 4,
            "workers": 8,
            "optimizer": "AdamW",
            "lr0": 0.0002,
            "lrf": 0.01,
            "weight_decay": 0.0001,
            "warmup_epochs": 3,
            "cos_lr": True,
            "amp": True,
            "deterministic": True,
            "patience": 0,
            "val": False,
            "plots": False,
        },
        "stage_args": {},
    },
    "P2": {
        "device": "cuda:0",
        "checkpoint_selection": "last",
        "train_args": {
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
        },
        "stage_args": {"close_mosaic": 20},
    },
    "Y3": {
        "device": "cuda:0",
        "checkpoint_selection": "last",
        "train_args": {
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
        },
        "stage_args": {"close_mosaic": 20},
    },
}
FORMAL_INFERENCE_CONTRACTS: dict[str, dict[str, Any]] = {
    "M1": {
        "device": "cuda:0",
        "batch_size": 8,
        "model": {
            "adapter": "ultralytics",
            "family": "yolo",
            "imgsz": 1024,
            "confidence": 0.001,
            "iou": 0.70,
            "max_detections": 500,
            "half": True,
            "agnostic_nms": False,
        },
        "input": {"split": "val"},
        "tiling": {"enabled": False, "force": False},
        "score_thresholds": {
            "ship": 0.001,
            "aircraft": 0.001,
            "vehicle": 0.001,
        },
        "fine_score_thresholds": {},
        "evaluation": {
            "gt": "",
            "project_config": "configs/project.yaml",
            "metrics_output": "",
        },
    },
    "M3": {
        "device": "cuda:0",
        "batch_size": 4,
        "model": {
            "adapter": "ultralytics",
            "family": "rtdetr",
            "imgsz": 1024,
            "confidence": 0.001,
            "iou": 0.70,
            "max_detections": 300,
            "half": True,
            "agnostic_nms": False,
        },
        "input": {"split": "val"},
        "tiling": {"enabled": False, "force": False},
        "score_thresholds": {
            "ship": 0.001,
            "aircraft": 0.001,
            "vehicle": 0.001,
        },
        "fine_score_thresholds": {},
        "evaluation": {
            "gt": "",
            "project_config": "configs/project.yaml",
            "metrics_output": "",
        },
    },
    "P2": {
        "device": "cuda:0",
        "batch_size": 8,
        "model": {
            "adapter": "ultralytics",
            "family": "yolo",
            "imgsz": 1024,
            "confidence": 0.001,
            "iou": 0.70,
            "max_detections": 500,
            "half": True,
            "agnostic_nms": False,
        },
        "input": {"split": "val"},
        "tiling": {"enabled": False, "force": False},
        "score_thresholds": {
            "ship": 0.001,
            "aircraft": 0.001,
            "vehicle": 0.001,
        },
        "fine_score_thresholds": {},
        "evaluation": {
            "gt": "",
            "project_config": "configs/project.yaml",
            "metrics_output": "",
        },
    },
    "Y3": {
        "device": "cuda:0",
        "batch_size": 8,
        "model": {
            "adapter": "ultralytics",
            "family": "yolo",
            "imgsz": 1024,
            "confidence": 0.001,
            "iou": 0.70,
            "max_detections": 500,
            "half": True,
            "agnostic_nms": False,
        },
        "input": {"split": "val"},
        "tiling": {"enabled": False, "force": False},
        "score_thresholds": {
            "ship": 0.001,
            "aircraft": 0.001,
            "vehicle": 0.001,
        },
        "fine_score_thresholds": {},
        "evaluation": {
            "gt": "",
            "project_config": "configs/project.yaml",
            "metrics_output": "",
        },
    },
}
FORMAL_EXPERIMENT_SPECS: dict[str, dict[str, Any]] = {
    "M1": {
        "model_family": "yolo",
        "model_name": "yolo26s",
        "input_size": 1024,
        "foundation_epochs": 160,
        "low_score_threshold": 0.001,
        "max_detections": 500,
    },
    "M3": {
        "model_family": "rtdetr",
        "model_name": "rtdetr-l",
        "input_size": 1024,
        "foundation_epochs": 120,
        "low_score_threshold": 0.001,
        "max_detections": 300,
    },
    "P2": {
        "model_family": "yolo",
        "model_name": "yolo26s-p2",
        "input_size": 1024,
        "foundation_epochs": 160,
        "low_score_threshold": 0.001,
        "max_detections": 500,
    },
    "Y3": {
        "model_family": "yolo",
        "model_name": "yolo26s-p2-ibs-pair",
        "input_size": 1024,
        "foundation_epochs": 160,
        "low_score_threshold": 0.001,
        "max_detections": 500,
    },
}


@dataclass(frozen=True)
class CV3Sample:
    """One image in an authoritative grouped CV manifest."""

    image_id: int
    relative_path: str
    fold: int
    group_id: str
    group_rule: str


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Write deterministic UTF-8 JSON atomically."""

    content = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n").encode(
        "utf-8"
    )
    _atomic_write(Path(path), content)


def _safe_relative_path(value: Any, field: str) -> str:
    text = str(value).strip()
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} 必须是数据根目录内的安全相对路径: {value!r}")
    return path.as_posix()


def _sha256(value: Any, field: str) -> str:
    text = str(value).strip().lower()
    if SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{field} 必须是 64 位小写 SHA256")
    return text


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} 必须是正整数")
    return value


def load_cv3_manifest(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_image_count: int | None = EXPECTED_IMAGE_COUNT,
) -> tuple[dict[str, Any], tuple[CV3Sample, ...]]:
    """Load and validate an authoritative three-fold grouped manifest."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"CV3 manifest 不存在: {manifest_path}")
    actual_sha256 = sha256_file(manifest_path)
    if expected_sha256 is not None and actual_sha256 != _sha256(
        expected_sha256, "expected_manifest_sha256"
    ):
        raise ValueError(
            f"CV3 manifest SHA 不匹配: expected={expected_sha256}, actual={actual_sha256}"
        )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("CV3 manifest 顶层必须是对象")
    if int(document.get("fold_count", -1)) != EXPECTED_FOLD_COUNT:
        raise ValueError(f"CV3 fold_count 必须为 {EXPECTED_FOLD_COUNT}")
    raw_samples = document.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("CV3 manifest 必须包含非空 samples 列表")

    samples: list[CV3Sample] = []
    seen_ids: set[int] = set()
    group_folds: dict[str, int] = {}
    for index, record in enumerate(raw_samples):
        if not isinstance(record, Mapping):
            raise ValueError(f"samples[{index}] 必须是对象")
        try:
            image_id = int(record["image_id"])
            fold = int(record["fold"])
            group_id = str(record["group_id"]).strip()
        except KeyError as error:
            raise ValueError(f"samples[{index}] 缺少字段 {error.args[0]}") from error
        if isinstance(record.get("image_id"), bool) or image_id <= 0:
            raise ValueError(f"samples[{index}].image_id 必须是正整数")
        if image_id in seen_ids:
            raise ValueError(f"CV3 image_id 重复: {image_id}")
        if isinstance(record.get("fold"), bool) or fold not in range(EXPECTED_FOLD_COUNT):
            raise ValueError(f"samples[{index}].fold 必须是 0/1/2")
        if not group_id:
            raise ValueError(f"samples[{index}].group_id 不能为空")
        previous_fold = group_folds.setdefault(group_id, fold)
        if previous_fold != fold:
            raise ValueError(f"来源组跨折: {group_id} -> {previous_fold}/{fold}")
        samples.append(
            CV3Sample(
                image_id=image_id,
                relative_path=_safe_relative_path(
                    record.get("relative_path"), f"samples[{index}].relative_path"
                ),
                fold=fold,
                group_id=group_id,
                group_rule=str(record.get("group_rule", "")).strip(),
            )
        )
        seen_ids.add(image_id)

    if expected_image_count is not None and len(samples) != expected_image_count:
        raise ValueError(f"CV3 应覆盖 {expected_image_count} 张图，实际 {len(samples)} 张")
    if {sample.fold for sample in samples} != set(range(EXPECTED_FOLD_COUNT)):
        raise ValueError("CV3 三个 fold 必须均非空")
    return document, tuple(samples)


def build_fold_view_document(
    source_document: Mapping[str, Any],
    samples: Sequence[CV3Sample],
    *,
    held_out_fold: int,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    """Create a read-only ``split`` view compatible with the C model loader."""

    if held_out_fold not in range(EXPECTED_FOLD_COUNT):
        raise ValueError("held_out_fold 必须是 0/1/2")
    source_sha = _sha256(source_manifest_sha256, "source_manifest_sha256")
    source_version = str(source_document.get("version", "")).strip()
    if not source_version:
        raise ValueError("CV3 manifest version 不能为空")
    records = [
        {
            "image_id": sample.image_id,
            "relative_path": sample.relative_path,
            "split": "val" if sample.fold == held_out_fold else "train",
            "source_fold": sample.fold,
            "group_id": sample.group_id,
            "group_rule": sample.group_rule,
        }
        for sample in sorted(samples, key=lambda item: item.image_id)
    ]
    return {
        "version": f"{source_version}__heldout_fold{held_out_fold}_view_v1",
        "data_version": source_document.get("data_version", "official_raw_v1"),
        "view_contract": "cv3_fold_to_split_view_v1",
        "source_manifest_version": source_version,
        "source_manifest_sha256": source_sha,
        "held_out_fold": held_out_fold,
        "fold_count": EXPECTED_FOLD_COUNT,
        "train_images": sum(record["split"] == "train" for record in records),
        "val_images": sum(record["split"] == "val" for record in records),
        "samples": records,
    }


def prepare_oof_run_plan(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    model_key: str,
    model_family: str,
    model_name: str,
    seed: int,
    input_size: int,
    foundation_epochs: int,
    low_score_threshold: float,
    max_detections: int,
    pretrained_weight: str,
    pretrained_weight_sha256: str,
    detection_data_lock: str,
    detection_data_lock_sha256: str,
    expected_manifest_sha256: str | None = None,
    expected_image_count: int | None = EXPECTED_IMAGE_COUNT,
) -> dict[str, Any]:
    """Materialize three loader-compatible views and a frozen OOF run plan."""

    model_key = model_key.strip().upper()
    if model_key not in FORMAL_EXPERIMENT_SPECS:
        raise ValueError(f"model_key 必须是 {sorted(FORMAL_EXPERIMENT_SPECS)} 之一")
    if not model_family.strip() or not model_name.strip():
        raise ValueError("model_family/model_name 不能为空")
    if seed < 0 or input_size <= 0 or foundation_epochs <= 0 or max_detections <= 0:
        raise ValueError("seed/input_size/foundation_epochs/max_detections 参数非法")
    if not 0.0 < low_score_threshold <= 0.01:
        raise ValueError("低阈值必须位于 (0, 0.01]")
    actual_spec = {
        "model_family": model_family.strip().lower(),
        "model_name": model_name.strip().lower(),
        "input_size": input_size,
        "foundation_epochs": foundation_epochs,
        "low_score_threshold": low_score_threshold,
        "max_detections": max_detections,
    }
    if actual_spec != FORMAL_EXPERIMENT_SPECS[model_key]:
        raise ValueError(
            f"{model_key} 正式实验参数必须与预注册规格完全一致: "
            f"{FORMAL_EXPERIMENT_SPECS[model_key]}"
        )
    pretrained_sha = _sha256(pretrained_weight_sha256, "pretrained_weight_sha256")
    pretrained_path = Path(pretrained_weight).expanduser().resolve()
    if not pretrained_path.is_file():
        raise FileNotFoundError(f"原预训练权重不存在: {pretrained_path}")
    actual_pretrained_sha = sha256_file(pretrained_path)
    if actual_pretrained_sha != pretrained_sha:
        raise ValueError(
            f"原预训练权重 SHA 不匹配: expected={pretrained_sha}, actual={actual_pretrained_sha}"
        )
    data_lock_path = Path(detection_data_lock).expanduser().resolve()
    if not data_lock_path.is_file():
        raise FileNotFoundError(f"正式检测数据锁不存在: {data_lock_path}")
    data_lock_sha = _sha256(
        detection_data_lock_sha256,
        "detection_data_lock_sha256",
    )
    if (
        expected_image_count == EXPECTED_IMAGE_COUNT
        and data_lock_sha != FORMAL_DETECTION_DATA_LOCK_SHA256
    ):
        raise ValueError("正式 4481 图实验必须使用唯一冻结检测数据锁")
    if sha256_file(data_lock_path) != data_lock_sha:
        raise ValueError("正式检测数据锁 SHA 不匹配")
    data_lock_payload = json.loads(data_lock_path.read_text(encoding="utf-8"))
    if not isinstance(data_lock_payload, Mapping):
        raise ValueError("正式检测数据锁顶层必须是对象")
    if data_lock_payload.get("schema_version") != FORMAL_DETECTION_DATA_LOCK_VERSION:
        raise ValueError("正式检测数据锁 schema_version 非法")
    data_lock_summary = data_lock_payload.get("summary")
    data_lock_contract = data_lock_payload.get("contract")
    if not isinstance(data_lock_summary, Mapping) or not isinstance(data_lock_contract, Mapping):
        raise ValueError("正式检测数据锁缺少 summary/contract")
    lock_fingerprint = _sha256(
        data_lock_payload.get("lock_fingerprint"),
        "detection_data_lock.lock_fingerprint",
    )
    inventory_fingerprint = _sha256(
        data_lock_payload.get("inventory_fingerprint"),
        "detection_data_lock.inventory_fingerprint",
    )
    lock_body = dict(data_lock_payload)
    lock_body.pop("lock_fingerprint", None)
    if _canonical_json_sha256(lock_body) != lock_fingerprint:
        raise ValueError("正式检测数据锁内部 lock_fingerprint 不匹配")
    inventory = data_lock_payload.get("inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("正式检测数据锁缺少非空 inventory")
    if _canonical_json_sha256(inventory) != inventory_fingerprint:
        raise ValueError("正式检测数据锁内部 inventory_fingerprint 不匹配")
    lock_image_count = _positive_int(
        data_lock_summary.get("image_count"),
        "detection_data_lock.summary.image_count",
    )
    lock_label_count = _positive_int(
        data_lock_summary.get("label_file_count"),
        "detection_data_lock.summary.label_file_count",
    )
    lock_object_count = _positive_int(
        data_lock_summary.get("object_count"),
        "detection_data_lock.summary.object_count",
    )
    if lock_label_count != lock_image_count or len(inventory) != lock_image_count:
        raise ValueError("正式检测数据锁图像/标签/inventory 数量不一致")
    if (
        data_lock_summary.get("p02_formal_gt_equivalence") is not True
        or data_lock_summary.get("yolo_formal_gt_equivalence") is not True
    ):
        raise ValueError("正式检测数据锁未证明两项 GT 等价性")

    source_document, samples = load_cv3_manifest(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
        expected_image_count=expected_image_count,
    )
    source_sha = sha256_file(manifest_path)
    bound_cv3_sha = _sha256(
        data_lock_contract.get("cv3_manifest_sha256"),
        "detection_data_lock.contract.cv3_manifest_sha256",
    )
    if lock_image_count != len(samples):
        raise ValueError("正式检测数据锁图像数与 CV3 manifest 不一致")
    if bound_cv3_sha != source_sha:
        raise ValueError("正式检测数据锁未绑定当前 CV3 manifest")
    destination = Path(output_dir)
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(f"OOF 计划输出已存在且不是目录: {destination}")
        if any(destination.iterdir()):
            raise FileExistsError(f"OOF 计划目录非空，禁止覆盖: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    data_lock_plan_contract = {
        "path": str(data_lock_path),
        "sha256": data_lock_sha,
        "lock_fingerprint": lock_fingerprint,
        "inventory_fingerprint": inventory_fingerprint,
        "image_count": lock_image_count,
        "label_file_count": lock_label_count,
        "object_count": lock_object_count,
    }
    fold_entries: list[dict[str, Any]] = []
    for fold in range(EXPECTED_FOLD_COUNT):
        fold_dir = destination / f"fold_{fold}"
        view_path = fold_dir / "split_view.json"
        view = build_fold_view_document(
            source_document,
            samples,
            held_out_fold=fold,
            source_manifest_sha256=source_sha,
        )
        atomic_write_json(view_path, view)
        template = {
            "contract_version": OOF_CONTRACT_VERSION,
            "status": "template_not_executed",
            "model_key": model_key,
            "model_family": model_family,
            "model_name": model_name,
            "held_out_fold": fold,
            "seed": seed,
            "source_manifest_sha256": source_sha,
            "fold_view_manifest": "split_view.json",
            "fold_view_manifest_sha256": sha256_file(view_path),
            "initialization": {
                "pretrained_weight": str(pretrained_path),
                "pretrained_weight_sha256": pretrained_sha,
                "resume": False,
            },
            "detection_data_lock": dict(data_lock_plan_contract),
            "inference": {
                "input_size": input_size,
                "foundation_epochs": foundation_epochs,
                "low_score_threshold": low_score_threshold,
                "max_detections": max_detections,
                "subset": "val",
            },
        }
        atomic_write_json(fold_dir / "fold_metadata.template.json", template)
        fold_entries.append(
            {
                "fold": fold,
                "fold_dir": f"fold_{fold}",
                "split_view": f"fold_{fold}/split_view.json",
                "split_view_sha256": sha256_file(view_path),
                "train_images": view["train_images"],
                "val_images": view["val_images"],
            }
        )

    plan = {
        "contract_version": OOF_CONTRACT_VERSION,
        "status": "ready_for_three_independent_training_runs",
        "model_key": model_key,
        "model_family": model_family,
        "model_name": model_name,
        "seed": seed,
        "source_manifest": str(Path(manifest_path)),
        "source_manifest_version": source_document["version"],
        "source_manifest_sha256": source_sha,
        "fold_count": EXPECTED_FOLD_COUNT,
        "image_count": len(samples),
        "training_rule": (
            "Each fold starts independently from the same recorded official "
            "pretrained weight; the held-out fold is never used for checkpoint "
            "selection; fixed-epoch last checkpoints are required; resume and "
            "cross-fold checkpoint reuse are forbidden."
        ),
        "training_config_contract": FORMAL_TRAINING_CONTRACTS[model_key],
        "inference_config_contract": FORMAL_INFERENCE_CONTRACTS[model_key],
        "input_size": input_size,
        "foundation_epochs": foundation_epochs,
        "low_score_threshold": low_score_threshold,
        "max_detections": max_detections,
        "initial_pretrained_weight": str(pretrained_path),
        "initial_pretrained_weight_sha256": pretrained_sha,
        "detection_data_lock": data_lock_plan_contract,
        "folds": fold_entries,
    }
    atomic_write_json(destination / "oof_run_plan.json", plan)
    return plan


def _load_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} 必须是 YAML 对象")
    return payload


def _resolve_config_path(value: Any, config_path: Path, field: str) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} 不能为空")
    result = Path(text).expanduser()
    if not result.is_absolute():
        result = config_path.parent / result
    return result.resolve()


def _validate_training_config(
    path: Path,
    *,
    expected_family: str,
    expected_pretrained_weight: str,
    expected_pretrained_sha256: str,
    expected_manifest_sha256: str,
    expected_seed: int,
    expected_input_size: int,
    expected_foundation_epochs: int,
    expected_training_contract: Mapping[str, Any],
) -> None:
    config = _load_yaml_mapping(path, "resolved training config")
    model = config.get("model")
    data = config.get("data")
    train = config.get("train")
    if (
        not isinstance(model, Mapping)
        or not isinstance(data, Mapping)
        or not isinstance(train, Mapping)
    ):
        raise ValueError("训练配置必须包含 model/data/train 对象")
    if int(config.get("seed", -1)) != expected_seed:
        raise ValueError("训练配置 seed 与 OOF 计划不一致")
    if str(model.get("family", "")).strip().lower() != expected_family.lower():
        raise ValueError("训练配置 model.family 与 OOF 计划不一致")
    configured_weight = str(model.get("weights", "")).strip()
    if not configured_weight:
        raise ValueError("每折训练必须直接从计划中的原预训练权重开始")
    configured_weight_path = _resolve_config_path(configured_weight, path, "训练配置 model.weights")
    if not configured_weight_path.is_file():
        raise FileNotFoundError(f"训练配置原预训练权重不存在: {configured_weight_path}")
    if configured_weight_path != Path(expected_pretrained_weight).resolve():
        raise ValueError("训练配置未指向 OOF 计划冻结的原预训练权重路径")
    if sha256_file(configured_weight_path) != expected_pretrained_sha256:
        raise ValueError("训练配置原预训练权重 SHA 与 OOF 计划不一致")
    stages = train.get("stages")
    if not isinstance(stages, list) or len(stages) != 1:
        raise ValueError("正式 OOF 只允许一个 foundation 训练阶段")
    stage = stages[0]
    if not isinstance(stage, Mapping):
        raise ValueError("foundation stage 必须是对象")
    if str(stage.get("name", "")).strip() != "foundation":
        raise ValueError("正式 OOF 唯一训练阶段必须命名为 foundation")
    if bool(stage.get("balanced", False)):
        raise ValueError("正式 foundation OOF 禁止启用未入选的再平衡阶段")
    if "resume" in config or "resume" in train or "resume" in stage:
        raise ValueError("正式 OOF 配置不得包含 resume")
    if int(stage.get("epochs", -1)) != expected_foundation_epochs:
        raise ValueError("foundation epochs 与冻结 OOF 计划不一致")
    train_args = train.get("args")
    if not isinstance(train_args, Mapping):
        raise ValueError("训练配置缺少 train.args")
    if int(train_args.get("imgsz", -1)) != expected_input_size:
        raise ValueError("训练 input size 与冻结 OOF 计划不一致")
    if str(train.get("device", "")).strip() != str(expected_training_contract["device"]):
        raise ValueError("训练 device 与冻结 OOF 计划不一致")
    if str(train.get("checkpoint_selection", "")).strip() != str(
        expected_training_contract["checkpoint_selection"]
    ):
        raise ValueError("正式 OOF 必须使用固定 epoch 的 last checkpoint")
    expected_args = expected_training_contract.get("train_args")
    expected_stage_args = expected_training_contract.get("stage_args")
    if not isinstance(expected_args, Mapping) or dict(train_args) != dict(expected_args):
        raise ValueError("训练 train.args 与冻结完整配置合同不一致")
    stage_args = stage.get("args", {})
    if not isinstance(stage_args, Mapping) or not isinstance(expected_stage_args, Mapping):
        raise ValueError("训练 stage args 合同非法")
    if dict(stage_args) != dict(expected_stage_args):
        raise ValueError("训练 foundation stage.args 与冻结配置合同不一致")
    manifest_value = str(data.get("manifest", "")).strip()
    if not manifest_value:
        raise ValueError("训练配置 data.manifest 不能为空")
    manifest_path = _resolve_config_path(manifest_value, path, "训练配置 data.manifest")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"训练配置指向的 manifest 不存在: {manifest_path}")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("训练配置未指向本折冻结 split view")


def _validate_inference_config(
    path: Path,
    *,
    expected_family: str,
    expected_threshold: float,
    expected_max_detections: int,
    expected_manifest_sha256: str,
    expected_input_size: int,
    expected_checkpoint_path: Path,
    expected_predictions_path: Path,
    expected_inference_contract: Mapping[str, Any],
) -> None:
    config = _load_yaml_mapping(path, "resolved inference config")
    model = config.get("model")
    input_config = config.get("input")
    thresholds = config.get("score_thresholds")
    if not isinstance(model, Mapping) or not isinstance(input_config, Mapping):
        raise ValueError("推理配置必须包含 model/input 对象")
    if str(model.get("family", "")).strip().lower() != expected_family.lower():
        raise ValueError("推理配置 model.family 与 OOF 计划不一致")
    if int(model.get("imgsz", -1)) != expected_input_size:
        raise ValueError("推理 input size 与冻结 OOF 计划不一致")
    confidence = float(model.get("confidence", math.nan))
    if not math.isclose(confidence, expected_threshold, abs_tol=1e-12):
        raise ValueError("模型内部候选阈值必须等于冻结低阈值")
    if int(model.get("max_detections", -1)) != expected_max_detections:
        raise ValueError("推理 max_detections 与 OOF 计划不一致")
    configured_checkpoint = _resolve_config_path(
        model.get("checkpoint"), path, "推理配置 model.checkpoint"
    )
    if configured_checkpoint != expected_checkpoint_path.resolve():
        raise ValueError("推理配置 model.checkpoint 与交付 checkpoint 不是同一实体")
    configured_output = _resolve_config_path(
        config.get("output_json"), path, "推理配置 output_json"
    )
    if configured_output != expected_predictions_path.resolve():
        raise ValueError("推理配置 output_json 与交付 predictions 不是同一实体")
    if input_config.get("split") != "val":
        raise ValueError("每折 OOF 推理必须只读取 split=val")
    manifest_value = str(input_config.get("manifest", "")).strip()
    if not manifest_value:
        raise ValueError("推理配置 input.manifest 不能为空")
    manifest_path = _resolve_config_path(manifest_value, path, "推理配置 input.manifest")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"推理配置指向的 manifest 不存在: {manifest_path}")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("推理配置未指向本折冻结 split view")
    if not isinstance(thresholds, Mapping):
        raise ValueError("推理配置缺少 score_thresholds")
    for coarse in ("ship", "aircraft", "vehicle"):
        if not math.isclose(
            float(thresholds.get(coarse, math.nan)),
            expected_threshold,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{coarse} 输出阈值必须等于冻结低阈值")
    fine_thresholds = config.get("fine_score_thresholds", {})
    if fine_thresholds not in ({}, None):
        raise ValueError("正式低阈值 OOF 禁止使用细类独立阈值")

    expected_model = expected_inference_contract.get("model")
    expected_input = expected_inference_contract.get("input")
    if not isinstance(expected_model, Mapping) or not isinstance(expected_input, Mapping):
        raise ValueError("冻结推理配置合同非法")
    actual_model = dict(model)
    actual_model.pop("checkpoint", None)
    if actual_model != dict(expected_model):
        raise ValueError("推理 model 参数与冻结完整配置合同不一致")
    actual_input = dict(input_config)
    actual_input.pop("data_root", None)
    actual_input.pop("manifest", None)
    if actual_input != dict(expected_input):
        raise ValueError("推理 input 参数与冻结完整配置合同不一致")
    for field in (
        "device",
        "batch_size",
        "tiling",
        "score_thresholds",
        "fine_score_thresholds",
        "evaluation",
    ):
        if config.get(field) != expected_inference_contract.get(field):
            raise ValueError(f"推理 {field} 与冻结完整配置合同不一致")
    allowed_top_level = {
        "model",
        "device",
        "batch_size",
        "input",
        "tiling",
        "score_thresholds",
        "fine_score_thresholds",
        "output_json",
        "evaluation",
    }
    if set(config) != allowed_top_level:
        raise ValueError("正式推理配置顶层字段必须与冻结合同完全一致")


def _validate_train_summary(
    path: Path,
    *,
    expected_family: str,
    expected_pretrained_weight: str,
    expected_pretrained_sha256: str,
    expected_seed: int,
    expected_foundation_epochs: int,
    expected_checkpoint_path: Path,
    expected_training_contract: Mapping[str, Any],
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("train_summary.json 顶层必须是对象")
    if payload.get("dry_run") is not False:
        raise ValueError("正式 fold train_summary 必须记录 dry_run=false")
    if int(payload.get("seed", -1)) != expected_seed:
        raise ValueError("train_summary seed 与 OOF 计划不一致")
    if str(payload.get("model_family", "")).strip().lower() != expected_family.lower():
        raise ValueError("train_summary model_family 与 OOF 计划不一致")
    if payload.get("checkpoint_selection") != "last":
        raise ValueError("正式 train_summary 必须声明 checkpoint_selection=last")
    expected_pretrained = Path(expected_pretrained_weight).resolve()
    initial_weights = _resolve_config_path(
        payload.get("initial_weights"), path, "train_summary.initial_weights"
    )
    if initial_weights != expected_pretrained:
        raise ValueError("train_summary initial_weights 不是冻结原预训练权重")
    if not initial_weights.is_file():
        raise FileNotFoundError(f"train_summary 原预训练权重不存在: {initial_weights}")
    if sha256_file(initial_weights) != expected_pretrained_sha256:
        raise ValueError("train_summary 原预训练权重 SHA 不一致")
    stages = payload.get("stages")
    if not isinstance(stages, list) or len(stages) != 1:
        raise ValueError("train_summary 必须且只能包含 foundation 阶段")
    stage = stages[0]
    if not isinstance(stage, Mapping):
        raise ValueError("train_summary foundation stage 必须是对象")
    if str(stage.get("name", "")).strip() != "foundation":
        raise ValueError("train_summary 唯一阶段必须命名为 foundation")
    if stage.get("balanced") is not False:
        raise ValueError("train_summary foundation 不得使用 balanced 数据")
    input_weights = _resolve_config_path(
        stage.get("input_weights"), path, "train_summary stage.input_weights"
    )
    if input_weights != expected_pretrained:
        raise ValueError("train_summary stage.input_weights 存在跨折或续训")
    arguments = stage.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("train_summary foundation.arguments 必须是对象")
    if int(arguments.get("epochs", -1)) != expected_foundation_epochs:
        raise ValueError("train_summary foundation epochs 与计划不一致")
    if int(arguments.get("seed", -1)) != expected_seed:
        raise ValueError("train_summary foundation seed 与计划不一致")
    if arguments.get("resume") not in (None, False):
        raise ValueError("train_summary 表明正式 fold 使用了 resume")
    expected_arguments = expected_training_contract.get("train_args")
    expected_stage_arguments = expected_training_contract.get("stage_args")
    if not isinstance(expected_arguments, Mapping) or not isinstance(
        expected_stage_arguments, Mapping
    ):
        raise ValueError("train_summary 对应的训练合同非法")
    expected_scientific_arguments = {
        **dict(expected_arguments),
        **dict(expected_stage_arguments),
        "epochs": expected_foundation_epochs,
        "device": expected_training_contract["device"],
        "seed": expected_seed,
    }
    for key, expected_value in expected_scientific_arguments.items():
        if arguments.get(key) != expected_value:
            raise ValueError(f"train_summary foundation.arguments.{key} 与冻结合同不一致")
    allowed_dynamic = {"data", "project", "name", "exist_ok"}
    if set(arguments) - set(expected_scientific_arguments) - allowed_dynamic:
        raise ValueError("train_summary foundation.arguments 含未冻结的额外参数")
    if stage.get("checkpoint_selection") != "last":
        raise ValueError("train_summary foundation 未声明 last checkpoint")
    last_path = _resolve_config_path(stage.get("last"), path, "train_summary foundation.last")
    selected_path = _resolve_config_path(
        stage.get("selected_checkpoint"),
        path,
        "train_summary foundation.selected_checkpoint",
    )
    if (
        last_path != expected_checkpoint_path.resolve()
        or selected_path != expected_checkpoint_path.resolve()
    ):
        raise ValueError("train_summary foundation.last 与交付 checkpoint 不一致")


def _validate_runtime_provenance(
    path: Path,
    payload: Mapping[str, Any],
    *,
    expected_config_path: Path,
    expected_checkpoint_path: Path,
    expected_predictions_path: Path,
) -> None:
    if payload.get("schema_version") != INFERENCE_RUNTIME_SCHEMA_VERSION:
        raise ValueError(f"runtime schema_version 必须是 {INFERENCE_RUNTIME_SCHEMA_VERSION}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("runtime artifacts 必须是对象")
    expected = {
        "config": expected_config_path,
        "checkpoint": expected_checkpoint_path,
        "predictions": expected_predictions_path,
    }
    for name, expected_path in expected.items():
        artifact = artifacts.get(name)
        if not isinstance(artifact, Mapping):
            raise ValueError(f"runtime artifacts.{name} 必须是对象")
        artifact_path = _resolve_config_path(
            artifact.get("path"),
            path,
            f"runtime artifacts.{name}.path",
        )
        if artifact_path != expected_path.resolve():
            raise ValueError(f"runtime {name} path 与交付文件不是同一实体")
        recorded_sha = _sha256(
            artifact.get("sha256"),
            f"runtime artifacts.{name}.sha256",
        )
        if recorded_sha != sha256_file(expected_path):
            raise ValueError(f"runtime {name} SHA 与交付文件不一致")


def _validate_data_lock_verification(
    report_path: Path,
    *,
    plan_contract: Mapping[str, Any],
) -> None:
    lock_path = Path(str(plan_contract.get("path", ""))).expanduser().resolve()
    if not lock_path.is_file():
        raise FileNotFoundError(f"正式检测数据锁不存在: {lock_path}")
    expected_lock_sha = _sha256(
        plan_contract.get("sha256"),
        "plan detection_data_lock.sha256",
    )
    if sha256_file(lock_path) != expected_lock_sha:
        raise ValueError("正式检测数据锁在计划冻结后发生变化")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, Mapping) or report.get("status") != "pass":
        raise ValueError("本折检测数据锁全量 verify 未通过")
    if report.get("schema_version") != FORMAL_DETECTION_DATA_LOCK_VERSION:
        raise ValueError("本折检测数据锁 verify 报告 schema_version 非法")
    report_lock_path = Path(str(report.get("lock_path", ""))).expanduser().resolve()
    if report_lock_path != lock_path:
        raise ValueError("本折检测数据锁 verify 报告未绑定计划中的 lock 实体")
    expected_fields = {
        "lock_file_sha256": expected_lock_sha,
        "lock_fingerprint": _sha256(
            plan_contract.get("lock_fingerprint"),
            "plan detection_data_lock.lock_fingerprint",
        ),
        "inventory_fingerprint": _sha256(
            plan_contract.get("inventory_fingerprint"),
            "plan detection_data_lock.inventory_fingerprint",
        ),
        "image_count": _positive_int(
            plan_contract.get("image_count"),
            "plan detection_data_lock.image_count",
        ),
        "label_file_count": _positive_int(
            plan_contract.get("label_file_count"),
            "plan detection_data_lock.label_file_count",
        ),
        "object_count": _positive_int(
            plan_contract.get("object_count"),
            "plan detection_data_lock.object_count",
        ),
        "p02_formal_gt_equivalence": True,
        "yolo_formal_gt_equivalence": True,
    }
    mismatches = {
        field: {"expected": expected, "actual": report.get(field)}
        for field, expected in expected_fields.items()
        if report.get(field) != expected
    }
    if mismatches:
        raise ValueError(f"本折检测数据锁 verify 报告与计划不一致: {mismatches}")


def finalize_fold_delivery(
    *,
    plan_path: str | Path,
    held_out_fold: int,
    train_config_path: str | Path,
    train_summary_path: str | Path,
    infer_config_path: str | Path,
    environment_path: str | Path,
    checkpoint_path: str | Path,
    predictions_path: str | Path,
    runtime_path: str | Path,
    data_lock_verification_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate one completed fold and freeze its small delivery metadata."""

    plan_file = Path(plan_path)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if plan.get("contract_version") != OOF_CONTRACT_VERSION:
        raise ValueError("未知 OOF run plan contract")
    if held_out_fold not in range(EXPECTED_FOLD_COUNT):
        raise ValueError("held_out_fold 必须是 0/1/2")
    fold_plan = plan["folds"][held_out_fold]
    view_path = plan_file.parent / fold_plan["split_view"]
    if sha256_file(view_path) != fold_plan["split_view_sha256"]:
        raise ValueError("fold split view SHA 不匹配")
    view = json.loads(view_path.read_text(encoding="utf-8"))
    if int(view.get("held_out_fold", -1)) != held_out_fold:
        raise ValueError("fold split view 的 held_out_fold 不一致")
    val_ids = {
        int(record["image_id"]) for record in view["samples"] if record.get("split") == "val"
    }

    train_config = Path(train_config_path)
    train_summary = Path(train_summary_path)
    infer_config = Path(infer_config_path)
    environment = Path(environment_path)
    checkpoint = Path(checkpoint_path)
    predictions = Path(predictions_path)
    runtime = Path(runtime_path)
    data_lock_verification = Path(data_lock_verification_path)
    for artifact, label in (
        (train_config, "train config"),
        (train_summary, "train summary"),
        (infer_config, "infer config"),
        (environment, "environment"),
        (checkpoint, "checkpoint"),
        (predictions, "predictions"),
        (runtime, "runtime"),
        (data_lock_verification, "data lock verification"),
    ):
        if not artifact.is_file():
            raise FileNotFoundError(f"{label} 不存在: {artifact}")
    plan_data_lock = plan.get("detection_data_lock")
    if not isinstance(plan_data_lock, Mapping):
        raise ValueError("OOF run plan 缺少正式检测数据锁合同")
    _validate_data_lock_verification(
        data_lock_verification,
        plan_contract=plan_data_lock,
    )

    _validate_training_config(
        train_config,
        expected_family=str(plan["model_family"]),
        expected_pretrained_weight=str(plan["initial_pretrained_weight"]),
        expected_pretrained_sha256=str(plan["initial_pretrained_weight_sha256"]),
        expected_manifest_sha256=str(fold_plan["split_view_sha256"]),
        expected_seed=int(plan["seed"]),
        expected_input_size=int(plan["input_size"]),
        expected_foundation_epochs=int(plan["foundation_epochs"]),
        expected_training_contract=plan["training_config_contract"],
    )
    _validate_inference_config(
        infer_config,
        expected_family=str(plan["model_family"]),
        expected_threshold=float(plan["low_score_threshold"]),
        expected_max_detections=int(plan["max_detections"]),
        expected_manifest_sha256=str(fold_plan["split_view_sha256"]),
        expected_input_size=int(plan["input_size"]),
        expected_checkpoint_path=checkpoint,
        expected_predictions_path=predictions,
        expected_inference_contract=plan["inference_config_contract"],
    )
    _validate_train_summary(
        train_summary,
        expected_family=str(plan["model_family"]),
        expected_pretrained_weight=str(plan["initial_pretrained_weight"]),
        expected_pretrained_sha256=str(plan["initial_pretrained_weight_sha256"]),
        expected_seed=int(plan["seed"]),
        expected_foundation_epochs=int(plan["foundation_epochs"]),
        expected_checkpoint_path=checkpoint,
        expected_training_contract=plan["training_config_contract"],
    )
    records = load_coco_prediction_records(predictions)
    validate_coco_prediction_records(
        records,
        allowed_category_ids=EXPECTED_CATEGORY_IDS,
    )
    predicted_ids = {int(record["image_id"]) for record in records}
    unexpected = predicted_ids - val_ids
    if unexpected:
        raise ValueError(f"fold {held_out_fold} 预测包含非本折验证图像: {sorted(unexpected)[:10]}")
    for index, record in enumerate(records):
        score = float(record["score"])
        if score + 1e-12 < float(plan["low_score_threshold"]):
            raise ValueError(f"predictions[{index}] score 低于冻结候选阈值")
    prediction_counts: dict[int, int] = {}
    for record in records:
        image_id = int(record["image_id"])
        prediction_counts[image_id] = prediction_counts.get(image_id, 0) + 1
    over_limit = {
        image_id: count
        for image_id, count in prediction_counts.items()
        if count > int(plan["max_detections"])
    }
    if over_limit:
        raise ValueError(
            f"单图 proposal_count 超出冻结 max_detections: {sorted(over_limit.items())[:10]}"
        )

    runtime_payload = json.loads(runtime.read_text(encoding="utf-8"))
    if not isinstance(runtime_payload, Mapping):
        raise ValueError("runtime JSON 顶层必须是对象")
    if int(runtime_payload.get("images", -1)) != int(view["val_images"]):
        raise ValueError("runtime images 必须等于本折全部验证图像数")
    _validate_runtime_provenance(
        runtime,
        runtime_payload,
        expected_config_path=infer_config,
        expected_checkpoint_path=checkpoint,
        expected_predictions_path=predictions,
    )
    checkpoint_sha = sha256_file(checkpoint)
    metadata = {
        "contract_version": OOF_CONTRACT_VERSION,
        "status": "fold_delivery_complete",
        "run_plan_sha256": sha256_file(plan_file),
        "model_key": plan["model_key"],
        "model_family": plan["model_family"],
        "model_name": plan["model_name"],
        "held_out_fold": held_out_fold,
        "seed": int(plan["seed"]),
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "fold_view_manifest": str(view_path),
        "fold_view_manifest_sha256": sha256_file(view_path),
        "train_images": int(view["train_images"]),
        "val_images": int(view["val_images"]),
        "initialization": {
            "pretrained_weight": plan["initial_pretrained_weight"],
            "pretrained_weight_sha256": plan["initial_pretrained_weight_sha256"],
            "resume": False,
        },
        "inference": {
            "input_size": int(plan["input_size"]),
            "foundation_epochs": int(plan["foundation_epochs"]),
            "low_score_threshold": float(plan["low_score_threshold"]),
            "max_detections": int(plan["max_detections"]),
            "images_with_predictions": len(predicted_ids),
            "proposal_count": len(records),
        },
        "checkpoint_selection": "fixed_epoch_last",
        "detection_data_lock": dict(plan_data_lock),
        "artifacts": {
            "train_config": str(train_config),
            "train_config_sha256": sha256_file(train_config),
            "train_summary": str(train_summary),
            "train_summary_sha256": sha256_file(train_summary),
            "infer_config": str(infer_config),
            "infer_config_sha256": sha256_file(infer_config),
            "environment": str(environment),
            "environment_sha256": sha256_file(environment),
            "checkpoint": str(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_sha,
            "predictions": str(predictions),
            "predictions_sha256": sha256_file(predictions),
            "runtime": str(runtime),
            "runtime_sha256": sha256_file(runtime),
            "data_lock_verification": str(data_lock_verification),
            "data_lock_verification_sha256": sha256_file(data_lock_verification),
        },
    }
    atomic_write_json(output_path, metadata)
    return metadata


def _csv_bytes(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    with tempfile.SpooledTemporaryFile(mode="w+", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.seek(0)
        return handle.read().encode("utf-8")


def _load_formal_crop_image_sizes(path: Path) -> dict[int, tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"formal crop manifest 不存在: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != FORMAL_CROP_MANIFEST_SHA256:
        raise ValueError(
            "formal crop manifest 不是唯一冻结 formal_crop_manifest_v2: "
            f"expected={FORMAL_CROP_MANIFEST_SHA256}, actual={actual_sha}"
        )
    image_sizes: dict[int, tuple[int, int]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "manifest_version",
            "formal_image_id",
            "source_width",
            "source_height",
        }
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError("formal crop manifest 缺少版本或 image-size 字段")
        for line_number, row in enumerate(reader, 2):
            if row["manifest_version"] != FORMAL_CROP_MANIFEST_VERSION:
                raise ValueError(f"formal crop manifest 第 {line_number} 行版本不一致")
            image_id = int(row["formal_image_id"])
            size = (int(row["source_width"]), int(row["source_height"]))
            if image_id <= 0 or size[0] <= 0 or size[1] <= 0:
                raise ValueError(f"formal crop manifest 第 {line_number} 行尺寸非法")
            previous = image_sizes.setdefault(image_id, size)
            if previous != size:
                raise ValueError(f"formal image_id={image_id} 尺寸不一致")
    if not image_sizes:
        raise ValueError("formal crop manifest 不含任何 image size")
    return image_sizes


def _validate_coco_bbox_within_image(
    record: Mapping[str, Any],
    *,
    image_size: tuple[int, int],
    context: str,
) -> None:
    x, y, width, height = [float(value) for value in record["bbox"]]
    image_width, image_height = image_size
    tolerance = 1e-6
    if x < -tolerance or y < -tolerance:
        raise ValueError(f"{context} bbox 左上角为负")
    if x + width > image_width + tolerance:
        raise ValueError(f"{context} bbox 超出图像宽度 {image_width}")
    if y + height > image_height + tolerance:
        raise ValueError(f"{context} bbox 超出图像高度 {image_height}")


def audit_and_aggregate_oof(
    *,
    manifest_path: str | Path,
    plan_path: str | Path,
    run_root: str | Path,
    output_dir: str | Path,
    expected_manifest_sha256: str | None = None,
    expected_image_count: int | None = EXPECTED_IMAGE_COUNT,
    formal_crop_manifest_path: str | Path | None = None,
    diagnostic_without_formal_crop: bool = False,
) -> dict[str, Any]:
    """Audit all three fold deliveries and write the model-independent OOF set."""

    source_document, samples = load_cv3_manifest(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
        expected_image_count=expected_image_count,
    )
    source_sha = sha256_file(manifest_path)
    if formal_crop_manifest_path is None:
        if not diagnostic_without_formal_crop:
            raise ValueError(
                "正式 OOF aggregate 必须一次性绑定冻结 formal_crop_manifest_v2；"
                "仅诊断运行可显式启用 diagnostic_without_formal_crop"
            )
        image_sizes: dict[int, tuple[int, int]] | None = None
        crop_manifest: dict[str, Any] | None = None
    else:
        if diagnostic_without_formal_crop:
            raise ValueError("formal crop 与 diagnostic_without_formal_crop 不得同时启用")
        crop_path = Path(formal_crop_manifest_path)
        image_sizes = _load_formal_crop_image_sizes(crop_path)
        crop_manifest = {
            "version": FORMAL_CROP_MANIFEST_VERSION,
            "path": str(crop_path),
            "sha256": FORMAL_CROP_MANIFEST_SHA256,
        }
    plan_file = Path(plan_path)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    plan_sha = sha256_file(plan_file)
    if plan.get("contract_version") != OOF_CONTRACT_VERSION:
        raise ValueError("未知 OOF run plan contract")
    if plan.get("source_manifest_sha256") != source_sha:
        raise ValueError("OOF plan 与正式 CV3 manifest SHA 不一致")
    if int(plan.get("fold_count", -1)) != EXPECTED_FOLD_COUNT:
        raise ValueError("OOF plan fold_count 必须为 3")
    plan_data_lock = plan.get("detection_data_lock")
    if not isinstance(plan_data_lock, Mapping):
        raise ValueError("OOF plan 缺少正式检测数据锁合同")

    by_image = {sample.image_id: sample for sample in samples}
    all_image_ids = set(by_image)
    seen_oof_images: set[int] = set()
    all_predictions: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    checkpoint_shas: list[str] = []
    root = Path(run_root)
    for fold in range(EXPECTED_FOLD_COUNT):
        metadata_path = root / f"fold_{fold}" / "fold_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"缺少 fold metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "fold_delivery_complete":
            raise ValueError(f"fold {fold} delivery 尚未完成")
        if metadata.get("run_plan_sha256") != plan_sha:
            raise ValueError(f"fold {fold} run plan SHA 不一致")
        for key in ("model_key", "model_family", "model_name", "seed"):
            if metadata.get(key) != plan.get(key):
                raise ValueError(f"fold {fold} 的 {key} 与 run plan 不一致")
        if int(metadata.get("held_out_fold", -1)) != fold:
            raise ValueError(f"fold {fold} metadata held_out_fold 不一致")
        if metadata.get("source_manifest_sha256") != source_sha:
            raise ValueError(f"fold {fold} source manifest SHA 不一致")
        if metadata.get("checkpoint_selection") != "fixed_epoch_last":
            raise ValueError(f"fold {fold} 未使用固定 epoch last checkpoint")
        fold_plan = plan["folds"][fold]
        if metadata.get("fold_view_manifest_sha256") != fold_plan["split_view_sha256"]:
            raise ValueError(f"fold {fold} split view SHA 不一致")
        if int(metadata.get("train_images", -1)) != int(fold_plan["train_images"]):
            raise ValueError(f"fold {fold} train image count 不一致")
        if int(metadata.get("val_images", -1)) != int(fold_plan["val_images"]):
            raise ValueError(f"fold {fold} val image count 不一致")
        initialization = metadata.get("initialization")
        if not isinstance(initialization, Mapping):
            raise ValueError(f"fold {fold} 缺少 initialization")
        if initialization.get("resume") is not False:
            raise ValueError(f"fold {fold} 禁止 resume 或跨折复用 checkpoint")
        if (
            initialization.get("pretrained_weight_sha256")
            != plan["initial_pretrained_weight_sha256"]
        ):
            raise ValueError(f"fold {fold} 不是从冻结原预训练权重独立开始")
        if (
            Path(str(initialization.get("pretrained_weight", ""))).resolve()
            != Path(str(plan["initial_pretrained_weight"])).resolve()
        ):
            raise ValueError(f"fold {fold} 原预训练权重路径与计划不一致")
        inference = metadata.get("inference")
        if not isinstance(inference, Mapping):
            raise ValueError(f"fold {fold} 缺少 inference")
        for field in ("input_size", "foundation_epochs", "max_detections"):
            if int(inference.get(field, -1)) != int(plan[field]):
                raise ValueError(f"fold {fold} inference.{field} 与计划不一致")
        if not math.isclose(
            float(inference.get("low_score_threshold", math.nan)),
            float(plan["low_score_threshold"]),
            abs_tol=1e-12,
        ):
            raise ValueError(f"fold {fold} inference.low_score_threshold 与计划不一致")

        artifacts = metadata.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError(f"fold {fold} 缺少 artifacts")
        for artifact_name in (
            "train_config",
            "train_summary",
            "infer_config",
            "environment",
            "checkpoint",
            "predictions",
            "runtime",
            "data_lock_verification",
        ):
            artifact_path = Path(str(artifacts.get(artifact_name, "")))
            if not artifact_path.is_file():
                raise FileNotFoundError(f"fold {fold} {artifact_name} 文件不存在: {artifact_path}")
            if sha256_file(artifact_path) != artifacts.get(f"{artifact_name}_sha256"):
                raise ValueError(f"fold {fold} {artifact_name} SHA 不一致")
            if artifact_name == "checkpoint" and artifact_path.stat().st_size != int(
                artifacts.get("checkpoint_size_bytes", -1)
            ):
                raise ValueError(f"fold {fold} checkpoint size 不一致")
        checkpoint_path = Path(str(artifacts["checkpoint"]))
        prediction_path = Path(str(artifacts["predictions"]))
        if metadata.get("detection_data_lock") != plan_data_lock:
            raise ValueError(f"fold {fold} 检测数据锁合同与计划不一致")
        _validate_data_lock_verification(
            Path(str(artifacts["data_lock_verification"])),
            plan_contract=plan_data_lock,
        )
        _validate_training_config(
            Path(str(artifacts["train_config"])),
            expected_family=str(plan["model_family"]),
            expected_pretrained_weight=str(plan["initial_pretrained_weight"]),
            expected_pretrained_sha256=str(plan["initial_pretrained_weight_sha256"]),
            expected_manifest_sha256=str(fold_plan["split_view_sha256"]),
            expected_seed=int(plan["seed"]),
            expected_input_size=int(plan["input_size"]),
            expected_foundation_epochs=int(plan["foundation_epochs"]),
            expected_training_contract=plan["training_config_contract"],
        )
        _validate_inference_config(
            Path(str(artifacts["infer_config"])),
            expected_family=str(plan["model_family"]),
            expected_threshold=float(plan["low_score_threshold"]),
            expected_max_detections=int(plan["max_detections"]),
            expected_manifest_sha256=str(fold_plan["split_view_sha256"]),
            expected_input_size=int(plan["input_size"]),
            expected_checkpoint_path=checkpoint_path,
            expected_predictions_path=prediction_path,
            expected_inference_contract=plan["inference_config_contract"],
        )
        _validate_train_summary(
            Path(str(artifacts["train_summary"])),
            expected_family=str(plan["model_family"]),
            expected_pretrained_weight=str(plan["initial_pretrained_weight"]),
            expected_pretrained_sha256=str(plan["initial_pretrained_weight_sha256"]),
            expected_seed=int(plan["seed"]),
            expected_foundation_epochs=int(plan["foundation_epochs"]),
            expected_checkpoint_path=checkpoint_path,
            expected_training_contract=plan["training_config_contract"],
        )
        runtime_payload = json.loads(Path(str(artifacts["runtime"])).read_text(encoding="utf-8"))
        if not isinstance(runtime_payload, Mapping) or int(
            runtime_payload.get("images", -1)
        ) != int(fold_plan["val_images"]):
            raise ValueError(f"fold {fold} runtime image count 不一致")
        _validate_runtime_provenance(
            Path(str(artifacts["runtime"])),
            runtime_payload,
            expected_config_path=Path(str(artifacts["infer_config"])),
            expected_checkpoint_path=checkpoint_path,
            expected_predictions_path=prediction_path,
        )
        records = [dict(record) for record in load_coco_prediction_records(prediction_path)]
        validate_coco_prediction_records(
            records,
            allowed_category_ids=EXPECTED_CATEGORY_IDS,
        )
        expected_ids = {sample.image_id for sample in samples if sample.fold == fold}
        predicted_ids = {int(record["image_id"]) for record in records}
        if not predicted_ids <= expected_ids:
            raise ValueError(f"fold {fold} prediction 含非本折图像")
        if seen_oof_images & expected_ids:
            raise ValueError("同一图像被两个 OOF fold 声明")
        seen_oof_images.update(expected_ids)
        counts: dict[int, int] = {image_id: 0 for image_id in expected_ids}
        checkpoint_sha = _sha256(
            artifacts.get("checkpoint_sha256"), f"fold {fold} checkpoint_sha256"
        )
        checkpoint_shas.append(checkpoint_sha)
        for source_index, record in enumerate(records):
            image_id = int(record["image_id"])
            counts[image_id] += 1
            if float(record["score"]) + 1e-12 < float(plan["low_score_threshold"]):
                raise ValueError(f"fold {fold} prediction score 低于冻结候选阈值")
            if counts[image_id] > int(plan["max_detections"]):
                raise ValueError(
                    f"fold {fold} image_id={image_id} proposal_count 超出 "
                    f"max_detections={plan['max_detections']}"
                )
            if image_sizes is not None:
                if image_id not in image_sizes:
                    raise ValueError(f"formal crop 缺少 prediction image_id={image_id} 的尺寸")
                _validate_coco_bbox_within_image(
                    record,
                    image_size=image_sizes[image_id],
                    context=(f"fold {fold} predictions[{source_index}] image_id={image_id}"),
                )
            proposal_uid = f"{plan['model_key'].lower()}-f{fold}-i{image_id}-p{source_index:06d}"
            x, y, width, height = [float(value) for value in record["bbox"]]
            proposal_rows.append(
                {
                    "proposal_uid": proposal_uid,
                    "image_id": image_id,
                    "fold": fold,
                    "category_id": int(record["category_id"]),
                    "x": f"{x:.10g}",
                    "y": f"{y:.10g}",
                    "width": f"{width:.10g}",
                    "height": f"{height:.10g}",
                    "score": f"{float(record['score']):.10g}",
                    "model_key": plan["model_key"],
                    "checkpoint_sha256": checkpoint_sha,
                    "source_prediction_index": source_index,
                }
            )
        for image_id in sorted(expected_ids):
            sample = by_image[image_id]
            image_rows.append(
                {
                    "image_id": image_id,
                    "relative_path": sample.relative_path,
                    "fold": fold,
                    "group_id": sample.group_id,
                    "model_key": plan["model_key"],
                    "checkpoint_sha256": checkpoint_sha,
                    "prediction_count": counts[image_id],
                }
            )
        all_predictions.extend(records)
        fold_summaries.append(
            {
                "fold": fold,
                "images": len(expected_ids),
                "images_with_predictions": len(predicted_ids),
                "proposals": len(records),
                "checkpoint_sha256": checkpoint_sha,
                "predictions_sha256": artifacts["predictions_sha256"],
                "metadata_sha256": sha256_file(metadata_path),
            }
        )

    if seen_oof_images != all_image_ids:
        missing = sorted(all_image_ids - seen_oof_images)
        raise ValueError(f"OOF 图像覆盖不完整，缺少 {missing[:10]}")
    if len(image_rows) != len(samples):
        raise ValueError("每张图必须恰好出现一次 OOF image row")
    if (
        len(checkpoint_shas) != EXPECTED_FOLD_COUNT
        or len(set(checkpoint_shas)) != EXPECTED_FOLD_COUNT
    ):
        raise ValueError("三折必须产生三个不同的独立 last checkpoint")
    if image_sizes is not None and set(image_sizes) != all_image_ids:
        missing = sorted(all_image_ids - set(image_sizes))
        extra = sorted(set(image_sizes) - all_image_ids)
        raise ValueError(
            "formal crop image_id 集合与正式 CV3 不一致: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"OOF 汇总目录非空，禁止覆盖: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    predictions_path = destination / "predictions_oof_low.json"
    atomic_write_json(predictions_path, all_predictions)
    images_path = destination / "oof_images.csv"
    proposals_path = destination / "oof_proposals.csv"
    _atomic_write(
        images_path,
        _csv_bytes(
            (
                "image_id",
                "relative_path",
                "fold",
                "group_id",
                "model_key",
                "checkpoint_sha256",
                "prediction_count",
            ),
            image_rows,
        ),
    )
    _atomic_write(
        proposals_path,
        _csv_bytes(
            (
                "proposal_uid",
                "image_id",
                "fold",
                "category_id",
                "x",
                "y",
                "width",
                "height",
                "score",
                "model_key",
                "checkpoint_sha256",
                "source_prediction_index",
            ),
            proposal_rows,
        ),
    )

    metadata = {
        "contract_version": OOF_CONTRACT_VERSION,
        "status": (
            "complete_downstream_ready"
            if crop_manifest is not None
            else "diagnostic_only_no_formal_crop"
        ),
        "downstream_admission": crop_manifest is not None,
        "model_key": plan["model_key"],
        "model_family": plan["model_family"],
        "model_name": plan["model_name"],
        "run_plan_sha256": plan_sha,
        "seed": int(plan["seed"]),
        "source_manifest_version": source_document["version"],
        "source_manifest_sha256": source_sha,
        "fold_count": EXPECTED_FOLD_COUNT,
        "image_count": len(image_rows),
        "proposal_count": len(proposal_rows),
        "low_score_threshold": float(plan["low_score_threshold"]),
        "checkpoint_selection": "fixed_epoch_last",
        "initial_pretrained_weight_sha256": plan["initial_pretrained_weight_sha256"],
        "fold_checkpoint_sha256": checkpoint_shas,
        "folds": fold_summaries,
        "formal_crop_manifest": crop_manifest,
        "detection_data_lock": dict(plan_data_lock),
        "artifacts": {
            "oof_images": {
                "path": str(images_path),
                "sha256": sha256_file(images_path),
            },
            "oof_proposals": {
                "path": str(proposals_path),
                "sha256": sha256_file(proposals_path),
            },
            "predictions_oof_low": {
                "path": str(predictions_path),
                "sha256": sha256_file(predictions_path),
            },
        },
        "audit": {
            "each_image_exactly_once": True,
            "prediction_ids_within_held_out_fold": True,
            "group_cross_fold_count": 0,
            "resume_used": False,
            "same_pretrained_initialization_all_folds": True,
            "distinct_last_checkpoint_all_folds": True,
            "formal_crop_exact_sha_bound": crop_manifest is not None,
            "detection_data_bytes_locked_each_fold": True,
            "prediction_boxes_within_source_images": image_sizes is not None,
            "per_image_max_detections_enforced": True,
        },
    }
    metadata_path = destination / "oof_metadata.json"
    atomic_write_json(metadata_path, metadata)
    return metadata


__all__ = [
    "CV3Sample",
    "EXPECTED_CATEGORY_IDS",
    "EXPECTED_FOLD_COUNT",
    "EXPECTED_IMAGE_COUNT",
    "FORMAL_CROP_MANIFEST_SHA256",
    "FORMAL_CROP_MANIFEST_VERSION",
    "FORMAL_DETECTION_DATA_LOCK_SHA256",
    "FORMAL_DETECTION_DATA_LOCK_VERSION",
    "OOF_CONTRACT_VERSION",
    "atomic_write_json",
    "audit_and_aggregate_oof",
    "build_fold_view_document",
    "finalize_fold_delivery",
    "load_cv3_manifest",
    "prepare_oof_run_plan",
    "sha256_file",
]
