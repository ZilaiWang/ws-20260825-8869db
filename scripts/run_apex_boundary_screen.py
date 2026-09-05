#!/usr/bin/env python3
"""Train and evaluate APEX A0/A1 on frozen P40 proposals.

A0 is a real-proposal-only frozen-embedding control.  A1 adds the plan-16 LMP
jitter negatives and visual prototype margins.  Calibration is source-group
disjoint from fitting.  Both arms are rescue-only: boxes, detector scores and
fine labels remain immutable and only low-score Ship/Vehicle tails may enter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.augmentation.apex_boundary import select_precision_threshold
from rsdet.data.crop_classification import render_crop
from rsdet.evaluation.coco import load_coco_ground_truth
from rsdet.evaluation.official_metric import evaluate_predictions, evaluate_ranking_metrics
from rsdet.evaluation.platform_protocol import (
    build_platform_observed_metrics,
    platform_metrics_payload,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.experiments.fixed_proxy import quality_contribution
from rsdet.features.p04_inputs import D4_VIEW_IDS, apply_d4_view
from rsdet.features.p04_teachers import build_teacher
from rsdet.models.apex_boundary import ApexBoundaryClassifier
from rsdet.models.crop_classifier import build_convnext_tiny_classifier, sha256_file
from rsdet.postprocess.nms import class_aware_nms_predictions
from rsdet.utils.config import load_config

ARM_POLICIES = {
    "a0": {
        "prototypes": False,
        "jitter": False,
        "scale": False,
        "background": False,
        "head": "logistic",
    },
    "a0m": {
        "prototypes": False,
        "jitter": False,
        "scale": False,
        "background": False,
        "head": "mlp_rank",
    },
    "a1": {
        "prototypes": True,
        "jitter": True,
        "scale": False,
        "background": False,
        "head": "logistic",
    },
    "a1m": {
        "prototypes": True,
        "jitter": True,
        "scale": False,
        "background": False,
        "head": "mlp_rank",
    },
    "a2": {
        "prototypes": False,
        "jitter": False,
        "scale": True,
        "background": False,
        "head": "logistic",
    },
    "a3m": {
        "prototypes": False,
        "jitter": False,
        "scale": False,
        "background": True,
        "head": "mlp_rank",
    },
    "a4": {
        "prototypes": True,
        "jitter": True,
        "scale": True,
        "background": False,
        "head": "logistic",
    },
}
TARGET_CATEGORIES = frozenset({0, 1, 2, 3, 24})
REAL_PROPOSAL_ROLES = frozenset({"canonical_tp", "fp_duplicate", "fp_cls", "fp_bg"})
COARSE_FINE_IDS = {"ship": frozenset({0, 1, 2, 3}), "vehicle": frozenset({24})}
MINIMUM_RESCUE_PRECISION = {"ship": 0.95, "vehicle": 0.90}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _stable_bucket(value: str, modulus: int = 5) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % modulus


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("empty or duplicate-row manifest")
    if set(int(row["fold"]) for row in rows) != {0, 1, 2}:
        raise ValueError("manifest does not cover CV3")
    return rows


def _normalise(image: Image.Image) -> Any:
    from torchvision.transforms import functional

    tensor = functional.to_tensor(image)
    return functional.normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def _load_backbone(checkpoint: Path, imagenet: Path, fold: int) -> tuple[Any, dict[str, Any]]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    resolved = payload.get("resolved_config", {})
    if int(resolved.get("fold", -1)) != fold or resolved.get("policy") != "tight":
        raise ValueError(f"fold{fold} P03 checkpoint contract mismatch")
    model = build_convnext_tiny_classifier(
        25, weight_path=imagenet, regime="fine_tune", verify_weight_sha256=True
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, {
        "checkpoint_sha256": sha256_file(checkpoint),
        "epoch": int(payload.get("epoch", -1)),
        "resolved_config": resolved,
    }


def _forward_features(model: Any, tensors: Any) -> Any:
    import torch

    values = model.features(tensors)
    values = model.avgpool(values)
    values = model.classifier[0](values)
    return torch.flatten(values, 1)


def _render_training_crop(image: Image.Image, row: Mapping[str, Any]) -> Image.Image:
    box = [float(value) for value in row["bbox_xyxy"]]
    if row.get("outside_policy") != "reflect":
        return render_crop(image, box, 224)
    left = max(0, math.ceil(-box[0]))
    top = max(0, math.ceil(-box[1]))
    right = max(0, math.ceil(box[2] - image.width))
    bottom = max(0, math.ceil(box[3] - image.height))
    if left or top or right or bottom:
        array = np.asarray(image)
        array = np.pad(array, ((top, bottom), (left, right), (0, 0)), mode="reflect")
        image = Image.fromarray(array)
        box = [box[0] + left, box[1] + top, box[2] + left, box[3] + top]
    return render_crop(image, box, 224)


class _ImageCache:
    def __init__(self, root: Path, capacity: int = 8) -> None:
        self.root = root.resolve()
        self.capacity = capacity
        self.values: OrderedDict[str, Image.Image] = OrderedDict()

    def get(self, relative_path: str) -> Image.Image:
        if relative_path in self.values:
            image = self.values.pop(relative_path)
            self.values[relative_path] = image
            return image
        supplied = Path(relative_path)
        path = supplied.resolve() if supplied.is_absolute() else (self.root / supplied).resolve()
        if not supplied.is_absolute():
            path.relative_to(self.root)
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.load()
        self.values[relative_path] = image
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)
        return image


def _extract_manifest_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    data_root: Path,
    checkpoint: Path,
    imagenet: Path,
    fold: int,
    batch_size: int,
    view_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    model, provenance = _load_backbone(checkpoint, imagenet, fold)
    device = torch.device("cuda")
    model.to(device).eval()
    cache = _ImageCache(data_root)
    outputs: list[np.ndarray] = []
    view_ids = D4_VIEW_IDS[:1] if view_mode == "identity" else D4_VIEW_IDS
    started = time.perf_counter()
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        tensors = []
        for row in batch:
            crop = _render_training_crop(cache.get(str(row["relative_path"])), row)
            tensors.extend(_normalise(apply_d4_view(crop, view)) for view in view_ids)
        tensors = torch.stack(tensors).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            features = _forward_features(model, tensors)
        features = features.reshape(len(batch), len(view_ids), -1).mean(1)
        outputs.append(features.float().cpu().numpy())
        if offset and offset % (batch_size * 50) == 0:
            print(json.dumps({"fold_backbone": fold, "features": offset}), flush=True)
    values = np.concatenate(outputs).astype(np.float32)
    if values.shape[0] != len(rows) or not np.isfinite(values).all():
        raise RuntimeError("feature cache is incomplete or non-finite")
    provenance["rows"] = len(rows)
    provenance["view_mode"] = view_mode
    provenance["views"] = list(view_ids)
    provenance["elapsed_seconds"] = time.perf_counter() - started
    return values, provenance


def _dino_teacher(args: argparse.Namespace) -> Any:
    missing = [
        name
        for name in ("dinov2_repo", "dinov2_weights", "dinov2_weight_sha256")
        if not getattr(args, name)
    ]
    if missing:
        raise ValueError(f"DINOv2 feature backbone missing arguments: {missing}")
    return build_teacher(
        "dinov2_vitb14",
        device="cuda",
        compute_dtype="float16",
        options={
            "repo": args.dinov2_repo,
            "weights": args.dinov2_weights,
            "weight_sha256": args.dinov2_weight_sha256,
            "include_patch_mean": True,
        },
    )


def _extract_manifest_dino_features(
    rows: Sequence[Mapping[str, Any]],
    *,
    data_root: Path,
    teacher: Any,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache = _ImageCache(data_root)
    outputs: list[np.ndarray] = []
    started = time.perf_counter()
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        images = [
            _render_training_crop(cache.get(str(row["relative_path"])), row) for row in batch
        ]
        values = teacher.extract(
            images, sample_keys=[str(row["row_id"]) for row in batch]
        )["dino_cls_patchmean"]
        outputs.append(np.asarray(values, dtype=np.float32))
        if offset and offset % (batch_size * 50) == 0:
            print(json.dumps({"dinov2_features": offset}), flush=True)
    matrix = np.concatenate(outputs).astype(np.float32)
    if matrix.shape[0] != len(rows) or not np.isfinite(matrix).all():
        raise RuntimeError("DINOv2 feature cache is incomplete or non-finite")
    return matrix, {
        "rows": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "teacher": teacher.metadata(),
        "feature_name": "dino_cls_patchmean",
    }


def _thresholds(frontier_path: Path) -> dict[int, float]:
    values = _read_json(frontier_path)["frontiers"]["0.150"]["crossfit_thresholds"]
    output = {int(key): float(value) for key, value in values.items()}
    if set(output) != {0, 1, 2}:
        raise ValueError("P40 crossfit threshold coverage mismatch")
    return output


def _real_tail(
    row: Mapping[str, Any], thresholds: Mapping[int, float], floors: Mapping[str, float]
) -> bool:
    return (
        str(row["role"]) in REAL_PROPOSAL_ROLES
        and float(row["score"]) >= floors[str(row["coarse"])]
        and float(row["score"]) < thresholds[int(row["fold"])]
    )


def _fit_arms(
    rows: list[dict[str, Any]],
    features_by_fold: Mapping[int, np.ndarray],
    *,
    thresholds: Mapping[int, float],
    floors: Mapping[str, float],
    output: Path,
    arms: Sequence[str],
    target_coarse: Sequence[str],
    calibration_mode: str,
    calibration_fine_ids: frozenset[int] | None = None,
) -> dict[str, Any]:
    calibration: dict[str, Any] = {}
    for outer_fold in range(3):
        features = features_by_fold[outer_fold]
        training_universe = [
            index for index, row in enumerate(rows) if int(row["fold"]) != outer_fold
        ]
        for arm in arms:
            policy = ARM_POLICIES[arm]
            for coarse in target_coarse:
                eligible_indices = [
                    index
                    for index in training_universe
                    if str(rows[index]["coarse"]) == coarse
                    and (policy["jitter"] or str(rows[index]["role"]) != "jitter_hard_negative")
                    and (policy["scale"] or str(rows[index]["role"]) != "object_scale_positive")
                    and (
                        policy["background"]
                        or str(rows[index]["role"]) != "retrieved_background_negative"
                    )
                ]
                if set(int(rows[index]["target"]) for index in eligible_indices) != {0, 1}:
                    raise RuntimeError(
                        f"insufficient {outer_fold}/{arm}/{coarse} eligible fit data"
                    )
                minimum_precision = 0.95 if coarse == "ship" else 0.90
                minimum_true_positives = 10 if coarse == "ship" else 5
                calibration_indices: list[int] = []
                probability_parts: list[np.ndarray] = []
                if calibration_mode == "single_split":
                    calibration_groups = {
                        str(rows[index]["source_group"])
                        for index in training_universe
                        if _stable_bucket(str(rows[index]["source_group"])) == 0
                    }
                    fit_indices = [
                        index
                        for index in eligible_indices
                        if str(rows[index]["source_group"]) not in calibration_groups
                    ]
                    calibration_indices = [
                        index
                        for index in training_universe
                        if str(rows[index]["coarse"]) == coarse
                        and str(rows[index]["source_group"]) in calibration_groups
                        and (
                            calibration_fine_ids is None
                            or int(rows[index]["category_id"]) in calibration_fine_ids
                        )
                        and _real_tail(rows[index], thresholds, floors)
                    ]
                    model = ApexBoundaryClassifier(
                        use_prototypes=bool(policy["prototypes"]), head=str(policy["head"])
                    ).fit(
                        features[fit_indices],
                        [rows[index] for index in fit_indices],
                        [int(rows[index]["target"]) for index in fit_indices],
                        [float(rows[index]["sample_weight"]) for index in fit_indices],
                    )
                    probability_parts.append(
                        model.predict_proba(
                            features[calibration_indices],
                            [rows[index] for index in calibration_indices],
                        )
                    )
                else:
                    calibration_groups = {
                        str(rows[index]["source_group"]) for index in training_universe
                    }
                    for inner_bucket in range(5):
                        heldout_groups = {
                            group
                            for group in calibration_groups
                            if _stable_bucket(group) == inner_bucket
                        }
                        inner_fit = [
                            index
                            for index in eligible_indices
                            if str(rows[index]["source_group"]) not in heldout_groups
                        ]
                        inner_calibration = [
                            index
                            for index in training_universe
                            if str(rows[index]["coarse"]) == coarse
                            and str(rows[index]["source_group"]) in heldout_groups
                            and (
                                calibration_fine_ids is None
                                or int(rows[index]["category_id"]) in calibration_fine_ids
                            )
                            and _real_tail(rows[index], thresholds, floors)
                        ]
                        if not inner_calibration:
                            continue
                        inner_model = ApexBoundaryClassifier(
                            use_prototypes=bool(policy["prototypes"]),
                            head=str(policy["head"]),
                            random_state=42 + inner_bucket,
                        ).fit(
                            features[inner_fit],
                            [rows[index] for index in inner_fit],
                            [int(rows[index]["target"]) for index in inner_fit],
                            [float(rows[index]["sample_weight"]) for index in inner_fit],
                        )
                        calibration_indices.extend(inner_calibration)
                        probability_parts.append(
                            inner_model.predict_proba(
                                features[inner_calibration],
                                [rows[index] for index in inner_calibration],
                            )
                        )
                    fit_indices = eligible_indices
                    model = ApexBoundaryClassifier(
                        use_prototypes=bool(policy["prototypes"]), head=str(policy["head"])
                    ).fit(
                        features[fit_indices],
                        [rows[index] for index in fit_indices],
                        [int(rows[index]["target"]) for index in fit_indices],
                        [float(rows[index]["sample_weight"]) for index in fit_indices],
                    )
                if not calibration_indices or not probability_parts:
                    raise RuntimeError(f"empty {calibration_mode} calibration for {outer_fold}")
                probabilities = np.concatenate(probability_parts)
                selected = select_precision_threshold(
                    probabilities,
                    [int(rows[index]["target"]) for index in calibration_indices],
                    minimum_precision=minimum_precision,
                    minimum_true_positives=(
                        2 if calibration_mode == "single_split" else minimum_true_positives
                    ),
                )
                positive_source_groups = 0
                if selected is not None:
                    positive_source_groups = len(
                        {
                            str(rows[index]["source_group"])
                            for index, probability in zip(
                                calibration_indices, probabilities, strict=True
                            )
                            if float(probability) >= selected.threshold
                            and int(rows[index]["target"]) == 1
                        }
                    )
                    if calibration_mode == "inner_oof" and positive_source_groups < 3:
                        selected = None
                quality_threshold = 1.1 if selected is None else selected.threshold
                destination = output / "models" / f"fold_{outer_fold}" / arm
                destination.mkdir(parents=True, exist_ok=True)
                joblib.dump(
                    {
                        "model": model,
                        "quality_threshold": quality_threshold,
                        "coarse": coarse,
                        "arm": arm,
                        "fold": outer_fold,
                        "calibration": None if selected is None else asdict(selected),
                    },
                    destination / f"{coarse}.joblib",
                )
                key = f"fold{outer_fold}:{arm}:{coarse}"
                calibration[key] = {
                    "fit_rows": len(fit_indices),
                    "fit_groups": len({rows[index]["source_group"] for index in fit_indices}),
                    "calibration_rows": len(calibration_indices),
                    "calibration_groups": len(calibration_groups),
                    "calibration_mode": calibration_mode,
                    "minimum_true_positives": (
                        2 if calibration_mode == "single_split" else minimum_true_positives
                    ),
                    "positive_source_groups": positive_source_groups,
                    "minimum_precision": minimum_precision,
                    "calibration_fine_ids": (
                        sorted(calibration_fine_ids) if calibration_fine_ids else None
                    ),
                    "selection": None if selected is None else asdict(selected),
                    "fail_closed": selected is None,
                }
                print(json.dumps({key: calibration[key]}), flush=True)
    return calibration


def _load_images_csv(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        int(row["image_id"]): {
            "relative_path": row["relative_path"],
            "fold": int(row["fold"]),
            "source_group": row["group_id"],
        }
        for row in rows
    }


def _yolo_ground_truth(
    images: Mapping[int, Mapping[str, Any]], data_root: Path
) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = {}
    for image_id, metadata in images.items():
        relative = str(metadata["relative_path"])
        image_path = data_root / relative
        with Image.open(image_path) as image:
            width, height = image.size
        label_path = data_root / Path(relative.replace("images/", "labels/", 1)).with_suffix(".txt")
        records = []
        for text in label_path.read_text().splitlines() if label_path.is_file() else []:
            category, cx, cy, box_width, box_height = text.split()
            cx, cy, box_width, box_height = map(float, (cx, cy, box_width, box_height))
            records.append(
                {
                    "category_id": int(category),
                    "bbox_xyxy": [
                        (cx - box_width / 2) * width,
                        (cy - box_height / 2) * height,
                        (cx + box_width / 2) * width,
                        (cy + box_height / 2) * height,
                    ],
                }
            )
        output[image_id] = records
    return output


def _raw_internal(raw: Sequence[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source_index, row in enumerate(raw):
        x, y, width, height = map(float, row["bbox"])
        output[int(row["image_id"])].append(
            {
                "bbox_xyxy": [x, y, x + width, y + height],
                "category_id": int(row["category_id"]),
                "score": float(row["score"]),
                "source_prediction_index": source_index,
            }
        )
    return output


def _metrics(
    gt: dict[int, list[dict[str, Any]]], pred: dict[int, list[dict[str, Any]]]
) -> dict[str, Any]:
    protocol = parse_evaluation_protocol(load_config("configs/project.yaml"))
    kwargs = {
        "class_names": protocol.class_names,
        "category_mapping": protocol.category_mapping,
        "iou_thresholds": protocol.iou_thresholds,
    }
    pooled = evaluate_predictions(gt, pred, **kwargs)
    ranking = evaluate_ranking_metrics(gt, pred, require_complete_taxonomy=True, **kwargs)
    platform = platform_metrics_payload(build_platform_observed_metrics(ranking))
    return {
        "platform": platform,
        "quality_contribution": quality_contribution(platform),
        "pooled": asdict(pooled),
        "per_fine": {str(key): asdict(value) for key, value in ranking.per_fine.items()},
    }


def _incremental_rescue_ledger(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    target_coarse: Sequence[str],
) -> dict[str, Any]:
    """Summarise effective TP/FP count changes after rescue and final NMS.

    The detector boxes are immutable, but class-aware NMS can replace a control
    prediction with a rescued proposal.  Consequently this is an *effective*
    post-NMS ledger derived from official-matcher counts, not a raw count of
    proposals admitted by the classifier.  A negative FP delta is retained as
    an explicit removal and never used to inflate the incremental precision.
    """

    per_coarse: dict[str, Any] = {}
    for coarse in target_coarse:
        fine_ids = COARSE_FINE_IDS[coarse]
        per_fine: dict[str, Any] = {}
        added_tp = removed_tp = added_fp = removed_fp = 0
        for fine_id in sorted(fine_ids):
            control_row = control["per_fine"][str(fine_id)]
            candidate_row = candidate["per_fine"][str(fine_id)]
            delta_tp = int(candidate_row["tp"]) - int(control_row["tp"])
            delta_fp = int(candidate_row["fp"]) - int(control_row["fp"])
            per_fine[str(fine_id)] = {"delta_tp": delta_tp, "delta_fp": delta_fp}
            added_tp += max(delta_tp, 0)
            removed_tp += max(-delta_tp, 0)
            added_fp += max(delta_fp, 0)
            removed_fp += max(-delta_fp, 0)

        denominator = added_tp + added_fp
        precision = added_tp / denominator if denominator else None
        minimum = MINIMUM_RESCUE_PRECISION[coarse]
        per_coarse[coarse] = {
            "per_fine": per_fine,
            "effective_added_tp": added_tp,
            "effective_removed_tp": removed_tp,
            "effective_added_fp": added_fp,
            "effective_removed_fp": removed_fp,
            "net_delta_tp": added_tp - removed_tp,
            "net_delta_fp": added_fp - removed_fp,
            "effective_incremental_precision": precision,
            "minimum_precision": minimum,
            "precision_gate_pass": bool(
                added_tp > 0
                and removed_tp == 0
                and precision is not None
                and precision >= minimum
            ),
        }
    return {"schema_version": "effective_rescue_ledger_v1", "per_coarse": per_coarse}


def _attach_rescue_evidence(
    result: dict[str, Any], control: Mapping[str, Any], target_coarse: Sequence[str]
) -> None:
    result["delta_quality_vs_nms_control"] = (
        result["quality_contribution"] - control["quality_contribution"]
    )
    result["incremental_rescue_ledger"] = _incremental_rescue_ledger(
        control, result, target_coarse
    )


def _apply_normal_cached(
    raw: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    features_by_fold: Mapping[int, np.ndarray],
    *,
    images: Mapping[int, Mapping[str, Any]],
    thresholds: Mapping[int, float],
    floors: Mapping[str, float],
    model_root: Path,
    arm: str,
    target_coarse: Sequence[str],
    target_fine_ids: frozenset[int] | None,
) -> tuple[
    dict[int, list[dict[str, Any]]],
    dict[int, list[dict[str, Any]]],
    dict[str, Any],
]:
    internal = _raw_internal(raw)
    baseline = {
        image_id: [
            row
            for row in internal.get(image_id, [])
            if row["score"] >= thresholds[int(meta["fold"])]
        ]
        for image_id, meta in images.items()
    }
    nms_control = class_aware_nms_predictions(baseline, 0.5, category_ids=sorted(TARGET_CATEGORIES))
    candidate = {image_id: [dict(row) for row in values] for image_id, values in baseline.items()}
    audit: Counter[str] = Counter()
    for fold in range(3):
        for coarse in target_coarse:
            indices = [
                index
                for index, row in enumerate(rows)
                if int(row["fold"]) == fold
                and str(row["coarse"]) == coarse
                and (
                    target_fine_ids is None
                    or int(row["category_id"]) in target_fine_ids
                )
                and _real_tail(row, thresholds, floors)
            ]
            if not indices:
                continue
            bundle = joblib.load(model_root / f"fold_{fold}" / arm / f"{coarse}.joblib")
            probabilities = bundle["model"].predict_proba(
                features_by_fold[fold][indices], [rows[index] for index in indices]
            )
            for index, probability in zip(indices, probabilities, strict=True):
                audit[f"{coarse}:examined"] += 1
                if float(probability) < float(bundle["quality_threshold"]):
                    continue
                source_index = int(str(rows[index]["row_id"]).rsplit(":", 1)[1])
                source = raw[source_index]
                if int(source["image_id"]) != int(rows[index]["image_id"]):
                    raise AssertionError("manifest/raw proposal identity mismatch")
                x, y, width, height = map(float, source["bbox"])
                candidate[int(source["image_id"])].append(
                    {
                        "bbox_xyxy": [x, y, x + width, y + height],
                        "category_id": int(source["category_id"]),
                        "score": float(source["score"]),
                        "source_prediction_index": source_index,
                    }
                )
                audit[f"{coarse}:rescued"] += 1
    candidate = class_aware_nms_predictions(candidate, 0.5, category_ids=sorted(TARGET_CATEGORIES))
    return candidate, nms_control, dict(sorted(audit.items()))


def _apply_models(
    raw: Sequence[Mapping[str, Any]],
    *,
    images: Mapping[int, Mapping[str, Any]],
    image_paths: Mapping[int, Path],
    thresholds: Mapping[int, float],
    floors: Mapping[str, float],
    model_root: Path,
    checkpoints: Mapping[int, Path],
    imagenet: Path,
    arm: str,
    batch_size: int,
    view_mode: str,
    target_coarse: Sequence[str],
    target_fine_ids: frozenset[int] | None,
) -> tuple[
    dict[int, list[dict[str, Any]]],
    dict[int, list[dict[str, Any]]],
    dict[str, Any],
]:
    import torch

    internal = _raw_internal(raw)
    baseline = {
        image_id: [
            row
            for row in internal.get(image_id, [])
            if row["score"] >= thresholds[int(meta["fold"])]
        ]
        for image_id, meta in images.items()
    }
    nms_control = class_aware_nms_predictions(baseline, 0.5, category_ids=sorted(TARGET_CATEGORIES))
    candidate = {image_id: [dict(row) for row in rows] for image_id, rows in baseline.items()}
    audit: Counter[str] = Counter()
    view_ids = D4_VIEW_IDS[:1] if view_mode == "identity" else D4_VIEW_IDS
    for fold in range(3):
        backbone, _ = _load_backbone(checkpoints[fold], imagenet, fold)
        backbone.cuda().eval()
        bundles = {
            coarse: joblib.load(model_root / f"fold_{fold}" / arm / f"{coarse}.joblib")
            for coarse in target_coarse
        }
        for image_id, metadata in images.items():
            if int(metadata["fold"]) != fold:
                continue
            rows: list[dict[str, Any]] = []
            for row in internal.get(image_id, []):
                category_id = int(row["category_id"])
                if category_id not in TARGET_CATEGORIES:
                    continue
                if target_fine_ids is not None and category_id not in target_fine_ids:
                    continue
                coarse = "ship" if category_id < 4 else "vehicle"
                if floors[coarse] <= float(row["score"]) < thresholds[fold]:
                    box = row["bbox_xyxy"]
                    side = math.sqrt(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
                    rows.append(
                        {
                            **row,
                            "relative_path": str(image_paths[image_id]),
                            "fold": fold,
                            "source_group": str(metadata.get("source_group", f"proxy:{image_id}")),
                            "coarse": coarse,
                            "scale_bin": "tiny"
                            if side < 32
                            else "small"
                            if side < 64
                            else "medium"
                            if side < 128
                            else "large",
                        }
                    )
            if not rows:
                continue
            with Image.open(image_paths[image_id]) as source:
                image = source.convert("RGB")
                image.load()
            features: list[np.ndarray] = []
            for offset in range(0, len(rows), batch_size):
                batch = rows[offset : offset + batch_size]
                tensors = []
                for row in batch:
                    crop = render_crop(image, row["bbox_xyxy"], 224)
                    tensors.extend(_normalise(apply_d4_view(crop, view)) for view in view_ids)
                tensors = torch.stack(tensors).cuda()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    values = _forward_features(backbone, tensors)
                values = values.reshape(len(batch), len(view_ids), -1).mean(1)
                features.append(values.float().cpu().numpy())
            matrix = np.concatenate(features)
            for coarse in target_coarse:
                indices = [index for index, row in enumerate(rows) if row["coarse"] == coarse]
                if not indices:
                    continue
                bundle = bundles[coarse]
                probabilities = bundle["model"].predict_proba(
                    matrix[indices], [rows[index] for index in indices]
                )
                for index, probability in zip(indices, probabilities, strict=True):
                    audit[f"{coarse}:examined"] += 1
                    if float(probability) >= float(bundle["quality_threshold"]):
                        candidate[image_id].append(
                            {
                                key: rows[index][key]
                                for key in (
                                    "bbox_xyxy",
                                    "category_id",
                                    "score",
                                    "source_prediction_index",
                                )
                            }
                        )
                        audit[f"{coarse}:rescued"] += 1
        del backbone
        torch.cuda.empty_cache()
    candidate = class_aware_nms_predictions(candidate, 0.5, category_ids=sorted(TARGET_CATEGORIES))
    return candidate, nms_control, dict(sorted(audit.items()))


def _apply_dino_models(
    raw: Sequence[Mapping[str, Any]],
    *,
    images: Mapping[int, Mapping[str, Any]],
    image_paths: Mapping[int, Path],
    thresholds: Mapping[int, float],
    floors: Mapping[str, float],
    model_root: Path,
    teacher: Any,
    arm: str,
    batch_size: int,
    target_coarse: Sequence[str],
    target_fine_ids: frozenset[int] | None,
) -> tuple[
    dict[int, list[dict[str, Any]]],
    dict[int, list[dict[str, Any]]],
    dict[str, Any],
]:
    internal = _raw_internal(raw)
    baseline = {
        image_id: [
            row
            for row in internal.get(image_id, [])
            if row["score"] >= thresholds[int(meta["fold"])]
        ]
        for image_id, meta in images.items()
    }
    nms_control = class_aware_nms_predictions(
        baseline, 0.5, category_ids=sorted(TARGET_CATEGORIES)
    )
    candidate = {image_id: [dict(row) for row in values] for image_id, values in baseline.items()}
    audit: Counter[str] = Counter()
    for fold in range(3):
        bundles = {
            coarse: joblib.load(model_root / f"fold_{fold}" / arm / f"{coarse}.joblib")
            for coarse in target_coarse
        }
        for image_id, metadata in images.items():
            if int(metadata["fold"]) != fold:
                continue
            rows: list[dict[str, Any]] = []
            for row in internal.get(image_id, []):
                category_id = int(row["category_id"])
                if category_id not in TARGET_CATEGORIES:
                    continue
                if target_fine_ids is not None and category_id not in target_fine_ids:
                    continue
                coarse = "ship" if category_id < 4 else "vehicle"
                if coarse not in target_coarse:
                    continue
                if floors[coarse] <= float(row["score"]) < thresholds[fold]:
                    box = row["bbox_xyxy"]
                    side = math.sqrt(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
                    rows.append(
                        {
                            **row,
                            "fold": fold,
                            "source_group": str(
                                metadata.get("source_group", f"proxy:{image_id}")
                            ),
                            "coarse": coarse,
                            "scale_bin": "tiny"
                            if side < 32
                            else "small"
                            if side < 64
                            else "medium"
                            if side < 128
                            else "large",
                        }
                    )
            if not rows:
                continue
            with Image.open(image_paths[image_id]) as source:
                source_image = source.convert("RGB")
                source_image.load()
            chunks: list[np.ndarray] = []
            for offset in range(0, len(rows), batch_size):
                batch = rows[offset : offset + batch_size]
                crops = [render_crop(source_image, row["bbox_xyxy"], 224) for row in batch]
                values = teacher.extract(
                    crops,
                    sample_keys=[
                        f"{image_id}:{row['source_prediction_index']}" for row in batch
                    ],
                )["dino_cls_patchmean"]
                chunks.append(np.asarray(values, dtype=np.float32))
            matrix = np.concatenate(chunks)
            for coarse in target_coarse:
                indices = [index for index, row in enumerate(rows) if row["coarse"] == coarse]
                if not indices:
                    continue
                bundle = bundles[coarse]
                probabilities = bundle["model"].predict_proba(
                    matrix[indices], [rows[index] for index in indices]
                )
                for index, probability in zip(indices, probabilities, strict=True):
                    audit[f"{coarse}:examined"] += 1
                    if float(probability) >= float(bundle["quality_threshold"]):
                        candidate[image_id].append(
                            {
                                key: rows[index][key]
                                for key in (
                                    "bbox_xyxy",
                                    "category_id",
                                    "score",
                                    "source_prediction_index",
                                )
                            }
                        )
                        audit[f"{coarse}:rescued"] += 1
    candidate = class_aware_nms_predictions(
        candidate, 0.5, category_ids=sorted(TARGET_CATEGORIES)
    )
    return candidate, nms_control, dict(sorted(audit.items()))


def _run_train(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    rows = _load_manifest(args.manifest)
    thresholds = _thresholds(args.frontier)
    floors = {"ship": args.ship_floor, "vehicle": args.vehicle_floor}
    checkpoints = {
        fold: args.p03_root / f"ft-tight-224-fold{fold}" / "final_checkpoint.pt"
        for fold in range(3)
    }
    features: dict[int, np.ndarray] = {}
    provenance: dict[str, Any] = {}
    if args.feature_backbone == "dinov2b":
        if args.view_mode != "identity":
            raise ValueError("DINOv2-B first screen is frozen to identity view")
        teacher = _dino_teacher(args)
        values, meta = _extract_manifest_dino_features(
            rows, data_root=args.data_root, teacher=teacher, batch_size=args.batch_size
        )
        np.save(args.output / "features_dinov2b.npy", values)
        features = {fold: values for fold in range(3)}
        provenance["shared_dinov2b"] = meta
    else:
        for fold in range(3):
            values, meta = _extract_manifest_features(
                rows,
                data_root=args.data_root,
                checkpoint=checkpoints[fold],
                imagenet=args.imagenet,
                fold=fold,
                batch_size=args.batch_size,
                view_mode=args.view_mode,
            )
            features[fold] = values
            np.save(args.output / f"features_fold{fold}.npy", values)
            provenance[str(fold)] = meta
    target_fine_ids = frozenset(args.target_fine_ids) if args.target_fine_ids else None
    calibration = _fit_arms(
        rows,
        features,
        thresholds=thresholds,
        floors=floors,
        output=args.output,
        arms=args.arms,
        target_coarse=args.target_coarse,
        calibration_mode=args.calibration_mode,
        calibration_fine_ids=target_fine_ids,
    )
    images = _load_images_csv(args.images_csv)
    raw = _read_json(args.predictions)
    gt = _yolo_ground_truth(images, args.data_root)
    internal = _raw_internal(raw)
    baseline = {
        image_id: [
            row for row in internal.get(image_id, []) if row["score"] >= thresholds[meta["fold"]]
        ]
        for image_id, meta in images.items()
    }
    comparisons: dict[str, Any] = {"baseline": _metrics(gt, baseline)}
    for arm in args.arms:
        candidate, nms_control, audit = _apply_normal_cached(
            raw,
            rows,
            features,
            images=images,
            thresholds=thresholds,
            floors=floors,
            model_root=args.output / "models",
            arm=arm,
            target_coarse=args.target_coarse,
            target_fine_ids=target_fine_ids,
        )
        control_metrics = _metrics(gt, nms_control)
        if "nms_control" not in comparisons:
            comparisons["nms_control"] = control_metrics
        elif comparisons["nms_control"] != control_metrics:
            raise AssertionError("A0/A1 NMS controls differ")
        result = _metrics(gt, candidate)
        result["rescue_audit"] = audit
        _attach_rescue_evidence(result, comparisons["nms_control"], args.target_coarse)
        comparisons[arm] = result
    summary = {
        "status": "complete",
        "experiment_id": "HERA-GUARD-APEX-A0-A1-P40-CV3-V1",
        "arms": {arm: ARM_POLICIES[arm] for arm in args.arms},
        "thresholds": thresholds,
        "floors": floors,
        "calibration": calibration,
        "backbone_provenance": provenance,
        "view_mode": args.view_mode,
        "feature_backbone": args.feature_backbone,
        "target_coarse": args.target_coarse,
        "target_fine_ids": sorted(target_fine_ids) if target_fine_ids else None,
        "calibration_mode": args.calibration_mode,
        "normal_oof": comparisons,
        "rescue_only_invariants": {
            "boxes": "immutable",
            "scores": "immutable",
            "fine_labels": "immutable",
        },
    }
    (args.output / "train_normal_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "status.txt").write_text("train_normal_complete\n")
    print(
        json.dumps(
            {arm: comparisons[arm].get("delta_quality_vs_nms_control") for arm in args.arms},
            indent=2,
        )
    )


def _run_refit(args: argparse.Namespace) -> None:
    """Refit model heads from an audited immutable feature cache."""

    args.output.mkdir(parents=True, exist_ok=False)
    rows = _load_manifest(args.manifest)
    thresholds = _thresholds(args.frontier)
    floors = {"ship": args.ship_floor, "vehicle": args.vehicle_floor}
    features = {
        fold: np.load(args.source_feature_root / f"features_fold{fold}.npy", mmap_mode="r")
        for fold in range(3)
    }
    if any(values.shape[0] != len(rows) for values in features.values()):
        raise ValueError("feature cache and manifest differ")
    target_fine_ids = frozenset(args.target_fine_ids) if args.target_fine_ids else None
    calibration = _fit_arms(
        rows,
        features,
        thresholds=thresholds,
        floors=floors,
        output=args.output,
        arms=args.arms,
        target_coarse=args.target_coarse,
        calibration_mode=args.calibration_mode,
        calibration_fine_ids=target_fine_ids,
    )
    images = _load_images_csv(args.images_csv)
    raw = _read_json(args.predictions)
    gt = _yolo_ground_truth(images, args.data_root)
    internal = _raw_internal(raw)
    baseline = {
        image_id: [
            row for row in internal.get(image_id, []) if row["score"] >= thresholds[meta["fold"]]
        ]
        for image_id, meta in images.items()
    }
    comparisons: dict[str, Any] = {"baseline": _metrics(gt, baseline)}
    for arm in args.arms:
        candidate, nms_control, audit = _apply_normal_cached(
            raw,
            rows,
            features,
            images=images,
            thresholds=thresholds,
            floors=floors,
            model_root=args.output / "models",
            arm=arm,
            target_coarse=args.target_coarse,
            target_fine_ids=target_fine_ids,
        )
        control_metrics = _metrics(gt, nms_control)
        comparisons.setdefault("nms_control", control_metrics)
        if comparisons["nms_control"] != control_metrics:
            raise AssertionError("A0/A1 NMS controls differ")
        result = _metrics(gt, candidate)
        result["rescue_audit"] = audit
        _attach_rescue_evidence(result, comparisons["nms_control"], args.target_coarse)
        comparisons[arm] = result
    summary = {
        "status": "complete",
        "experiment_id": "HERA-GUARD-APEX-A0-A1-P40-CV3-SOURCE-SAFE-V2",
        "source_feature_root": str(args.source_feature_root),
        "source_feature_sha256": {
            str(fold): sha256_file(args.source_feature_root / f"features_fold{fold}.npy")
            for fold in range(3)
        },
        "prototype_policy": "positive/jitter/active_fp_role_specific_and_exclude_current_source",
        "thresholds": thresholds,
        "floors": floors,
        "calibration": calibration,
        "normal_oof": comparisons,
        "target_coarse": args.target_coarse,
        "target_fine_ids": sorted(target_fine_ids) if target_fine_ids else None,
        "calibration_mode": args.calibration_mode,
    }
    (args.output / "train_normal_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "status.txt").write_text("train_normal_complete\n")
    print(
        json.dumps(
            {arm: comparisons[arm]["delta_quality_vs_nms_control"] for arm in args.arms},
            indent=2,
        )
    )


def _run_proxy(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=False)
    thresholds = _thresholds(args.frontier)
    floors = {"ship": args.ship_floor, "vehicle": args.vehicle_floor}
    checkpoints = {
        fold: args.p03_root / f"ft-tight-224-fold{fold}" / "final_checkpoint.pt"
        for fold in range(3)
    }
    document = _read_json(args.proxy_root / "ground_truth.json")
    images = {
        int(row["id"]): {
            "fold": int(row["fold"]),
            "source_group": f"{args.condition}:{row['id']}",
        }
        for row in document["images"]
    }
    image_paths = {
        int(row["id"]): args.proxy_root / f"fold_{int(row['fold'])}" / "images" / row["file_name"]
        for row in document["images"]
    }
    raw = _read_json(args.proxy_predictions)
    internal = _raw_internal(raw)
    baseline = {
        image_id: [
            row for row in internal.get(image_id, []) if row["score"] >= thresholds[meta["fold"]]
        ]
        for image_id, meta in images.items()
    }
    gt = load_coco_ground_truth(args.proxy_root / "ground_truth.json")
    comparison: dict[str, Any] = {"baseline": _metrics(gt, baseline)}
    target_fine_ids = frozenset(args.target_fine_ids) if args.target_fine_ids else None
    teacher = _dino_teacher(args) if args.feature_backbone == "dinov2b" else None
    for arm in args.arms:
        if teacher is None:
            candidate, nms_control, audit = _apply_models(
                raw,
                images=images,
                image_paths=image_paths,
                thresholds=thresholds,
                floors=floors,
                model_root=args.model_root,
                checkpoints=checkpoints,
                imagenet=args.imagenet,
                arm=arm,
                batch_size=args.batch_size,
                view_mode=args.view_mode,
                target_coarse=args.target_coarse,
                target_fine_ids=target_fine_ids,
            )
        else:
            candidate, nms_control, audit = _apply_dino_models(
                raw,
                images=images,
                image_paths=image_paths,
                thresholds=thresholds,
                floors=floors,
                model_root=args.model_root,
                teacher=teacher,
                arm=arm,
                batch_size=args.batch_size,
                target_coarse=args.target_coarse,
                target_fine_ids=target_fine_ids,
            )
        control_metrics = _metrics(gt, nms_control)
        if "nms_control" not in comparison:
            comparison["nms_control"] = control_metrics
        elif comparison["nms_control"] != control_metrics:
            raise AssertionError("A0/A1 NMS controls differ")
        result = _metrics(gt, candidate)
        result["rescue_audit"] = audit
        _attach_rescue_evidence(result, comparison["nms_control"], args.target_coarse)
        comparison[arm] = result
    summary = {
        "status": "complete",
        "condition": args.condition,
        "thresholds": thresholds,
        "floors": floors,
        "view_mode": args.view_mode,
        "feature_backbone": args.feature_backbone,
        "target_coarse": args.target_coarse,
        "target_fine_ids": sorted(target_fine_ids) if target_fine_ids else None,
        "comparison": comparison,
    }
    (args.output / "comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (args.output / "status.txt").write_text("complete\n")
    print(
        json.dumps(
            {arm: comparison[arm]["delta_quality_vs_nms_control"] for arm in args.arms},
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--frontier", type=Path, required=True)
    shared.add_argument("--p03-root", type=Path, required=True)
    shared.add_argument("--imagenet", type=Path, required=True)
    shared.add_argument("--ship-floor", type=float, default=0.003)
    shared.add_argument("--vehicle-floor", type=float, default=0.001)
    shared.add_argument("--batch-size", type=int, default=64)
    shared.add_argument("--view-mode", choices=("identity", "d4"), default="identity")
    shared.add_argument(
        "--feature-backbone", choices=("p03", "dinov2b"), default="p03"
    )
    shared.add_argument("--dinov2-repo", type=Path)
    shared.add_argument("--dinov2-weights", type=Path)
    shared.add_argument("--dinov2-weight-sha256")
    shared.add_argument(
        "--target-coarse",
        nargs="+",
        choices=("ship", "vehicle"),
        default=["ship", "vehicle"],
        help="Coarse modules allowed to rescue; fitted models remain independently auditable.",
    )
    shared.add_argument(
        "--target-fine-ids",
        nargs="+",
        type=int,
        help=(
            "Optional frozen action whitelist. Models still fit the full selected coarse "
            "classes, but rescue is allowed only for these fine category IDs."
        ),
    )
    shared.add_argument(
        "--calibration-mode",
        choices=("single_split", "inner_oof"),
        default="single_split",
    )
    shared.add_argument("--arms", nargs="+", choices=tuple(ARM_POLICIES), default=["a0", "a1"])
    train = subparsers.add_parser("train-normal", parents=[shared])
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--images-csv", type=Path, required=True)
    train.add_argument("--predictions", type=Path, required=True)
    train.add_argument("--data-root", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    refit = subparsers.add_parser("refit-normal", parents=[shared])
    refit.add_argument("--manifest", type=Path, required=True)
    refit.add_argument("--images-csv", type=Path, required=True)
    refit.add_argument("--predictions", type=Path, required=True)
    refit.add_argument("--data-root", type=Path, required=True)
    refit.add_argument("--source-feature-root", type=Path, required=True)
    refit.add_argument("--output", type=Path, required=True)
    proxy = subparsers.add_parser("evaluate-proxy", parents=[shared])
    proxy.add_argument("--condition", choices=("hard", "sentinel"), required=True)
    proxy.add_argument("--proxy-root", type=Path, required=True)
    proxy.add_argument("--proxy-predictions", type=Path, required=True)
    proxy.add_argument("--model-root", type=Path, required=True)
    proxy.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    Image.MAX_IMAGE_PIXELS = None
    if args.command == "train-normal":
        _run_train(args)
    elif args.command == "refit-normal":
        _run_refit(args)
    else:
        _run_proxy(args)


if __name__ == "__main__":
    main()
