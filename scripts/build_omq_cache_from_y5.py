#!/usr/bin/env python3
"""Build deployable OMQ metadata/FPN caches from the frozen Y5 CV3 ledger.

The label order is exactly deployment order: coarse workpoint threshold,
class-aware NMS, then the repository's prediction-first official matcher.
Same-fine IoU is computed independently of closer wrong-class objects.  All
features are prediction-time evidence; GT is used only for labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou, evaluate_predictions_with_trace
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.postprocess.nms import class_aware_nms_predictions
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_assignments(formal) -> dict[int, tuple[int, str]]:
    result: dict[int, tuple[int, str]] = {}
    for obj in formal.objects.values():
        assignment = (int(obj.fold), str(obj.group_id))
        previous = result.setdefault(int(obj.image_id), assignment)
        if previous != assignment:
            raise ValueError(f"inconsistent assignment for image {obj.image_id}")
    return result


def _oto_support(predictions: list[dict], oto_dir: Path) -> np.ndarray:
    by_image: dict[int, list[dict]] = defaultdict(list)
    for fold in range(3):
        path = oto_dir / f"a5_oto_fold{fold}.json"
        for raw in json.loads(path.read_text(encoding="utf-8")):
            by_image[int(raw["image_id"])].append(
                {
                    "category_id": int(raw["category_id"]),
                    "score": float(raw["score"]),
                    "bbox_xyxy": [float(value) for value in raw["bbox_xyxy"]],
                }
            )
    output = np.zeros(len(predictions), dtype=np.float32)
    for index, pred in enumerate(predictions):
        for other in by_image.get(int(pred["image_id"]), ()):
            if other["category_id"] != int(pred["category_id"]) or other["score"] < 0.5:
                continue
            if compute_iou(pred["bbox_xyxy"], other["bbox_xyxy"]) > 0.5:
                output[index] = 1.0
                break
    return output


def _best_same_fine_iou(predictions: list[dict], gt: dict[int, list[dict]]) -> np.ndarray:
    by_image_class: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    for image_id, rows in gt.items():
        for row in rows:
            by_image_class[(int(image_id), int(row["category_id"]))].append(
                [float(value) for value in row["bbox_xyxy"]]
            )
    output = np.zeros(len(predictions), dtype=np.float32)
    for index, pred in enumerate(predictions):
        boxes = by_image_class.get(
            (int(pred["image_id"]), int(pred["category_id"])), ()
        )
        output[index] = max(
            (compute_iou(pred["bbox_xyxy"], box) for box in boxes), default=0.0
        )
    return output


def _metadata_features(
    nodes: pd.DataFrame,
    predictions: list[dict],
    has_oto: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    n = len(nodes)
    category = nodes["category_id"].to_numpy(dtype=np.int64)
    crop_category = nodes["crop_top1_class"].to_numpy(dtype=np.int64)
    coarse = np.where(category <= 3, 0, np.where(category <= 23, 1, 2)).astype(np.int64)
    numerical_names = [
        "detector_score",
        "crop_top1_probability",
        "crop_margin",
        "crop_entropy_normalized",
        "detector_crop_agree",
        "log_width",
        "log_height",
        "log_area",
        "log_aspect",
        "log_local_density",
        "d4_support_normalized",
        "has_oto",
    ]
    numerical = np.column_stack(
        (
            nodes["y5_score"].to_numpy(dtype=np.float32),
            nodes["crop_top1"].to_numpy(dtype=np.float32),
            nodes["crop_margin"].to_numpy(dtype=np.float32),
            nodes["crop_entropy"].to_numpy(dtype=np.float32) / math.log(25.0),
            nodes["detector_crop_agree"].to_numpy(dtype=np.float32),
            np.log1p(nodes["w"].to_numpy(dtype=np.float32)),
            np.log1p(nodes["h"].to_numpy(dtype=np.float32)),
            np.log1p(nodes["area"].to_numpy(dtype=np.float32)),
            np.log(np.maximum(nodes["aspect"].to_numpy(dtype=np.float32), 1.0)),
            np.log1p(nodes["local_density"].to_numpy(dtype=np.float32)),
            np.asarray(
                [float(row.get("d4_support", 0.0)) / 8.0 for row in predictions],
                dtype=np.float32,
            ),
            has_oto,
        )
    ).astype(np.float32)
    coarse_one_hot = np.eye(3, dtype=np.float32)[coarse]
    detector_one_hot = np.eye(25, dtype=np.float32)[category]
    crop_one_hot = np.eye(25, dtype=np.float32)[crop_category]
    names = (
        numerical_names
        + [f"coarse_{index}" for index in range(3)]
        + [f"detector_class_{index}" for index in range(25)]
        + [f"crop_class_{index}" for index in range(25)]
    )
    output = np.concatenate(
        (numerical, coarse_one_hot, detector_one_hot, crop_one_hot), axis=1
    )
    if output.shape != (n, len(names)) or not np.isfinite(output).all():
        raise RuntimeError("metadata feature construction failed")
    return output, names


def _fpn_features(
    nodes: pd.DataFrame, fpn_dir: Path
) -> tuple[np.ndarray, list[str]]:
    by_uid: dict[str, np.ndarray] = {}
    width = None
    for fold in range(3):
        path = fpn_dir / f"y5_fpn_feat_fold{fold}.npz"
        with np.load(path, allow_pickle=True) as payload:
            uids = payload["uids"]
            feats = payload["feats"]
            width = int(feats.shape[1]) if width is None else width
            if feats.ndim != 2 or int(feats.shape[1]) != width:
                raise ValueError("FPN cache dimensions are inconsistent")
            for uid, feature in zip(uids, feats, strict=True):
                key = str(uid)
                if key in by_uid:
                    raise ValueError(f"duplicate FPN proposal_uid: {key}")
                by_uid[key] = np.asarray(feature, dtype=np.float16)
    missing = [str(uid) for uid in nodes["proposal_uid"] if str(uid) not in by_uid]
    if missing:
        raise ValueError(f"FPN cache lacks {len(missing)} proposal UIDs")
    matrix = np.stack([by_uid[str(uid)] for uid in nodes["proposal_uid"]], axis=0)
    if not np.isfinite(matrix).all():
        raise ValueError("FPN cache contains NaN/Inf")
    return matrix, [f"fpn_{index}" for index in range(int(matrix.shape[1]))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--oto-dir", type=Path, required=True)
    parser.add_argument("--fpn-dir", type=Path)
    parser.add_argument("--formal-crop-manifest", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--feature-set", choices=("metadata", "metadata_fpn"), required=True)
    parser.add_argument("--threshold-ship", type=float, default=0.150)
    parser.add_argument("--threshold-aircraft", type=float, default=0.301)
    parser.add_argument("--threshold-vehicle", type=float, default=0.366)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.feature_set == "metadata_fpn" and args.fpn_dir is None:
        raise ValueError("metadata_fpn requires --fpn-dir")
    thresholds = np.asarray(
        [args.threshold_ship, args.threshold_aircraft, args.threshold_vehicle],
        dtype=np.float32,
    )
    if not np.isfinite(thresholds).all() or np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("workpoint thresholds must be within [0, 1]")
    if not 0.0 < args.nms_iou <= 1.0:
        raise ValueError("nms-iou must be within (0, 1]")

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    assignment = _image_assignments(formal)
    nodes = pd.read_csv(args.nodes).sort_values("idx").reset_index(drop=True)
    if nodes["idx"].astype(int).tolist() != list(range(len(nodes))):
        raise ValueError("node idx must be contiguous and prediction-aligned")
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    if len(predictions) != len(nodes):
        raise ValueError("prediction/node count mismatch")
    for index, (raw, uid) in enumerate(
        zip(predictions, nodes["proposal_uid"], strict=True)
    ):
        raw["image_id"] = int(raw["image_id"])
        raw["category_id"] = int(raw["category_id"])
        raw["score"] = float(raw["score"])
        raw["bbox_xyxy"] = [float(value) for value in raw["bbox_xyxy"]]
        raw["source_prediction_index"] = index
        if str(raw.get("proposal_uid")) != str(uid):
            raise ValueError(f"prediction/node UID mismatch at row {index}")

    category = nodes["category_id"].to_numpy(dtype=np.int64)
    coarse = np.where(category <= 3, 0, np.where(category <= 23, 1, 2)).astype(np.int64)
    active_before_nms = np.asarray(
        [predictions[index]["score"] >= thresholds[coarse[index]] for index in range(len(nodes))]
    )
    by_image: dict[int, list[dict]] = {int(image_id): [] for image_id in formal.boxes}
    for index in np.flatnonzero(active_before_nms):
        row = predictions[int(index)]
        by_image[int(row["image_id"])].append(row)
    kept = class_aware_nms_predictions(by_image, args.nms_iou)
    active_ids = {
        int(row["source_prediction_index"])
        for rows in kept.values()
        for row in rows
    }
    metrics, trace = evaluate_predictions_with_trace(
        formal.boxes,
        kept,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    protected_ids = {int(item.prediction_index) for item in trace.matches}
    active_fp_ids = {int(item.prediction_index) for item in trace.unmatched_predictions}
    if protected_ids & active_fp_ids or active_ids != protected_ids | active_fp_ids:
        raise RuntimeError("official active label partition is inconsistent")

    has_oto = _oto_support(predictions, args.oto_dir)
    best_iou = _best_same_fine_iou(predictions, formal.boxes)
    metadata, feature_names = _metadata_features(nodes, predictions, has_oto)
    features = metadata.astype(np.float16)
    if args.feature_set == "metadata_fpn":
        fpn, fpn_names = _fpn_features(nodes, args.fpn_dir)
        features = np.concatenate((features, fpn), axis=1).astype(np.float16)
        feature_names += fpn_names

    folds = np.asarray([assignment[int(value)][0] for value in nodes["image_id"]], dtype=np.int64)
    raw_groups = [assignment[int(value)][1] for value in nodes["image_id"]]
    group_names = sorted(set(raw_groups))
    group_map = {name: index for index, name in enumerate(group_names)}
    group_ids = np.asarray([group_map[name] for name in raw_groups], dtype=np.int64)
    active_mask = np.asarray([int(index in active_ids) for index in range(len(nodes))], dtype=np.uint8)
    protected = np.asarray([int(index in protected_ids) for index in range(len(nodes))], dtype=np.uint8)
    active_fp = np.asarray([int(index in active_fp_ids) for index in range(len(nodes))], dtype=np.uint8)
    bbox = np.asarray([row["bbox_xyxy"] for row in predictions], dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        detector_score=nodes["y5_score"].to_numpy(dtype=np.float32),
        best_same_fine_iou=best_iou,
        coarse_id=coarse,
        protected_tp=protected,
        active_fp=active_fp,
        active_mask=active_mask,
        group_id=group_ids,
        fold=folds,
        candidate_index=np.arange(len(nodes), dtype=np.int64),
        image_id=nodes["image_id"].to_numpy(dtype=np.int64),
        category_id=category,
        bbox_xyxy=bbox,
    )
    audit = {
        "status": "complete",
        "feature_set": args.feature_set,
        "rows": len(nodes),
        "feature_dim": int(features.shape[1]),
        "feature_names": feature_names,
        "fold_counts": {str(fold): int((folds == fold).sum()) for fold in range(3)},
        "group_count": len(group_names),
        "group_map": group_map,
        "workpoint_thresholds": {
            "ship": float(thresholds[0]),
            "aircraft": float(thresholds[1]),
            "vehicle": float(thresholds[2]),
        },
        "active_before_nms": int(active_before_nms.sum()),
        "active_after_nms": len(active_ids),
        "protected_tp": len(protected_ids),
        "active_fp": len(active_fp_ids),
        "official_pooled_recall": float(metrics.recall),
        "official_pooled_fdr": float(metrics.fdr),
        "input_sha256": {
            "nodes": _sha256(args.nodes),
            "predictions": _sha256(args.predictions),
            "formal_crop_manifest": _sha256(args.formal_crop_manifest),
        },
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
    }
    audit_path = args.output.with_suffix(args.output.suffix + ".audit.json")
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
