"""Formal M1/M3 OOF metric, error-attribution and complementarity analysis.

The module deliberately reuses :mod:`rsdet.evaluation.official_metric` for
every TP/FP/FN decision.  Error labels are a deterministic diagnostic layer
applied *after* official matching; they are not a second official metric.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from rsdet.evaluation.coco import load_coco_predictions
from rsdet.evaluation.official_metric import (
    OfficialEvaluationTrace,
    OverallMetrics,
    compute_iou,
    evaluate_predictions,
    evaluate_predictions_with_trace,
)
from rsdet.evaluation.protocol import EvaluationProtocol, parse_evaluation_protocol
from rsdet.experiments.cv3_oof import (
    FORMAL_CROP_MANIFEST_SHA256,
    FORMAL_CROP_MANIFEST_VERSION,
    OOF_CONTRACT_VERSION,
    sha256_file,
)
from rsdet.postprocess.calibration import build_threshold_grid, filter_predictions_by_score
from rsdet.predictions import load_coco_prediction_records
from rsdet.utils.config import load_config

ANALYSIS_VERSION = "m1_m3_cv3_oof_analysis_v1"
EXPECTED_MODEL_KEYS = ("M1", "M3")
ERROR_HIERARCHY = (
    "FP_DUP",
    "FP_CLS",
    "FP_LOC",
    "FP_BG",
)
FN_HIERARCHY = (
    "FN_CLS",
    "FN_LOC",
    "FN_MISS",
)


@dataclass(frozen=True)
class GroundTruthObject:
    """One unique GT object recovered from the frozen tight crop row."""

    annotation_uid: str
    image_id: int
    ground_truth_index: int
    fold: int
    group_id: str
    category_id: int
    class_name: str
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class FormalGroundTruth:
    """Official-evaluator input plus stable object identities."""

    boxes: dict[int, list[dict[str, Any]]]
    objects: dict[tuple[int, int], GroundTruthObject]
    image_ids: frozenset[int]
    annotation_count: int


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n").encode("utf-8"),
    )


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"empty CSV requires explicit fieldnames: {path}")
    names = list(fieldnames or rows[0].keys())
    with tempfile.SpooledTemporaryFile(
        mode="w+",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=names,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.seek(0)
        _atomic_write(path, handle.read().encode("utf-8"))


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} 顶层必须是映射")
    return dict(payload)


def load_analysis_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the frozen analysis protocol."""

    config_path = Path(path).expanduser().resolve()
    payload = _load_mapping(config_path, "analysis config")
    analysis = payload.get("analysis")
    if not isinstance(analysis, Mapping):
        raise ValueError("analysis config 缺少 analysis 映射")
    if analysis.get("version") != ANALYSIS_VERSION:
        raise ValueError(f"analysis.version 必须是 {ANALYSIS_VERSION}")
    if analysis.get("formal_crop_manifest_sha256") != FORMAL_CROP_MANIFEST_SHA256:
        raise ValueError("analysis 未绑定唯一冻结 formal crop SHA")
    if analysis.get("formal_crop_policy") != "tight":
        raise ValueError("正式检测 GT 必须来自 formal crop 的 tight 行")
    if int(analysis.get("expected_images", -1)) != 4481:
        raise ValueError("analysis.expected_images 必须为 4481")
    if int(analysis.get("expected_annotations", -1)) != 20933:
        raise ValueError("analysis.expected_annotations 必须为 20933")
    if float(analysis.get("candidate_floor", math.nan)) != 0.001:
        raise ValueError("analysis.candidate_floor 必须为冻结低阈值 0.001")
    grid = analysis.get("threshold_grid")
    if not isinstance(grid, Mapping):
        raise ValueError("analysis.threshold_grid 必须是映射")
    thresholds = build_threshold_grid(
        float(grid["start"]),
        float(grid["stop"]),
        float(grid["step"]),
    )
    stop = float(grid["stop"])
    if not math.isclose(thresholds[-1], stop, abs_tol=1e-12):
        thresholds.append(stop)
    if not math.isclose(thresholds[0], 0.001, abs_tol=1e-12):
        raise ValueError("threshold grid 必须从 candidate_floor=0.001 开始")
    if analysis.get("error_attribution_hierarchy") != [
        "official_match",
        *ERROR_HIERARCHY,
    ]:
        raise ValueError("error attribution hierarchy 与冻结协议不一致")
    if analysis.get("threshold_selection_scope") != "same_oof_exploratory_only":
        raise ValueError("阈值选择必须明确标记为 same-OOF exploratory")
    result = dict(analysis)
    result["thresholds"] = thresholds
    result["config_path"] = str(config_path)
    result["config_sha256"] = sha256_file(config_path)
    return result


def load_formal_ground_truth(
    path: str | Path,
    *,
    expected_sha256: str = FORMAL_CROP_MANIFEST_SHA256,
    expected_images: int = 4481,
    expected_annotations: int = 20933,
) -> FormalGroundTruth:
    """Recover one unique official GT row per object from ``crop_policy=tight``."""

    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"formal crop manifest 不存在: {manifest}")
    actual_sha = sha256_file(manifest)
    if actual_sha != expected_sha256:
        raise ValueError(f"formal crop SHA 不匹配: expected={expected_sha256}, actual={actual_sha}")
    boxes: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    objects: dict[tuple[int, int], GroundTruthObject] = {}
    seen_annotations: set[str] = set()
    image_ids: set[int] = set()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "manifest_version",
            "annotation_uid",
            "formal_image_id",
            "fold",
            "group_id",
            "class_id",
            "major_class",
            "gt_x0",
            "gt_y0",
            "gt_x1",
            "gt_y1",
            "crop_policy",
            "coordinate_semantics",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"formal crop 缺少 GT 字段: {sorted(missing)}")
        for line_number, row in enumerate(reader, 2):
            if row["manifest_version"] != FORMAL_CROP_MANIFEST_VERSION:
                raise ValueError(f"formal crop 第 {line_number} 行 manifest_version 非法")
            image_ids.add(int(row["formal_image_id"]))
            if row["crop_policy"] != "tight":
                continue
            if row["coordinate_semantics"] != "continuous_float_xyxy_half_open":
                raise ValueError(f"formal crop 第 {line_number} 行坐标语义不是冻结 half-open xyxy")
            annotation_uid = row["annotation_uid"].strip()
            if not annotation_uid or annotation_uid in seen_annotations:
                raise ValueError(f"tight annotation_uid 为空或重复: {annotation_uid!r}")
            seen_annotations.add(annotation_uid)
            image_id = int(row["formal_image_id"])
            category_id = int(row["class_id"])
            bbox = tuple(float(row[key]) for key in ("gt_x0", "gt_y0", "gt_x1", "gt_y1"))
            if (
                category_id not in range(25)
                or len(bbox) != 4
                or not all(math.isfinite(value) for value in bbox)
                or bbox[2] <= bbox[0]
                or bbox[3] <= bbox[1]
            ):
                raise ValueError(f"formal crop 第 {line_number} 行 GT 非法")
            gt_index = len(boxes[image_id])
            boxes[image_id].append(
                {
                    "bbox_xyxy": list(bbox),
                    "category_id": category_id,
                }
            )
            objects[(image_id, gt_index)] = GroundTruthObject(
                annotation_uid=annotation_uid,
                image_id=image_id,
                ground_truth_index=gt_index,
                fold=int(row["fold"]),
                group_id=row["group_id"].strip(),
                category_id=category_id,
                class_name=row["major_class"].strip(),
                bbox_xyxy=bbox,
            )
    if len(image_ids) != expected_images:
        raise ValueError(f"formal GT 图像数错误: {len(image_ids)} != {expected_images}")
    if len(objects) != expected_annotations:
        raise ValueError(f"formal GT 对象数错误: {len(objects)} != {expected_annotations}")
    return FormalGroundTruth(
        boxes=dict(boxes),
        objects=objects,
        image_ids=frozenset(image_ids),
        annotation_count=len(objects),
    )


def _artifact_record(
    metadata: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get(name), Mapping):
        raise ValueError(f"OOF metadata 缺少 artifacts.{name}")
    return artifacts[name]


def _artifact_sha(metadata: Mapping[str, Any], name: str) -> str:
    artifact = _artifact_record(metadata, name)
    value = str(artifact.get("sha256", "")).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"OOF metadata artifacts.{name}.sha256 非法")
    return value


def _audit_recorded_artifact_path(
    metadata: Mapping[str, Any],
    name: str,
    actual_path: Path,
) -> bool:
    """Validate artifact path identity when the recorded server path exists.

    Returned packages may be audited after relocation, so a missing historical
    absolute path is allowed only when its basename and content SHA still
    match.  During the formal server task the recorded path exists and must be
    the same file.
    """

    artifact = _artifact_record(metadata, name)
    recorded_text = str(artifact.get("path", "")).strip()
    if not recorded_text:
        raise ValueError(f"OOF metadata artifacts.{name}.path 不能为空")
    recorded = Path(recorded_text).expanduser()
    if recorded.name != actual_path.name:
        raise ValueError(f"OOF metadata artifacts.{name}.path 文件名不一致")
    if recorded.is_file():
        if recorded.resolve() != actual_path.resolve():
            raise ValueError(f"OOF metadata artifacts.{name}.path 实体不一致")
        return False
    return recorded.resolve() != actual_path.resolve()


def load_oof_aggregate(
    aggregate_dir: str | Path,
    *,
    expected_model_key: str,
    expected_manifest_sha256: str,
    expected_formal_crop_sha256: str,
    expected_images: int,
    candidate_floor: float,
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    """Audit one completed downstream-ready OOF aggregate and load predictions."""

    root = Path(aggregate_dir).expanduser().resolve()
    metadata_path = root / "oof_metadata.json"
    predictions_path = root / "predictions_oof_low.json"
    images_path = root / "oof_images.csv"
    proposals_path = root / "oof_proposals.csv"
    for path in (metadata_path, predictions_path, images_path, proposals_path):
        if not path.is_file():
            raise FileNotFoundError(f"OOF aggregate 缺少 {path.name}: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, Mapping):
        raise ValueError("oof_metadata.json 顶层必须是对象")
    checks = {
        "contract_version": OOF_CONTRACT_VERSION,
        "status": "complete_downstream_ready",
        "downstream_admission": True,
        "model_key": expected_model_key,
        "source_manifest_sha256": expected_manifest_sha256,
        "image_count": expected_images,
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"{expected_model_key} OOF metadata.{key} "
                f"应为 {expected!r}，实际 {metadata.get(key)!r}"
            )
    if not math.isclose(
        float(metadata.get("low_score_threshold", math.nan)),
        candidate_floor,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{expected_model_key} OOF 低阈值不是 candidate_floor")
    formal = metadata.get("formal_crop_manifest")
    if not isinstance(formal, Mapping) or formal.get("sha256") != (expected_formal_crop_sha256):
        raise ValueError(f"{expected_model_key} OOF 未绑定冻结 formal crop")
    fold_checkpoint_shas = metadata.get("fold_checkpoint_sha256")
    if not isinstance(fold_checkpoint_shas, list) or len(fold_checkpoint_shas) != 3:
        raise ValueError(f"{expected_model_key} OOF fold checkpoint 列表非法")
    normalized_fold_checkpoint_shas = [str(value).strip().lower() for value in fold_checkpoint_shas]
    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in normalized_fold_checkpoint_shas
    ):
        raise ValueError(f"{expected_model_key} OOF fold checkpoint SHA 非法")
    relocated_artifacts: list[str] = []
    for name, path in (
        ("oof_images", images_path),
        ("oof_proposals", proposals_path),
        ("predictions_oof_low", predictions_path),
    ):
        if _audit_recorded_artifact_path(metadata, name, path):
            relocated_artifacts.append(name)
        if sha256_file(path) != _artifact_sha(metadata, name):
            raise ValueError(f"{expected_model_key} OOF {name} SHA 闭环失败")

    seen_image_ids: set[int] = set()
    image_ledger: dict[int, dict[str, Any]] = {}
    with images_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        image_required = {
            "image_id",
            "fold",
            "model_key",
            "checkpoint_sha256",
            "prediction_count",
        }
        missing = image_required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{expected_model_key} oof_images 缺字段: {sorted(missing)}")
        rows = list(reader)
    if len(rows) != expected_images:
        raise ValueError(f"{expected_model_key} oof_images 不是 {expected_images} 行")
    for row in rows:
        image_id = int(row["image_id"])
        if image_id in seen_image_ids:
            raise ValueError(f"{expected_model_key} oof_images image_id 重复")
        seen_image_ids.add(image_id)
        if row["model_key"] != expected_model_key:
            raise ValueError(f"{expected_model_key} oof_images model_key 混入")
        prediction_count = int(row["prediction_count"])
        fold = int(row["fold"])
        checkpoint_sha = row["checkpoint_sha256"].strip().lower()
        if prediction_count < 0 or fold not in (0, 1, 2):
            raise ValueError(f"{expected_model_key} oof_images count/fold 非法")
        if len(checkpoint_sha) != 64 or any(
            character not in "0123456789abcdef" for character in checkpoint_sha
        ):
            raise ValueError(f"{expected_model_key} oof_images checkpoint SHA 非法")
        if checkpoint_sha != normalized_fold_checkpoint_shas[fold]:
            raise ValueError(f"{expected_model_key} oof_images checkpoint/fold 不一致")
        image_ledger[image_id] = {
            "prediction_count": prediction_count,
            "fold": fold,
            "checkpoint_sha256": checkpoint_sha,
        }

    raw_prediction_records = load_coco_prediction_records(predictions_path)
    pred_boxes = load_coco_predictions(predictions_path)
    pred_image_ids = set(pred_boxes)
    if not pred_image_ids <= seen_image_ids:
        raise ValueError(f"{expected_model_key} predictions 含未知 image_id")
    for image_id, records in pred_boxes.items():
        for index, record in enumerate(records):
            score = float(record["score"])
            if score + 1e-12 < candidate_floor:
                raise ValueError(
                    f"{expected_model_key} image={image_id} pred={index} 低于冻结 candidate floor"
                )
    prediction_counts = {image_id: len(pred_boxes.get(image_id, ())) for image_id in seen_image_ids}
    ledger_counts = {
        image_id: int(item["prediction_count"]) for image_id, item in image_ledger.items()
    }
    if prediction_counts != ledger_counts:
        mismatches = [
            (image_id, ledger_counts[image_id], prediction_counts[image_id])
            for image_id in sorted(seen_image_ids)
            if ledger_counts[image_id] != prediction_counts[image_id]
        ]
        raise ValueError(
            f"{expected_model_key} oof_images prediction_count 与 JSON 不一致: {mismatches[:10]}"
        )

    flat_predictions = [(int(record["image_id"]), record) for record in raw_prediction_records]
    with proposals_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        proposal_required = {
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
        }
        missing = proposal_required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{expected_model_key} oof_proposals 缺字段: {sorted(missing)}")
        proposal_rows = list(reader)
    metadata_proposal_count = int(metadata.get("proposal_count", -1))
    if not (len(proposal_rows) == len(flat_predictions) == metadata_proposal_count):
        raise ValueError(
            f"{expected_model_key} proposal 行数不闭环: "
            f"csv={len(proposal_rows)}, json={len(flat_predictions)}, "
            f"metadata={metadata_proposal_count}"
        )
    proposal_uids: set[str] = set()
    fold_source_indices = Counter()
    for row_number, (row, (json_image_id, prediction)) in enumerate(
        zip(proposal_rows, flat_predictions, strict=True),
        2,
    ):
        proposal_uid = row["proposal_uid"].strip()
        if not proposal_uid or proposal_uid in proposal_uids:
            raise ValueError(f"{expected_model_key} oof_proposals 第 {row_number} 行 UID 重复")
        proposal_uids.add(proposal_uid)
        image_id = int(row["image_id"])
        category_id = int(row["category_id"])
        fold = int(row["fold"])
        ledger = image_ledger.get(image_id)
        if ledger is None:
            raise ValueError(f"{expected_model_key} oof_proposals 第 {row_number} 行 image 未登记")
        if (
            image_id != json_image_id
            or category_id != int(prediction["category_id"])
            or row["model_key"] != expected_model_key
            or fold != int(ledger["fold"])
            or row["checkpoint_sha256"].strip().lower() != ledger["checkpoint_sha256"]
        ):
            raise ValueError(
                f"{expected_model_key} oof_proposals 第 {row_number} 行身份字段 "
                "与 JSON/image ledger 不一致"
            )
        source_index = int(row["source_prediction_index"])
        if source_index != fold_source_indices[fold]:
            raise ValueError(
                f"{expected_model_key} oof_proposals 第 {row_number} 行 "
                "source_prediction_index 非连续"
            )
        fold_source_indices[fold] += 1
        csv_values = [
            float(row["x"]),
            float(row["y"]),
            float(row["width"]),
            float(row["height"]),
            float(row["score"]),
        ]
        # ``audit_and_aggregate_oof`` freezes proposal CSV numbers with
        # ``.10g``.  Compare against that exact serialization contract, then
        # require a strict 1e-12 absolute tolerance after parsing.
        json_values = [
            float(f"{float(value):.10g}") for value in (*prediction["bbox"], prediction["score"])
        ]
        if any(
            not math.isclose(csv_value, json_value, rel_tol=0.0, abs_tol=1e-12)
            for csv_value, json_value in zip(
                csv_values,
                json_values,
                strict=True,
            )
        ):
            raise ValueError(
                f"{expected_model_key} oof_proposals 第 {row_number} 行数值 "
                "与 predictions JSON 不一致"
            )
    result = dict(metadata)
    result["metadata_path"] = str(metadata_path)
    result["metadata_sha256"] = sha256_file(metadata_path)
    result["predictions_path"] = str(predictions_path)
    result["predictions_sha256"] = sha256_file(predictions_path)
    result["image_ids"] = seen_image_ids
    result["relocated_artifacts"] = relocated_artifacts
    result["aggregate_cross_file_audit"] = {
        "status": "pass",
        "proposal_count": len(proposal_rows),
        "image_prediction_counts_match": True,
        "proposal_rows_match_predictions_json": True,
        "source_prediction_indices_contiguous_by_fold": True,
    }
    return result, pred_boxes


def _metrics_payload(metrics: OverallMetrics) -> dict[str, Any]:
    return {
        "overall_recall": metrics.recall,
        "overall_fdr": metrics.fdr,
        "details": dict(metrics.details),
        "per_class": {
            name: {
                "recall": item.recall,
                "fdr": item.fdr,
                "tp": item.tp,
                "fp": item.fp,
                "fn": item.fn,
            }
            for name, item in metrics.per_class.items()
        },
    }


def build_threshold_curve(
    gt_boxes: dict[int, list[dict[str, Any]]],
    pred_boxes: dict[int, list[dict[str, Any]]],
    *,
    thresholds: Sequence[float],
    protocol: EvaluationProtocol,
    latency_seconds: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build an exact score-prefix curve from one official matching trace.

    Official matching is score-descending.  Removing a low-score suffix cannot
    alter an earlier prediction's TP/FP state, so one full trace is sufficient.
    Three direct re-evaluations are hard-gated as an implementation parity
    check.
    """

    ordered_thresholds = sorted({float(value) for value in thresholds})
    if not ordered_thresholds:
        raise ValueError("thresholds 不能为空")
    floor = ordered_thresholds[0]
    floor_predictions = filter_predictions_by_score(pred_boxes, floor)
    floor_metrics, trace = evaluate_predictions_with_trace(
        gt_boxes,
        floor_predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    tp_keys = {(match.image_id, match.prediction_index) for match in trace.matches}
    events: list[tuple[float, int, str, bool]] = []
    for image_id, records in floor_predictions.items():
        for prediction_index, record in enumerate(records):
            category_id = int(record["category_id"])
            events.append(
                (
                    float(record["score"]),
                    category_id,
                    protocol.category_mapping[category_id],
                    (image_id, prediction_index) in tp_keys,
                )
            )
    if len(events) != len(trace.matches) + len(trace.unmatched_predictions):
        raise RuntimeError("official trace 未覆盖所有 candidate-floor predictions")
    events.sort(key=lambda item: item[0], reverse=True)
    total_gt = {
        class_name: sum(
            int(protocol.category_mapping[int(item["category_id"])] == class_name)
            for records in gt_boxes.values()
            for item in records
        )
        for class_name in protocol.class_names
    }
    fine_ids_by_coarse = {
        class_name: sorted(
            category_id
            for category_id, mapped in protocol.category_mapping.items()
            if mapped == class_name
        )
        for class_name in protocol.class_names
    }
    fine_total_gt = Counter(
        int(item["category_id"])
        for records in gt_boxes.values()
        for item in records
    )
    tp_counts = Counter({name: 0 for name in protocol.class_names})
    pred_counts = Counter({name: 0 for name in protocol.class_names})
    fine_tp_counts: Counter[int] = Counter()
    fine_pred_counts: Counter[int] = Counter()
    cursor = 0
    rows_desc: list[dict[str, Any]] = []
    for threshold in sorted(ordered_thresholds, reverse=True):
        while cursor < len(events) and events[cursor][0] >= threshold:
            _, category_id, class_name, is_tp = events[cursor]
            pred_counts[class_name] += 1
            tp_counts[class_name] += int(is_tp)
            fine_pred_counts[category_id] += 1
            fine_tp_counts[category_id] += int(is_tp)
            cursor += 1
        total_tp = sum(tp_counts.values())
        total_pred = sum(pred_counts.values())
        total_ground_truth = sum(total_gt.values())
        row: dict[str, Any] = {
            "threshold": threshold,
            "detections_kept": total_pred,
            "overall_recall": (total_tp / total_ground_truth if total_ground_truth else 1.0),
            "overall_fdr": ((total_pred - total_tp) / total_pred if total_pred else 0.0),
            "tp": total_tp,
            "fp": total_pred - total_tp,
            "fn": total_ground_truth - total_tp,
        }
        row["pooled_recall"] = row["overall_recall"]
        row["pooled_fdr"] = row["overall_fdr"]
        coarse_macro_recall: dict[str, float] = {}
        coarse_macro_fdr: dict[str, float] = {}
        for class_name, fine_ids in fine_ids_by_coarse.items():
            if not fine_ids:
                continue
            recalls = []
            fdrs = []
            for category_id in fine_ids:
                gt_count = fine_total_gt[category_id]
                tp_count = fine_tp_counts[category_id]
                pred_count = fine_pred_counts[category_id]
                recall = tp_count / gt_count if gt_count else 1.0
                fdr = (pred_count - tp_count) / pred_count if pred_count else 0.0
                recalls.append(recall)
                fdrs.append(fdr)
                row[f"fine_{category_id}_tp"] = tp_count
                row[f"fine_{category_id}_fp"] = pred_count - tp_count
                row[f"fine_{category_id}_fn"] = gt_count - tp_count
                row[f"fine_{category_id}_recall"] = recall
                row[f"fine_{category_id}_fdr"] = fdr
            coarse_macro_recall[class_name] = sum(recalls) / len(recalls)
            coarse_macro_fdr[class_name] = sum(fdrs) / len(fdrs)
            row[f"{class_name}_macro_recall"] = coarse_macro_recall[class_name]
            row[f"{class_name}_macro_fdr"] = coarse_macro_fdr[class_name]

        full_platform = set(("ship", "aircraft", "vehicle")) <= set(
            coarse_macro_recall
        )
        if full_platform:
            from rsdet.evaluation.absolute_score import platform_confirmed_score

            platform_rows = {
                name: {
                    "recall": coarse_macro_recall[name],
                    "fdr": coarse_macro_fdr[name],
                }
                for name in ("ship", "aircraft", "vehicle")
            }
            gate_recall = sum(item["recall"] for item in platform_rows.values()) / 3.0
            gate_fdr = sum(item["fdr"] for item in platform_rows.values()) / 3.0
            row["platform_gate_recall"] = gate_recall
            row["platform_gate_fdr"] = gate_fdr
            row["platform_recall_pass"] = gate_recall >= protocol.recall_min
            row["platform_fdr_pass"] = gate_fdr <= protocol.fdr_max
            row["official_gate_passed"] = bool(
                row["platform_recall_pass"] and row["platform_fdr_pass"]
            )
            score_latency = 0.0 if latency_seconds is None else float(latency_seconds)
            score = platform_confirmed_score(platform_rows, score_latency)
            row["platform_quality_score"] = sum(score["seven_subscores"][:6]) / 6.0
            row["absolute_score"] = (
                None if latency_seconds is None else float(score["total_score"])
            )
        else:
            # Sub-curves remain valid fitting primitives but are never formal gates.
            row["official_gate_passed"] = None
        for class_name in protocol.class_names:
            class_tp = tp_counts[class_name]
            class_pred = pred_counts[class_name]
            class_gt = total_gt[class_name]
            row.update(
                {
                    f"{class_name}_recall": class_tp / class_gt if class_gt else 1.0,
                    f"{class_name}_fdr": (
                        (class_pred - class_tp) / class_pred if class_pred else 0.0
                    ),
                    f"{class_name}_tp": class_tp,
                    f"{class_name}_fp": class_pred - class_tp,
                    f"{class_name}_fn": class_gt - class_tp,
                }
            )
        rows_desc.append(row)
    rows = sorted(rows_desc, key=lambda item: item["threshold"])

    parity_thresholds = sorted(
        {
            ordered_thresholds[0],
            ordered_thresholds[len(ordered_thresholds) // 2],
            ordered_thresholds[-1],
        }
    )
    by_threshold = {row["threshold"]: row for row in rows}
    parity_rows: list[dict[str, Any]] = []
    for threshold in parity_thresholds:
        direct = evaluate_predictions(
            gt_boxes,
            filter_predictions_by_score(pred_boxes, threshold),
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
        )
        derived = by_threshold[threshold]
        if (
            direct.details["tp"] != derived["tp"]
            or direct.details["fp"] != derived["fp"]
            or direct.details["fn"] != derived["fn"]
            or not math.isclose(
                direct.recall,
                float(derived["overall_recall"]),
                abs_tol=1e-15,
            )
            or not math.isclose(
                direct.fdr,
                float(derived["overall_fdr"]),
                abs_tol=1e-15,
            )
        ):
            raise RuntimeError(f"threshold curve 与官方直接评估不一致: {threshold}")
        parity_rows.append(
            {
                "threshold": threshold,
                "tp": direct.details["tp"],
                "fp": direct.details["fp"],
                "fn": direct.details["fn"],
                "status": "pass",
            }
        )
    if floor_metrics.details["tp"] != by_threshold[floor]["tp"]:
        raise RuntimeError("candidate floor trace parity 失败")
    return rows, {
        "status": "pass",
        "method": "official_score_prefix_trace_with_direct_spot_checks",
        "direct_checks": parity_rows,
    }


def select_exploratory_workpoint(
    curve: Sequence[Mapping[str, Any]],
    *,
    recall_min: float,
    fdr_max: float,
) -> dict[str, Any]:
    """Select the same-OOF descriptive workpoint without deployment admission."""

    if not curve:
        raise ValueError("curve cannot be empty")
    is_full_platform = all(
        "platform_gate_recall" in row and "platform_gate_fdr" in row for row in curve
    )
    recall_key = "platform_gate_recall" if is_full_platform else "overall_recall"
    fdr_key = "platform_gate_fdr" if is_full_platform else "overall_fdr"
    feasible = [row for row in curve if float(row[fdr_key]) <= fdr_max]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                float(row[recall_key]),
                -float(row[fdr_key]),
                float(row["threshold"]),
            ),
        )
        policy = "max_recall_under_official_fdr_then_low_fdr_then_high_threshold"
    else:
        selected = max(
            curve,
            key=lambda row: (
                -float(row[fdr_key]),
                float(row[recall_key]),
                float(row["threshold"]),
            ),
        )
        policy = "fallback_min_fdr_then_high_recall_then_high_threshold"
    return {
        "threshold": float(selected["threshold"]),
        "policy": policy,
        "same_oof_selection": True,
        "exploratory_only": True,
        "deployment_admission": False,
        "metric_scope": (
            "platform_observed_20260831" if is_full_platform else "diagnostic_subprotocol"
        ),
        "official_gate_passed": bool(
            is_full_platform
            and float(selected[recall_key]) >= recall_min
            and float(selected[fdr_key]) <= fdr_max
        ),
        "metrics": dict(selected),
    }


def _greedy_diagnostic_pairs(
    predictions: Sequence[tuple[int, int]],
    ground_truths: Sequence[tuple[int, int]],
    *,
    pred_boxes: Mapping[int, Sequence[Mapping[str, Any]]],
    gt_boxes: Mapping[int, Sequence[Mapping[str, Any]]],
    edge_allowed: Callable[
        [Mapping[str, Any], Mapping[str, Any], float],
        bool,
    ],
) -> list[tuple[tuple[int, int], tuple[int, int], float]]:
    edges: list[tuple[float, float, int, int, int, tuple[int, int], tuple[int, int]]] = []
    gt_by_image: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
    for key in ground_truths:
        gt_by_image[key[0]].append(key)
    for pred_key in predictions:
        image_id, pred_index = pred_key
        prediction = pred_boxes[image_id][pred_index]
        for gt_key in gt_by_image.get(image_id, ()):
            gt = gt_boxes[image_id][gt_key[1]]
            iou = compute_iou(
                list(prediction["bbox_xyxy"]),
                list(gt["bbox_xyxy"]),
            )
            if edge_allowed(prediction, gt, iou):
                edges.append(
                    (
                        -iou,
                        -float(prediction["score"]),
                        image_id,
                        pred_index,
                        gt_key[1],
                        pred_key,
                        gt_key,
                    )
                )
    edges.sort()
    used_predictions: set[tuple[int, int]] = set()
    used_ground_truths: set[tuple[int, int]] = set()
    result: list[tuple[tuple[int, int], tuple[int, int], float]] = []
    for negative_iou, _, _, _, _, pred_key, gt_key in edges:
        if pred_key in used_predictions or gt_key in used_ground_truths:
            continue
        used_predictions.add(pred_key)
        used_ground_truths.add(gt_key)
        result.append((pred_key, gt_key, -negative_iou))
    return result


def decompose_official_errors(
    formal_gt: FormalGroundTruth,
    pred_boxes: dict[int, list[dict[str, Any]]],
    *,
    threshold: float,
    protocol: EvaluationProtocol,
    model_key: str,
    include_cases: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[int, int], str]]:
    """Attribute official FP/FN with a frozen, count-conserving hierarchy."""

    filtered = filter_predictions_by_score(pred_boxes, threshold)
    metrics, trace = evaluate_predictions_with_trace(
        formal_gt.boxes,
        filtered,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    unmatched_predictions = {
        (item.image_id, item.prediction_index) for item in trace.unmatched_predictions
    }
    unmatched_ground_truths = {
        (item.image_id, item.ground_truth_index) for item in trace.unmatched_ground_truths
    }
    fp_reason: dict[tuple[int, int], str] = {}
    fn_reason: dict[tuple[int, int], str] = {}
    partner: dict[tuple[str, int, int], tuple[int, int, float]] = {}

    for pred_key in sorted(unmatched_predictions):
        image_id, pred_index = pred_key
        prediction = filtered[image_id][pred_index]
        category_id = int(prediction["category_id"])
        class_name = protocol.category_mapping[category_id]
        threshold_iou = protocol.iou_thresholds[class_name]
        if any(
            int(gt["category_id"]) == category_id
            and compute_iou(
                list(prediction["bbox_xyxy"]),
                list(gt["bbox_xyxy"]),
            )
            >= threshold_iou
            for gt in formal_gt.boxes.get(image_id, ())
        ):
            fp_reason[pred_key] = "FP_DUP"

    remaining_predictions = unmatched_predictions - set(fp_reason)
    class_pairs = _greedy_diagnostic_pairs(
        sorted(remaining_predictions),
        sorted(unmatched_ground_truths),
        pred_boxes=filtered,
        gt_boxes=formal_gt.boxes,
        edge_allowed=lambda prediction, gt, iou: (
            int(prediction["category_id"]) != int(gt["category_id"])
            and iou >= protocol.iou_thresholds[protocol.category_mapping[int(gt["category_id"])]]
        ),
    )
    for pred_key, gt_key, iou in class_pairs:
        fp_reason[pred_key] = "FP_CLS"
        fn_reason[gt_key] = "FN_CLS"
        partner[("pred", *pred_key)] = (*gt_key, iou)
        partner[("gt", *gt_key)] = (*pred_key, iou)

    remaining_predictions -= set(fp_reason)
    remaining_ground_truths = unmatched_ground_truths - set(fn_reason)
    localization_pairs = _greedy_diagnostic_pairs(
        sorted(remaining_predictions),
        sorted(remaining_ground_truths),
        pred_boxes=filtered,
        gt_boxes=formal_gt.boxes,
        edge_allowed=lambda prediction, gt, iou: (
            int(prediction["category_id"]) == int(gt["category_id"])
            and 0.0
            < iou
            < protocol.iou_thresholds[protocol.category_mapping[int(gt["category_id"])]]
        ),
    )
    for pred_key, gt_key, iou in localization_pairs:
        fp_reason[pred_key] = "FP_LOC"
        fn_reason[gt_key] = "FN_LOC"
        partner[("pred", *pred_key)] = (*gt_key, iou)
        partner[("gt", *gt_key)] = (*pred_key, iou)

    for pred_key in unmatched_predictions - set(fp_reason):
        fp_reason[pred_key] = "FP_BG"
    for gt_key in unmatched_ground_truths - set(fn_reason):
        fn_reason[gt_key] = "FN_MISS"

    fp_counts = Counter(fp_reason.values())
    fn_counts = Counter(fn_reason.values())
    if sum(fp_counts.values()) != int(metrics.details["fp"]):
        raise RuntimeError("FP error decomposition 不守恒")
    if sum(fn_counts.values()) != int(metrics.details["fn"]):
        raise RuntimeError("FN error decomposition 不守恒")

    cases: list[dict[str, Any]] = []
    for pred_key, reason in sorted(fp_reason.items()) if include_cases else ():
        image_id, pred_index = pred_key
        prediction = filtered[image_id][pred_index]
        partner_data = partner.get(("pred", *pred_key))
        gt_object = formal_gt.objects[(partner_data[0], partner_data[1])] if partner_data else None
        cases.append(
            {
                "model_key": model_key,
                "threshold": threshold,
                "case_side": "prediction",
                "reason": reason,
                "image_id": image_id,
                "item_uid": f"{model_key.lower()}-i{image_id}-p{pred_index:04d}",
                "category_id": int(prediction["category_id"]),
                "class_name": protocol.category_mapping[int(prediction["category_id"])],
                "score": float(prediction["score"]),
                "bbox_xyxy": " ".join(f"{float(value):.10g}" for value in prediction["bbox_xyxy"]),
                "paired_item_uid": (gt_object.annotation_uid if gt_object is not None else ""),
                "paired_category_id": (gt_object.category_id if gt_object is not None else ""),
                "paired_iou": partner_data[2] if partner_data else "",
            }
        )
    for gt_key, reason in sorted(fn_reason.items()) if include_cases else ():
        gt_object = formal_gt.objects[gt_key]
        partner_data = partner.get(("gt", *gt_key))
        prediction = filtered[partner_data[0]][partner_data[1]] if partner_data else None
        cases.append(
            {
                "model_key": model_key,
                "threshold": threshold,
                "case_side": "ground_truth",
                "reason": reason,
                "image_id": gt_object.image_id,
                "item_uid": gt_object.annotation_uid,
                "category_id": gt_object.category_id,
                "class_name": gt_object.class_name,
                "score": "",
                "bbox_xyxy": " ".join(f"{value:.10g}" for value in gt_object.bbox_xyxy),
                "paired_item_uid": (
                    f"{model_key.lower()}-i{partner_data[0]}-p{partner_data[1]:04d}"
                    if partner_data
                    else ""
                ),
                "paired_category_id": (int(prediction["category_id"]) if prediction else ""),
                "paired_iou": partner_data[2] if partner_data else "",
            }
        )

    per_class: dict[str, dict[str, int]] = {}
    for class_name in protocol.class_names:
        row = {name: 0 for name in (*ERROR_HIERARCHY, *FN_HIERARCHY)}
        for pred_key, reason in fp_reason.items():
            prediction = filtered[pred_key[0]][pred_key[1]]
            if protocol.category_mapping[int(prediction["category_id"])] == class_name:
                row[reason] += 1
        for gt_key, reason in fn_reason.items():
            if formal_gt.objects[gt_key].class_name == class_name:
                row[reason] += 1
        per_class[class_name] = row
    per_fine_category: dict[str, dict[str, int]] = {}
    for category_id in sorted(protocol.category_mapping):
        row = {name: 0 for name in (*ERROR_HIERARCHY, *FN_HIERARCHY)}
        for pred_key, reason in fp_reason.items():
            prediction = filtered[pred_key[0]][pred_key[1]]
            if int(prediction["category_id"]) == category_id:
                row[reason] += 1
        for gt_key, reason in fn_reason.items():
            if formal_gt.objects[gt_key].category_id == category_id:
                row[reason] += 1
        per_fine_category[str(category_id)] = row
    summary = {
        "model_key": model_key,
        "threshold": threshold,
        "official_metrics": _metrics_payload(metrics),
        "diagnostic_not_official_metric": True,
        "hierarchy": [
            "official TP",
            "FP_DUP: same-fine GT above official IoU, already consumed",
            "FP_CLS/FN_CLS: one-to-one different-fine overlap above GT threshold",
            "FP_LOC/FN_LOC: one-to-one same-fine partial overlap below threshold",
            "FP_BG: remaining unattributed prediction",
            "FN_MISS: remaining unattributed GT",
        ],
        "fp_counts": {name: fp_counts[name] for name in ERROR_HIERARCHY},
        "fn_counts": {name: fn_counts[name] for name in FN_HIERARCHY},
        "per_class": per_class,
        "per_fine_category": per_fine_category,
        "conservation": {
            "official_fp": int(metrics.details["fp"]),
            "decomposed_fp": sum(fp_counts.values()),
            "official_fn": int(metrics.details["fn"]),
            "decomposed_fn": sum(fn_counts.values()),
            "passed": True,
        },
        "fp_bg_semantics": (
            "unattributed after duplicate/class/localization rules; "
            "not proof of semantic background"
        ),
    }
    return summary, cases, fn_reason


def _trace_match_map(
    trace: OfficialEvaluationTrace,
) -> dict[tuple[int, int], Any]:
    return {(match.image_id, match.ground_truth_index): match for match in trace.matches}


def build_paired_object_outcomes(
    formal_gt: FormalGroundTruth,
    model_predictions: Mapping[str, dict[int, list[dict[str, Any]]]],
    *,
    thresholds: Mapping[str, float],
    protocol: EvaluationProtocol,
    label: str,
    precomputed_fn_reasons: (Mapping[str, Mapping[tuple[int, int], str]] | None) = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare M1/M3 on the exact same GT objects and compute recall oracle."""

    matches: dict[str, dict[tuple[int, int], Any]] = {}
    fn_reasons: dict[str, dict[tuple[int, int], str]] = {}
    official_tp: dict[str, int] = {}
    for model_key in EXPECTED_MODEL_KEYS:
        filtered = filter_predictions_by_score(
            model_predictions[model_key],
            float(thresholds[model_key]),
        )
        metrics, trace = evaluate_predictions_with_trace(
            formal_gt.boxes,
            filtered,
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
        )
        matches[model_key] = _trace_match_map(trace)
        official_tp[model_key] = int(metrics.details["tp"])
        if precomputed_fn_reasons is not None:
            supplied = precomputed_fn_reasons.get(model_key)
            if supplied is None:
                raise ValueError(f"precomputed FN reasons 缺少 {model_key}")
            fn_reasons[model_key] = dict(supplied)
        else:
            _, _, fn_reason = decompose_official_errors(
                formal_gt,
                model_predictions[model_key],
                threshold=float(thresholds[model_key]),
                protocol=protocol,
                model_key=model_key,
                include_cases=False,
            )
            fn_reasons[model_key] = fn_reason

    rows: list[dict[str, Any]] = []
    summary_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for gt_key, gt_object in sorted(
        formal_gt.objects.items(),
        key=lambda item: item[1].annotation_uid,
    ):
        m1_match = matches["M1"].get(gt_key)
        m3_match = matches["M3"].get(gt_key)
        m1_hit = m1_match is not None
        m3_hit = m3_match is not None
        if m1_hit and m3_hit:
            outcome = "both"
        elif m1_hit:
            outcome = "M1_only"
        elif m3_hit:
            outcome = "M3_only"
        else:
            outcome = "neither"
        for scope in (
            "overall",
            gt_object.class_name,
            f"category_{gt_object.category_id:02d}",
        ):
            summary_counts[scope][outcome] += 1
        rows.append(
            {
                "analysis_label": label,
                "annotation_uid": gt_object.annotation_uid,
                "image_id": gt_object.image_id,
                "fold": gt_object.fold,
                "group_id": gt_object.group_id,
                "category_id": gt_object.category_id,
                "class_name": gt_object.class_name,
                "gt_bbox_xyxy": " ".join(f"{value:.10g}" for value in gt_object.bbox_xyxy),
                "M1_threshold": float(thresholds["M1"]),
                "M1_matched": m1_hit,
                "M1_score": m1_match.score if m1_match else "",
                "M1_iou": m1_match.iou if m1_match else "",
                "M1_fn_reason": "" if m1_match else fn_reasons["M1"][gt_key],
                "M3_threshold": float(thresholds["M3"]),
                "M3_matched": m3_hit,
                "M3_score": m3_match.score if m3_match else "",
                "M3_iou": m3_match.iou if m3_match else "",
                "M3_fn_reason": "" if m3_match else fn_reasons["M3"][gt_key],
                "paired_outcome": outcome,
                "oracle_union_matched": m1_hit or m3_hit,
            }
        )
    for model_key in EXPECTED_MODEL_KEYS:
        if sum(bool(row[f"{model_key}_matched"]) for row in rows) != official_tp[model_key]:
            raise RuntimeError(f"{model_key} object pairing 与官方 TP 不一致")

    summaries: dict[str, Any] = {}
    for scope, counts in sorted(summary_counts.items()):
        total = sum(counts.values())
        oracle = counts["both"] + counts["M1_only"] + counts["M3_only"]
        summaries[scope] = {
            "objects": total,
            "both": counts["both"],
            "M1_only": counts["M1_only"],
            "M3_only": counts["M3_only"],
            "neither": counts["neither"],
            "M1_recall": (counts["both"] + counts["M1_only"]) / total,
            "M3_recall": (counts["both"] + counts["M3_only"]) / total,
            "oracle_union_recall": oracle / total,
        }
    return rows, {
        "label": label,
        "thresholds": dict(thresholds),
        "oracle_definition": "GT is recovered when either model has an official TP",
        "oracle_fdr": None,
        "deployable_ensemble_claim": False,
        "summary": summaries,
    }


def _ground_truth_coco(formal_gt: FormalGroundTruth) -> dict[str, Any]:
    images = [{"id": image_id} for image_id in sorted(formal_gt.image_ids)]
    annotations: list[dict[str, Any]] = []
    for annotation_id, (_, item) in enumerate(
        sorted(
            formal_gt.objects.items(),
            key=lambda pair: pair[1].annotation_uid,
        ),
        1,
    ):
        x1, y1, x2, y2 = item.bbox_xyxy
        annotations.append(
            {
                "id": annotation_id,
                "annotation_uid": item.annotation_uid,
                "image_id": item.image_id,
                "category_id": item.category_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            }
        )
    return {
        "info": {
            "source": FORMAL_CROP_MANIFEST_VERSION,
            "source_sha256": FORMAL_CROP_MANIFEST_SHA256,
            "coordinate_semantics": "continuous_float_xywh_derived_from_half_open_xyxy",
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": index} for index in range(25)],
    }


def run_formal_oof_analysis(
    *,
    config_path: str | Path,
    project_config_path: str | Path,
    formal_crop_manifest_path: str | Path,
    m1_aggregate_dir: str | Path,
    m3_aggregate_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the immutable CPU post-processing task for both formal OOF models."""

    config = load_analysis_config(config_path)
    protocol = parse_evaluation_protocol(load_config(project_config_path))
    if protocol.eval_version != "official_eval_v1":
        raise ValueError("本分析只接受冻结 official_eval_v1")
    formal_gt = load_formal_ground_truth(
        formal_crop_manifest_path,
        expected_sha256=config["formal_crop_manifest_sha256"],
        expected_images=int(config["expected_images"]),
        expected_annotations=int(config["expected_annotations"]),
    )
    for item in formal_gt.objects.values():
        mapped_class = protocol.category_mapping.get(item.category_id)
        if mapped_class != item.class_name:
            raise ValueError(
                "formal GT major_class 与 official protocol mapping 不一致: "
                f"annotation={item.annotation_uid}, category={item.category_id}, "
                f"formal={item.class_name}, protocol={mapped_class}"
            )
    aggregates: dict[str, dict[str, Any]] = {}
    predictions: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for model_key, aggregate_dir in (
        ("M1", m1_aggregate_dir),
        ("M3", m3_aggregate_dir),
    ):
        metadata, model_predictions = load_oof_aggregate(
            aggregate_dir,
            expected_model_key=model_key,
            expected_manifest_sha256=str(config["cv3_manifest_sha256"]),
            expected_formal_crop_sha256=str(config["formal_crop_manifest_sha256"]),
            expected_images=int(config["expected_images"]),
            candidate_floor=float(config["candidate_floor"]),
        )
        if metadata["image_ids"] != set(formal_gt.image_ids):
            raise ValueError(f"{model_key} OOF/formal GT image_id 集合不一致")
        metadata.pop("image_ids")
        aggregates[model_key] = metadata
        predictions[model_key] = model_predictions

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"analysis output 非空，禁止覆盖: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    gt_path = destination / "ground_truth_from_formal_crop.json"
    _write_json(gt_path, _ground_truth_coco(formal_gt))

    workpoints: dict[str, dict[str, Any]] = {}
    model_outputs: dict[str, Any] = {}
    floor_fn_reasons: dict[str, dict[tuple[int, int], str]] = {}
    selected_fn_reasons: dict[str, dict[tuple[int, int], str]] = {}
    threshold_fields = [
        "threshold",
        "detections_kept",
        "overall_recall",
        "overall_fdr",
        "tp",
        "fp",
        "fn",
        "official_gate_passed",
        *[
            f"{name}_{suffix}"
            for name in protocol.class_names
            for suffix in ("recall", "fdr", "tp", "fp", "fn")
        ],
    ]
    error_fields = [
        "model_key",
        "threshold",
        "case_side",
        "reason",
        "image_id",
        "item_uid",
        "category_id",
        "class_name",
        "score",
        "bbox_xyxy",
        "paired_item_uid",
        "paired_category_id",
        "paired_iou",
    ]
    for model_key in EXPECTED_MODEL_KEYS:
        model_dir = destination / model_key
        model_dir.mkdir()
        curve, parity = build_threshold_curve(
            formal_gt.boxes,
            predictions[model_key],
            thresholds=config["thresholds"],
            protocol=protocol,
        )
        _write_csv(
            model_dir / "threshold_curve.csv",
            curve,
            fieldnames=threshold_fields,
        )
        workpoint = select_exploratory_workpoint(
            curve,
            recall_min=protocol.recall_min,
            fdr_max=protocol.fdr_max,
        )
        workpoints[model_key] = workpoint
        _write_json(model_dir / "exploratory_workpoint.json", workpoint)
        floor_summary, _, floor_fn_reason = decompose_official_errors(
            formal_gt,
            predictions[model_key],
            threshold=float(config["candidate_floor"]),
            protocol=protocol,
            model_key=model_key,
            include_cases=False,
        )
        selected_summary, selected_cases, selected_fn_reason = decompose_official_errors(
            formal_gt,
            predictions[model_key],
            threshold=float(workpoint["threshold"]),
            protocol=protocol,
            model_key=model_key,
        )
        floor_fn_reasons[model_key] = floor_fn_reason
        selected_fn_reasons[model_key] = selected_fn_reason
        _write_json(
            model_dir / "candidate_floor_metrics_and_errors.json",
            floor_summary,
        )
        _write_json(
            model_dir / "exploratory_workpoint_metrics_and_errors.json",
            selected_summary,
        )
        _write_csv(
            model_dir / "exploratory_workpoint_error_cases.csv",
            selected_cases,
            fieldnames=error_fields,
        )
        model_outputs[model_key] = {
            "aggregate_metadata_sha256": aggregates[model_key]["metadata_sha256"],
            "predictions_sha256": aggregates[model_key]["predictions_sha256"],
            "candidate_floor": floor_summary["official_metrics"],
            "exploratory_workpoint": workpoint,
            "curve_parity": parity,
            "aggregate_cross_file_audit": aggregates[model_key]["aggregate_cross_file_audit"],
            "relocated_artifacts": aggregates[model_key]["relocated_artifacts"],
        }

    paired_fields = [
        "analysis_label",
        "annotation_uid",
        "image_id",
        "fold",
        "group_id",
        "category_id",
        "class_name",
        "gt_bbox_xyxy",
        "M1_threshold",
        "M1_matched",
        "M1_score",
        "M1_iou",
        "M1_fn_reason",
        "M3_threshold",
        "M3_matched",
        "M3_score",
        "M3_iou",
        "M3_fn_reason",
        "paired_outcome",
        "oracle_union_matched",
    ]
    paired_dir = destination / "paired"
    paired_dir.mkdir()
    candidate_thresholds = {
        model_key: float(config["candidate_floor"]) for model_key in EXPECTED_MODEL_KEYS
    }
    candidate_rows, candidate_summary = build_paired_object_outcomes(
        formal_gt,
        predictions,
        thresholds=candidate_thresholds,
        protocol=protocol,
        label="candidate_floor",
        precomputed_fn_reasons=floor_fn_reasons,
    )
    selected_thresholds = {
        model_key: float(workpoints[model_key]["threshold"]) for model_key in EXPECTED_MODEL_KEYS
    }
    selected_rows, selected_summary = build_paired_object_outcomes(
        formal_gt,
        predictions,
        thresholds=selected_thresholds,
        protocol=protocol,
        label="same_oof_exploratory_workpoints",
        precomputed_fn_reasons=selected_fn_reasons,
    )
    _write_csv(
        paired_dir / "object_outcomes_candidate_floor.csv",
        candidate_rows,
        fieldnames=paired_fields,
    )
    _write_csv(
        paired_dir / "object_outcomes_exploratory_workpoints.csv",
        selected_rows,
        fieldnames=paired_fields,
    )
    complementarity = {
        "candidate_floor": candidate_summary,
        "same_oof_exploratory_workpoints": selected_summary,
    }
    _write_json(paired_dir / "complementarity_and_oracle.json", complementarity)

    metadata = {
        "analysis_version": ANALYSIS_VERSION,
        "status": "complete_formal_oof_descriptive_analysis",
        "scientific_claim_scope": {
            "official_matching_metrics": True,
            "threshold_curve": True,
            "error_decomposition_is_diagnostic": True,
            "same_oof_selected_threshold_is_final": False,
            "oracle_union_is_deployable": False,
        },
        "protocol_versions": {
            "contract_version": protocol.contract_version,
            "eval_version": protocol.eval_version,
        },
        "inputs": {
            "analysis_config": str(Path(config_path).expanduser().resolve()),
            "analysis_config_sha256": config["config_sha256"],
            "project_config": str(Path(project_config_path).expanduser().resolve()),
            "project_config_sha256": sha256_file(project_config_path),
            "formal_crop_manifest": str(Path(formal_crop_manifest_path).expanduser().resolve()),
            "formal_crop_manifest_sha256": sha256_file(formal_crop_manifest_path),
            "M1_aggregate_metadata_sha256": aggregates["M1"]["metadata_sha256"],
            "M3_aggregate_metadata_sha256": aggregates["M3"]["metadata_sha256"],
            "M1_predictions_sha256": aggregates["M1"]["predictions_sha256"],
            "M3_predictions_sha256": aggregates["M3"]["predictions_sha256"],
        },
        "counts": {
            "images": len(formal_gt.image_ids),
            "ground_truth_objects": formal_gt.annotation_count,
        },
        "models": model_outputs,
        "paired": complementarity,
        "artifacts": {},
    }
    for path in sorted(destination.rglob("*")):
        if path.is_file() and path.name != "analysis_metadata.json":
            metadata["artifacts"][str(path.relative_to(destination))] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    metadata_path = destination / "analysis_metadata.json"
    _write_json(metadata_path, metadata)
    return metadata


__all__ = [
    "ANALYSIS_VERSION",
    "ERROR_HIERARCHY",
    "FN_HIERARCHY",
    "FormalGroundTruth",
    "GroundTruthObject",
    "build_paired_object_outcomes",
    "build_threshold_curve",
    "decompose_official_errors",
    "load_analysis_config",
    "load_formal_ground_truth",
    "load_oof_aggregate",
    "run_formal_oof_analysis",
    "select_exploratory_workpoint",
]
