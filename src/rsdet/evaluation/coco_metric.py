"""Reusable COCO-file entry point for the official Recall/FDR metric."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsdet.evaluation.official_metric import OverallMetrics, evaluate_predictions
from rsdet.utils.config import load_config


def _xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    if len(box) != 4 or box[2] < 0 or box[3] < 0:
        raise ValueError(f"非法 COCO bbox: {box}")
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _group_annotations(
    annotations: Sequence[Mapping[str, Any]],
    *,
    require_score: bool,
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        record: dict[str, Any] = {
            "bbox_xyxy": _xywh_to_xyxy(
                [float(value) for value in annotation["bbox"]]
            ),
            "category_id": int(annotation["category_id"]),
        }
        if require_score:
            if "score" not in annotation:
                raise ValueError("预测记录缺少 score")
            record["score"] = float(annotation["score"])
        grouped.setdefault(image_id, []).append(record)
    return grouped


def load_coco_ground_truth(path: Path) -> dict[int, list[dict[str, Any]]]:
    """Load a COCO ground-truth object and group annotations by image id."""
    data = _load_json(path)
    if not isinstance(data, Mapping) or not isinstance(data.get("annotations"), list):
        raise ValueError("GT 必须是包含 annotations 列表的 COCO JSON 对象")
    return _group_annotations(data["annotations"], require_score=False)


def load_coco_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    """Load a COCO detection list and group predictions by image id."""
    data = _load_json(path)
    if isinstance(data, list):
        annotations = data
    elif isinstance(data, Mapping) and isinstance(data.get("annotations"), list):
        annotations = data["annotations"]
    else:
        raise ValueError("预测文件必须是 COCO detection 列表或包含 annotations 的对象")
    return _group_annotations(annotations, require_score=True)


def evaluate_coco_files(
    gt_path: Path,
    prediction_path: Path,
    project_config_path: Path = Path("configs/project.yaml"),
    *,
    class_names: Sequence[str] | None = None,
) -> tuple[OverallMetrics, float, float]:
    """Evaluate two COCO files and return metrics plus the competition gates."""
    project_config = load_config(project_config_path)
    task_config = project_config["task"]
    official_config = project_config["official_evaluation"]
    names = list(class_names or task_config["class_names"])
    category_mapping = {
        int(category_id): str(class_name)
        for category_id, class_name in task_config["dataset_category_mapping"].items()
    }
    iou_thresholds = {
        str(class_name): float(threshold)
        for class_name, threshold in official_config["iou_thresholds"].items()
    }
    result = evaluate_predictions(
        load_coco_ground_truth(gt_path),
        load_coco_predictions(prediction_path),
        class_names=names,
        category_mapping=category_mapping,
        iou_thresholds=iou_thresholds,
    )
    return (
        result,
        float(official_config["recall_min"]),
        float(official_config["fdr_max"]),
    )


def metrics_document(
    result: OverallMetrics,
    *,
    recall_min: float,
    fdr_max: float,
) -> dict[str, Any]:
    """Serialize an official metric result using the project's stable schema."""
    return {
        "overall_recall": result.recall,
        "overall_fdr": result.fdr,
        "detection_gate": {
            "recall_min": recall_min,
            "fdr_max": fdr_max,
            "passed": result.recall >= recall_min and result.fdr <= fdr_max,
        },
        "details": result.details,
        "per_class": {
            name: {
                "recall": metrics.recall,
                "fdr": metrics.fdr,
                "tp": metrics.tp,
                "fp": metrics.fp,
                "fn": metrics.fn,
            }
            for name, metrics in result.per_class.items()
        },
    }


def evaluate_and_write(
    gt_path: Path,
    prediction_path: Path,
    output_path: Path,
    project_config_path: Path = Path("configs/project.yaml"),
    *,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run official evaluation and atomically replace the requested JSON file."""
    result, recall_min, fdr_max = evaluate_coco_files(
        gt_path,
        prediction_path,
        project_config_path,
        class_names=class_names,
    )
    document = metrics_document(
        result,
        recall_min=recall_min,
        fdr_max=fdr_max,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return document


__all__ = [
    "evaluate_and_write",
    "evaluate_coco_files",
    "load_coco_ground_truth",
    "load_coco_predictions",
    "metrics_document",
]
