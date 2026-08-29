#!/usr/bin/env python3
"""Apply formal fold-heldout Y5/M3 OER models to pseudo-10K proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.multi_detector_oer import (
    build_multi_detector_features,
    class_aware_nms_records,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pseudo_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    model_key: str,
    score_floor: float,
    stable_offset: int,
) -> list[dict[str, Any]]:
    """Normalize COCO xywh pseudo predictions to the OER feature contract."""
    result: list[dict[str, Any]] = []
    for source_order, raw in enumerate(rows):
        score = float(raw["score"])
        if score < score_floor:
            continue
        x, y, width, height = (float(value) for value in raw["bbox"])
        if width <= 0.0 or height <= 0.0:
            continue
        fold = int(raw["source_fold"])
        if fold not in {0, 1, 2}:
            raise ValueError(f"invalid source_fold={fold}")
        result.append(
            {
                "image_id": int(raw["image_id"]),
                "fold": fold,
                "category_id": int(raw["category_id"]),
                "bbox_xyxy": [x, y, x + width, y + height],
                "score": score,
                "detector_score": score,
                "model_key": model_key.upper(),
                "stable_order": stable_offset + source_order,
            }
        )
    return result


def to_coco_rows(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Serialize scored OER records without leaking internal feature fields."""
    output: list[dict[str, Any]] = []
    for item in records:
        x0, y0, x1, y1 = (float(value) for value in item["bbox_xyxy"])
        output.append(
            {
                "image_id": int(item["image_id"]),
                "category_id": int(item["category_id"]),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": float(item["score"]),
                "source_fold": int(item["fold"]),
                "source_model": str(item["model_key"]),
            }
        )
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--y5-predictions", type=Path, required=True)
    parser.add_argument("--m3-predictions", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--formal-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", default="score_geometry_agreement")
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--y5-floor", type=float, default=0.001)
    parser.add_argument("--m3-floor", type=float, default=0.03)
    parser.add_argument("--pre-nms-iou", type=float, default=0.50)
    parser.add_argument("--post-nms-iou", type=float, default=0.50)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    y5_rows = json.loads(args.y5_predictions.read_text(encoding="utf-8"))
    m3_rows = json.loads(args.m3_predictions.read_text(encoding="utf-8"))
    y5 = class_aware_nms_records(
        load_pseudo_records(
            y5_rows,
            model_key="Y5",
            score_floor=args.y5_floor,
            stable_offset=0,
        ),
        iou_threshold=args.pre_nms_iou,
    )
    m3 = class_aware_nms_records(
        load_pseudo_records(
            m3_rows,
            model_key="M3",
            score_floor=args.m3_floor,
            stable_offset=2_000_000,
        ),
        iou_threshold=args.pre_nms_iou,
    )
    records = y5 + m3
    records.sort(key=lambda item: int(item["stable_order"]))
    matrix, all_columns = build_multi_detector_features(
        records, category_mapping=protocol.category_mapping
    )
    all_column_index = {name: index for index, name in enumerate(all_columns)}
    probabilities = np.full(len(records), np.nan, dtype=np.float64)
    model_shas: dict[str, str] = {}
    model_columns: list[str] | None = None
    for fold in (0, 1, 2):
        path = args.model_dir / f"{args.variant}_heldout_fold{fold}.joblib"
        payload = joblib.load(path)
        columns = [str(value) for value in payload["columns"]]
        if model_columns is None:
            model_columns = columns
        elif columns != model_columns:
            raise RuntimeError("fold OER model columns differ")
        indices = [all_column_index[column] for column in columns]
        mask = np.asarray([int(item["fold"]) == fold for item in records])
        probabilities[mask] = payload["model"].predict_proba(matrix[mask][:, indices])[:, 1]
        model_shas[str(fold)] = _sha256(path)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("incomplete pseudo OER probabilities")

    scored: list[dict[str, Any]] = []
    for item, probability in zip(records, probabilities, strict=True):
        record = dict(item)
        record["score"] = float(probability)
        scored.append(record)
    scored = class_aware_nms_records(scored, iou_threshold=args.post_nms_iou)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = args.output_dir / "predictions.json"
    scored_path.write_text(
        json.dumps(to_coco_rows(scored), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    formal = json.loads(args.formal_summary.read_text(encoding="utf-8"))
    formal_variant = formal["variants"][args.variant]["frontier"]["levels"]
    threshold_outputs: dict[str, Any] = {}
    for level, level_payload in formal_variant.items():
        thresholds = {
            int(fold): float(value)
            for fold, value in level_payload["crossfit_thresholds"].items()
        }
        selected = [
            item for item in scored if float(item["score"]) >= thresholds[int(item["fold"])]
        ]
        path = args.output_dir / f"formal_threshold_{level.replace('.', 'p')}.json"
        path.write_text(
            json.dumps(to_coco_rows(selected), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        threshold_outputs[level] = {
            "thresholds": {str(key): value for key, value in thresholds.items()},
            "predictions": len(selected),
            "sha256": _sha256(path),
            "path": str(path),
        }

    summary = {
        "status": "complete",
        "protocol": "formal_fold_heldout_y5_m3_oer_on_pseudo10k_v1",
        "variant": args.variant,
        "counts": {
            "Y5_raw": len(y5_rows),
            "Y5_after_pre_nms": len(y5),
            "M3_raw": len(m3_rows),
            "M3_after_pre_nms": len(m3),
            "combined": len(records),
            "after_post_nms": len(scored),
        },
        "columns": model_columns,
        "model_sha256": model_shas,
        "predictions_sha256": _sha256(scored_path),
        "formal_threshold_outputs": threshold_outputs,
        "inputs": {
            "Y5": _sha256(args.y5_predictions),
            "M3": _sha256(args.m3_predictions),
            "formal_summary": _sha256(args.formal_summary),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
