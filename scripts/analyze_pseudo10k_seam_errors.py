#!/usr/bin/env python3
"""Measure prediction errors caused or confounded by pseudo-10K construction."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_predictions_with_trace
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.tiling.slicer import generate_tiles
from rsdet.utils.config import load_config

CANVAS_SIZE = 10_000
GRID = 10
CELL_SIZE = CANVAS_SIZE // GRID


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_source(path_text: str, source_root: Path) -> Path:
    original = Path(path_text)
    candidates = (
        original,
        source_root / original.name,
        source_root / "images" / "train" / original.name,
        source_root / "train" / original.name,
    )
    existing = list(dict.fromkeys(path for path in candidates if path.is_file()))
    if len(existing) != 1:
        raise FileNotFoundError(f"cannot uniquely resolve pseudo source {path_text!r}: {existing}")
    return existing[0]


def _source_content_rects(
    document: dict[str, Any], source_root: Path
) -> dict[int, list[tuple[float, float, float, float]]]:
    output: dict[int, list[tuple[float, float, float, float]]] = {}
    for image in document["images"]:
        sources = image.get("source_images")
        if not isinstance(sources, list) or len(sources) != 100:
            raise ValueError("each pseudo image must declare exactly 100 source_images")
        rects = []
        for index, path_text in enumerate(sources):
            with Image.open(_resolve_source(str(path_text), source_root)) as opened:
                width, height = opened.size
            scale = min(CELL_SIZE / width, CELL_SIZE / height)
            resized_width = max(1, round(width * scale))
            resized_height = max(1, round(height * scale))
            row, column = divmod(index, GRID)
            x = column * CELL_SIZE + (CELL_SIZE - resized_width) // 2
            y = row * CELL_SIZE + (CELL_SIZE - resized_height) // 2
            rects.append(
                (float(x), float(y), float(x + resized_width), float(y + resized_height))
            )
        output[int(image["id"])] = rects
    return output


def _distance_to_lines(value: float, lines: list[int]) -> float:
    return min((abs(value - line) for line in lines), default=math.inf)


def _center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _features(
    image_id: int,
    box: list[float],
    content_rects: dict[int, list[tuple[float, float, float, float]]],
    tile_lines: list[int],
) -> dict[str, Any]:
    center_x, center_y = _center(box)
    cell_lines = list(range(CELL_SIZE, CANVAS_SIZE, CELL_SIZE))
    cell_distance = min(
        _distance_to_lines(center_x, cell_lines), _distance_to_lines(center_y, cell_lines)
    )
    tile_distance = min(
        _distance_to_lines(center_x, tile_lines), _distance_to_lines(center_y, tile_lines)
    )
    column = min(GRID - 1, max(0, int(center_x // CELL_SIZE)))
    row = min(GRID - 1, max(0, int(center_y // CELL_SIZE)))
    rect = content_rects[image_id][row * GRID + column]
    in_flat_padding = not (
        rect[0] <= center_x < rect[2] and rect[1] <= center_y < rect[3]
    )
    return {
        "cell_seam_distance": cell_distance,
        "tile_line_distance": tile_distance,
        "in_flat_padding": in_flat_padding,
    }


def _area_exposure(margin: float) -> float:
    interior = max(0.0, CELL_SIZE - 2.0 * margin)
    return 1.0 - (interior / CELL_SIZE) ** 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    document = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    gt = load_coco_ground_truth(args.ground_truth)
    predictions = load_coco_predictions(args.predictions)
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    pooled, trace = evaluate_predictions_with_trace(
        gt,
        predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    content_rects = _source_content_rects(document, args.source_root)
    tiles = generate_tiles(CANVAS_SIZE, CANVAS_SIZE, 1024, 256)
    tile_lines = sorted(
        {
            value
            for tile in tiles
            for value in (
                tile.x_offset,
                tile.x_offset + tile.width,
                tile.y_offset,
                tile.y_offset + tile.height,
            )
            if 0 < value < CANVAS_SIZE
        }
    )
    matched_predictions = {
        (event.image_id, event.prediction_index) for event in trace.matches
    }
    matched_gt = {(event.image_id, event.ground_truth_index) for event in trace.matches}
    by_status: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for image_id, rows in predictions.items():
        for index, row in enumerate(rows):
            status = "tp" if (image_id, index) in matched_predictions else "fp"
            by_status[status].append(
                _features(image_id, row["bbox_xyxy"], content_rects, tile_lines)
            )
    for image_id, rows in gt.items():
        for index, row in enumerate(rows):
            status = "matched_gt" if (image_id, index) in matched_gt else "fn"
            by_status[status].append(
                _features(image_id, row["bbox_xyxy"], content_rects, tile_lines)
            )

    margins: dict[str, Any] = {}
    for margin in (8.0, 16.0, 32.0, 64.0):
        exposure = _area_exposure(margin)
        row: dict[str, Any] = {"canvas_area_exposure_fraction": exposure}
        near_counts: dict[str, int] = {}
        for status in ("tp", "fp", "matched_gt", "fn"):
            values = by_status[status]
            near = sum(item["cell_seam_distance"] <= margin for item in values)
            near_counts[status] = near
            fraction = near / len(values) if values else None
            row[status] = {
                "count": len(values),
                "near_artificial_seam": near,
                "near_fraction": fraction,
                "area_normalized_enrichment": (
                    fraction / exposure if fraction is not None and exposure > 0 else None
                ),
            }
        near_gt = near_counts["matched_gt"] + near_counts["fn"]
        far_matched_gt = len(by_status["matched_gt"]) - near_counts["matched_gt"]
        far_fn = len(by_status["fn"]) - near_counts["fn"]
        far_gt = far_matched_gt + far_fn
        near_predictions = near_counts["tp"] + near_counts["fp"]
        far_tp = len(by_status["tp"]) - near_counts["tp"]
        far_fp = len(by_status["fp"]) - near_counts["fp"]
        far_predictions = far_tp + far_fp
        row["conditional_error_rates"] = {
            "near_seam_miss_rate": near_counts["fn"] / near_gt if near_gt else None,
            "away_from_seam_miss_rate": far_fn / far_gt if far_gt else None,
            "near_seam_fdr": near_counts["fp"] / near_predictions if near_predictions else None,
            "away_from_seam_fdr": far_fp / far_predictions if far_predictions else None,
        }
        margins[str(int(margin))] = row

    padding = {
        status: {
            "count": len(values),
            "in_flat_padding": sum(item["in_flat_padding"] for item in values),
            "fraction": (
                sum(item["in_flat_padding"] for item in values) / len(values)
                if values
                else None
            ),
        }
        for status, values in sorted(by_status.items())
    }
    result = {
        "status": "complete",
        "role": "pseudo10k_artifact_error_diagnostic_not_hidden_distribution_estimate",
        "input_sha256": {
            "ground_truth": _sha256(args.ground_truth),
            "predictions": _sha256(args.predictions),
        },
        "pooled": {
            "recall": pooled.recall,
            "fdr": pooled.fdr,
            **pooled.details,
        },
        "artificial_cell_seam": margins,
        "flat_padding": padding,
        "interpretation": [
            "Artificial cell seams and flat letterbox bands do not exist in a continuous official scene.",
            "FP enrichment near those regions measures proxy contamination, not hidden-set model quality.",
            "The full P40 checkpoint has seen these source images during training, so absolute Recall/FDR is not a generalisation estimate.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
