#!/usr/bin/env python3
"""Run frozen P40 on production plus shifted padded grids and report a recall oracle."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.contracts import InferenceSample, Prediction, TileRecord
from rsdet.engine.predictor import predict_batches
from rsdet.evaluation.coco import load_coco_ground_truth
from rsdet.evaluation.official_metric import (
    evaluate_predictions_with_trace,
    evaluate_ranking_metrics,
)
from rsdet.evaluation.platform_protocol import build_platform_observed_metrics
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.pipeline.large_image import _extract_tile_image
from rsdet.postprocess.safe_tile_fusion import fuse_safe_tile_predictions
from rsdet.submission.competition import CompetitionDetector, load_submission_config
from rsdet.tiling.boundary_geometry import build_virtual_tiles
from rsdet.tiling.slicer import generate_tiles
from rsdet.utils.config import load_config


def _parse_phases(raw: str) -> tuple[tuple[int, int], ...]:
    phases = tuple(
        tuple(int(value) for value in token.split(":")) for token in raw.split(",") if token
    )
    if any(len(row) != 2 for row in phases) or not phases or phases[0] != (0, 0):
        raise ValueError("phases must start with 0:0 and contain x:y pairs")
    if len(set(phases)) != len(phases):
        raise ValueError("phases must be unique")
    return phases  # type: ignore[return-value]


def _resolve_image(root: Path, image: dict[str, Any]) -> Path:
    name = str(image["file_name"])
    candidates = []
    if "fold" in image:
        candidates.append(root / f"fold_{int(image['fold'])}" / "images" / name)
    candidates.extend((root / "images" / name, root / name))
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"expected one image for {name}, found={found}")
    return found[0]


def _shifted_tiles(
    image_id: int,
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
    phase: tuple[int, int],
) -> list[TileRecord]:
    if phase == (0, 0):
        tiles = generate_tiles(width, height, tile_size, overlap)
        for tile in tiles:
            tile.parent_image_id = image_id
        return tiles
    virtual = build_virtual_tiles(
        width,
        height,
        tile_size,
        overlap,
        phase_x=phase[0],
        phase_y=phase[1],
        padded_phase=True,
    )
    return [
        TileRecord(row.tile_id, image_id, row.x_start, row.y_start, tile_size, tile_size)
        for row in virtual
    ]


def _padded_crop(image: np.ndarray, tile: TileRecord) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1 = tile.x_offset, tile.y_offset
    x2, y2 = x1 + tile.width, y1 + tile.height
    source = image[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
    pads = (
        (max(0, -y1), max(0, y2 - height)),
        (max(0, -x1), max(0, x2 - width)),
        (0, 0),
    )
    mode = "reflect" if source.shape[0] > 1 and source.shape[1] > 1 else "edge"
    result = np.pad(source, pads, mode=mode)
    if result.shape[:2] != (tile.height, tile.width):
        raise AssertionError(f"invalid padded crop shape={result.shape} tile={tile}")
    return result.copy()


def _filter(prediction: Prediction, threshold: float) -> Prediction:
    indices = [i for i, score in enumerate(prediction.scores) if score >= threshold]
    return Prediction(
        prediction.image_id,
        [prediction.boxes_xyxy[i] for i in indices],
        [prediction.scores[i] for i in indices],
        [prediction.labels[i] for i in indices],
    )


def _prediction_rows(prediction: Prediction) -> list[dict[str, Any]]:
    rows = []
    for box, score, label in zip(
        prediction.boxes_xyxy, prediction.scores, prediction.labels, strict=True
    ):
        x1, y1, x2, y2 = box
        rows.append(
            {
                "image_id": prediction.image_id,
                "category_id": label,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": score,
            }
        )
    return rows


def _metrics(
    gt: Any, predictions: Any, protocol: Any, complete: bool
) -> tuple[dict[str, Any], Any]:
    pooled, trace = evaluate_predictions_with_trace(
        gt,
        predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    ranking = evaluate_ranking_metrics(
        gt,
        predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=complete,
    )
    platform = build_platform_observed_metrics(ranking) if complete else None
    return (
        {
            "pooled_recall": pooled.recall,
            "pooled_fdr": pooled.fdr,
            "tp": pooled.details["tp"],
            "fp": pooled.details["fp"],
            "fn": pooled.details["fn"],
            "gate_recall": None if platform is None else platform.gate_recall,
            "gate_fdr": None if platform is None else platform.gate_fdr,
            "per_coarse": {
                name: (
                    None
                    if name not in ranking.per_coarse
                    else {
                        "macro_recall": ranking.per_coarse[name].macro_recall,
                        "macro_fdr": ranking.per_coarse[name].macro_fdr,
                        "pooled_recall": ranking.per_coarse[name].pooled_recall,
                        "pooled_fdr": ranking.per_coarse[name].pooled_fdr,
                    }
                )
                for name in ("ship", "aircraft", "vehicle")
            },
        },
        trace,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--phases", default="0:0,384:0,0:384,384:384")
    parser.add_argument("--output-threshold", type=float, default=0.536)
    parser.add_argument("--allow-incomplete-taxonomy", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    phases = _parse_phases(args.phases)
    document = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    config = load_submission_config(args.config)
    if any(
        config.get(key) is not None
        for key in ("aircraft_classifier_model", "resolution_expert_model", "agreement_model")
    ):
        raise ValueError("phase oracle requires detector-only P40")
    config["device"] = args.device
    runtime = CompetitionDetector(config)
    pipeline = runtime.pipeline_config
    if pipeline.fusion != "safe":
        raise ValueError("phase oracle requires safe fusion")
    predictions_by_phase = {phase: [] for phase in phases}
    durations = Counter()
    tile_counts = Counter()
    Image.MAX_IMAGE_PIXELS = None
    for image_offset, image_row in enumerate(document["images"]):
        image_id = int(image_row["id"])
        with Image.open(_resolve_image(args.image_root, image_row)) as opened:
            image = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
        height, width = image.shape[:2]
        for phase in phases:
            before = time.perf_counter()
            tiles = _shifted_tiles(
                image_id,
                width,
                height,
                pipeline.tile_size,
                pipeline.overlap,
                phase,
            )
            samples = [
                InferenceSample(
                    tile.tile_id,
                    (
                        _extract_tile_image(image, tile)
                        if phase == (0, 0)
                        else _padded_crop(image, tile)
                    ),
                    tile.width,
                    tile.height,
                    {"parent_image_id": image_id},
                )
                for tile in tiles
            ]
            tile_predictions = predict_batches(
                runtime.detector,
                samples,
                batch_size=pipeline.batch_size,
                allowed_category_ids=range(25),
            )
            fused = fuse_safe_tile_predictions(
                tile_predictions,
                tiles,
                parent_image_id=image_id,
                image_width=width,
                image_height=height,
                score_threshold=pipeline.score_threshold,
                merge_iou=pipeline.merge_iou,
                merge_ios=pipeline.merge_ios,
                fine_nms_iou=pipeline.fine_nms_iou,
                border_margin=pipeline.border_margin,
                max_detections=pipeline.max_detections,
            )
            if isinstance(fused, tuple):
                raise AssertionError("unexpected audit tuple")
            predictions_by_phase[phase].extend(
                _prediction_rows(_filter(fused, args.output_threshold))
            )
            tile_counts[str(phase)] += len(tiles)
            durations[str(phase)] += time.perf_counter() - before
        print(f"images={image_offset + 1}/{len(document['images'])}", flush=True)

    args.output_dir.mkdir(parents=True)
    gt = load_coco_ground_truth(args.ground_truth)
    protocol = parse_evaluation_protocol(load_config(Path("configs/project.yaml")))
    complete = not args.allow_incomplete_taxonomy
    metrics = {}
    traces = {}
    for phase, rows in predictions_by_phase.items():
        name = f"phase_{phase[0]}_{phase[1]}"
        (args.output_dir / f"{name}_predictions.json").write_text(
            json.dumps(rows, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        loaded = {image_id: [] for image_id in gt}
        for row in rows:
            x, y, width, height = row["bbox"]
            loaded[int(row["image_id"])].append(
                {
                    "category_id": int(row["category_id"]),
                    "bbox_xyxy": [x, y, x + width, y + height],
                    "score": float(row["score"]),
                }
            )
        metrics[name], traces[name] = _metrics(gt, loaded, protocol, complete)

    gt_counts = Counter(int(row["category_id"]) for image_rows in gt.values() for row in image_rows)
    oracle_matches = {
        (row.image_id, row.category_id, row.ground_truth_index)
        for trace in traces.values()
        for row in trace.matches
    }
    oracle_hits = Counter(category_id for _, category_id, _ in oracle_matches)
    coarse_ids = {"ship": range(0, 4), "aircraft": range(4, 24), "vehicle": range(24, 25)}
    oracle_per_coarse = {}
    for name, ids in coarse_ids.items():
        active = [category_id for category_id in ids if gt_counts[category_id] > 0]
        oracle_per_coarse[name] = (
            None
            if not active
            else fmean(oracle_hits[category_id] / gt_counts[category_id] for category_id in active)
        )
    active_oracle = [value for value in oracle_per_coarse.values() if value is not None]
    phase_zero = metrics["phase_0_0"]
    payload = {
        "status": "complete",
        "role": "batis_phase_recall_oracle_diagnostic_not_deployable",
        "phases": [list(phase) for phase in phases],
        "tile_counts": dict(tile_counts),
        "phase_seconds": dict(durations),
        "metrics": metrics,
        "phase_oracle": {
            "per_coarse_macro_recall": oracle_per_coarse,
            "mean_over_supported_coarse": fmean(active_oracle),
            "delta_vs_production_gate_recall": (
                None
                if phase_zero["gate_recall"] is None
                else fmean(active_oracle) - phase_zero["gate_recall"]
            ),
            "union_matched_gt": len(oracle_matches),
            "not_deployable_and_has_no_fdr": True,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
