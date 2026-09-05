#!/usr/bin/env python3
"""Replay BATIS E3: at most K boundary-risk re-centering windows per image."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.contracts import InferenceSample
from rsdet.engine.predictor import predict_batches
from rsdet.pipeline.sparse_recenter import (
    RecenterCandidate,
    boundary_risk_requests,
    cluster_same_fine,
    overlap,
    select_windows,
)
from rsdet.postprocess.nms import nms
from rsdet.submission.competition import CompetitionDetector, load_submission_config
from rsdet.tiling.coordinates import clip_bbox, tile_to_full


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_image(root: Path, image: dict[str, Any]) -> Path:
    name = str(image["file_name"])
    candidates = []
    if image.get("fold") is not None:
        candidates.append(root / f"fold_{int(image['fold'])}" / "images" / name)
    elif name.startswith("fold") and "_" in name:
        prefix = name.split("_", 1)[0]
        if prefix[4:].isdigit():
            candidates.append(root / f"fold_{int(prefix[4:])}" / "images" / name)
    candidates.extend((root / "images" / name, root / name))
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"expected one image for {name}, found={found}")
    return found[0]


def _load_baseline(path: Path) -> dict[int, list[dict[str, Any]]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in json.loads(path.read_text(encoding="utf-8")):
        x, y, width, height = (float(value) for value in row["bbox"])
        by_image[int(row["image_id"])].append(
            {
                "box": [x, y, x + width, y + height],
                "score": float(row["score"]),
                "label": int(row["category_id"]),
            }
        )
    return by_image


def _ledger_candidates(image: dict[str, Any]) -> list[RecenterCandidate]:
    width = int(image["width"])
    height = int(image["height"])
    candidates = []
    for tile in image["tiles"]:
        x_offset = int(tile["x_offset"])
        y_offset = int(tile["y_offset"])
        tile_width = int(tile["width"])
        tile_height = int(tile["height"])
        internal_edges = (
            x_offset > 0,
            y_offset > 0,
            x_offset + tile_width < width,
            y_offset + tile_height < height,
        )
        for row in tile["detections"]:
            label = int(row["label"])
            if label not in {0, 1, 2, 3, 24}:
                continue
            local_box = tuple(float(value) for value in row["bbox_xyxy_local"])
            global_box = tuple(
                float(value)
                for value in clip_bbox(tile_to_full(local_box, x_offset, y_offset), width, height)
            )
            if global_box[2] <= global_box[0] or global_box[3] <= global_box[1]:
                continue
            candidates.append(
                RecenterCandidate(
                    box=global_box,
                    local_box=local_box,
                    score=float(row["score"]),
                    label=label,
                    tile_id=int(tile["tile_id"]),
                    tile_width=tile_width,
                    tile_height=tile_height,
                    internal_edges=internal_edges,
                )
            )
    return candidates


def _crop(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    result = image[y : y + height, x : x + width]
    if result.shape[:2] != (height, width):
        raise AssertionError("recenter window lies outside image")
    return result.copy()


def _center_in_core(box: list[float], width: int, height: int, margin: int) -> bool:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    effective = min(float(margin), width / 4.0, height / 4.0)
    return effective <= cx <= width - effective and effective <= cy <= height - effective


def _fine_nms(rows: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append(row)
    output = []
    for label in sorted(by_label):
        ordered = sorted(by_label[label], key=lambda row: (-float(row["score"]), row["box"]))
        keep = nms(
            [row["box"] for row in ordered],
            [float(row["score"]) for row in ordered],
            iou_threshold,
        )
        output.extend(ordered[index] for index in keep)
    return sorted(output, key=lambda row: (-float(row["score"]), int(row["label"])))


def _to_coco(image_id: int, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        x1, y1, x2, y2 = (float(value) for value in row["box"])
        output.append(
            {
                "image_id": image_id,
                "category_id": int(row["label"]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(row["score"]),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rescue-floor", type=float, default=0.25)
    parser.add_argument("--output-threshold", type=float, default=0.536)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--safe-core-margin", type=int, default=128)
    parser.add_argument("--query-iou", type=float, default=0.25)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not 0.0 <= args.rescue_floor <= args.output_threshold <= 1.0:
        raise ValueError("invalid score band")
    if not 0.0 <= args.query_iou <= 1.0:
        raise ValueError("query_iou must be in [0, 1]")
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    config = load_submission_config(args.config)
    if any(
        config.get(key) is not None
        for key in ("aircraft_classifier_model", "resolution_expert_model", "agreement_model")
    ):
        raise ValueError("E3 mechanism qualification requires detector-only P40")
    config["device"] = args.device
    runtime = CompetitionDetector(config)
    pipeline = runtime.pipeline_config
    baseline = _load_baseline(args.baseline_predictions)
    output_rows = []
    image_audits = []
    durations = []
    reason_counts: Counter[str] = Counter()
    Image.MAX_IMAGE_PIXELS = None
    for offset, image_row in enumerate(ledger["images"]):
        before = time.perf_counter()
        image_id = int(image_row["image_id"])
        width = int(image_row["width"])
        height = int(image_row["height"])
        candidates = _ledger_candidates(image_row)
        clusters = cluster_same_fine(
            candidates, merge_iou=pipeline.merge_iou, merge_ios=pipeline.merge_ios
        )
        requests = boundary_risk_requests(
            clusters,
            rescue_floor=args.rescue_floor,
            output_threshold=args.output_threshold,
            overlap_size=pipeline.overlap,
        )
        windows = select_windows(
            requests,
            image_width=width,
            image_height=height,
            window_size=pipeline.tile_size,
            max_windows=args.max_windows,
        )
        for window in windows:
            reason_counts.update(window.reasons)
        image_meta = {
            "file_name": image_row["file_name"],
            "fold": image_row.get("fold"),
        }
        with Image.open(_resolve_image(args.image_root, image_meta)) as opened:
            image = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
        samples = [
            InferenceSample(
                index,
                _crop(image, row.x_start, row.y_start, row.width, row.height),
                row.width,
                row.height,
                {"parent_image_id": image_id},
            )
            for index, row in enumerate(windows)
        ]
        recentered = (
            []
            if not samples
            else predict_batches(
                runtime.detector,
                samples,
                batch_size=pipeline.batch_size,
                allowed_category_ids=range(25),
            )
        )
        combined = [dict(row) for row in baseline.get(image_id, [])]
        accepted = added = replaced = 0
        rejected = Counter()
        accepted_rows = []
        for window, prediction in zip(windows, recentered, strict=True):
            choices = []
            for box, score, label in zip(
                prediction.boxes_xyxy,
                prediction.scores,
                prediction.labels,
                strict=True,
            ):
                if int(label) != window.label:
                    continue
                if float(score) < args.output_threshold:
                    continue
                local_box = [float(value) for value in box]
                if not _center_in_core(
                    local_box, window.width, window.height, args.safe_core_margin
                ):
                    rejected["outside_safe_core"] += 1
                    continue
                global_box = [
                    local_box[0] + window.x_start,
                    local_box[1] + window.y_start,
                    local_box[2] + window.x_start,
                    local_box[3] + window.y_start,
                ]
                if overlap(global_box, window.query_box)[0] < args.query_iou:
                    rejected["query_iou"] += 1
                    continue
                choices.append({"box": global_box, "score": float(score), "label": int(label)})
            if not choices:
                rejected["no_qualified_detection"] += 1
                continue
            selected = max(choices, key=lambda row: float(row["score"]))
            merge_indices = []
            for index, existing in enumerate(combined):
                if int(existing["label"]) != int(selected["label"]):
                    continue
                iou, ios = overlap(existing["box"], selected["box"])
                if iou >= pipeline.merge_iou or ios >= pipeline.merge_ios:
                    merge_indices.append(index)
            if merge_indices:
                best = max(merge_indices, key=lambda index: float(combined[index]["score"]))
                if float(selected["score"]) <= float(combined[best]["score"]):
                    rejected["not_higher_than_existing"] += 1
                    continue
                combined = [row for index, row in enumerate(combined) if index not in merge_indices]
                replaced += 1
            else:
                added += 1
            combined.append(selected)
            accepted_rows.append(selected)
            accepted += 1
        combined = _fine_nms(combined, pipeline.fine_nms_iou)
        if pipeline.max_detections is not None:
            combined = combined[: pipeline.max_detections]
        output_rows.extend(_to_coco(image_id, combined))
        durations.append(time.perf_counter() - before)
        image_audits.append(
            {
                "image_id": image_id,
                "candidate_count": len(candidates),
                "cluster_count": len(clusters),
                "request_count": len(requests),
                "window_count": len(windows),
                "accepted_count": accepted,
                "added_count": added,
                "replaced_count": replaced,
                "rejected": dict(rejected),
                "windows": [
                    {
                        "box": list(row.box),
                        "query_box": list(row.query_box),
                        "label": row.label,
                        "priority": row.priority,
                        "reasons": list(row.reasons),
                    }
                    for row in windows
                ],
                "accepted": accepted_rows,
            }
        )
        print(
            f"images={offset + 1}/{len(ledger['images'])} windows={len(windows)} "
            f"accepted={accepted}",
            flush=True,
        )
    args.output_dir.mkdir(parents=True)
    prediction_path = args.output_dir / "e3_k8_predictions.json"
    prediction_path.write_text(
        json.dumps(output_rows, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    payload = {
        "status": "complete",
        "role": "batis_e3_sparse_recenter_mechanism_qualification",
        "frozen_parameters": {
            "rescue_floor": args.rescue_floor,
            "output_threshold": args.output_threshold,
            "max_windows": args.max_windows,
            "safe_core_margin": args.safe_core_margin,
            "query_iou": args.query_iou,
            "selected_category_ids": [0, 1, 2, 3, 24],
            "aircraft_untouched": True,
            "replacement_requires_higher_score": True,
        },
        "inputs_sha256": {
            "config": _sha256(args.config),
            "ledger": _sha256(args.ledger),
            "baseline_predictions": _sha256(args.baseline_predictions),
        },
        "images": len(ledger["images"]),
        "requests": sum(row["request_count"] for row in image_audits),
        "windows": sum(row["window_count"] for row in image_audits),
        "accepted": sum(row["accepted_count"] for row in image_audits),
        "added": sum(row["added_count"] for row in image_audits),
        "replaced": sum(row["replaced_count"] for row in image_audits),
        "reason_counts": dict(reason_counts),
        "mean_seconds_per_image": fmean(durations),
        "prediction_sha256": _sha256(prediction_path),
        "image_audits": image_audits,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "image_audits"}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
