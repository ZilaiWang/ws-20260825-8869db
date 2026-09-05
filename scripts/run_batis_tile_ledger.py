#!/usr/bin/env python3
"""Export one P40 tile ledger and replay BATIS H0/H1A/H1B/H2 without reinference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from PIL import Image

from rsdet.contracts import InferenceSample, Prediction, TileRecord
from rsdet.engine.predictor import predict_batches
from rsdet.pipeline.large_image import _extract_tile_image
from rsdet.postprocess.safe_tile_fusion import SafeFusionAudit, fuse_safe_tile_predictions
from rsdet.submission.competition import CompetitionDetector, load_submission_config
from rsdet.tiling.slicer import generate_tiles


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_image(root: Path, image: dict[str, Any]) -> Path:
    file_name = str(image["file_name"])
    candidates = []
    if "fold" in image:
        candidates.append(root / f"fold_{int(image['fold'])}" / "images" / file_name)
    candidates.extend((root / "images" / file_name, root / file_name))
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(f"expected one image for {file_name}, found={existing}")
    return existing[0]


def _prediction_to_coco(prediction: Prediction) -> list[dict[str, Any]]:
    rows = []
    for box, score, label in zip(
        prediction.boxes_xyxy,
        prediction.scores,
        prediction.labels,
        strict=True,
    ):
        x1, y1, x2, y2 = (float(value) for value in box)
        rows.append(
            {
                "image_id": int(prediction.image_id),
                "category_id": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            }
        )
    return rows


def _filter_score(prediction: Prediction, threshold: float) -> Prediction:
    keep = [index for index, score in enumerate(prediction.scores) if float(score) >= threshold]
    return Prediction(
        prediction.image_id,
        [prediction.boxes_xyxy[index] for index in keep],
        [prediction.scores[index] for index in keep],
        [prediction.labels[index] for index in keep],
    )


def _sum_audits(audits: list[SafeFusionAudit]) -> dict[str, int]:
    if not audits:
        return {}
    keys = tuple(audits[0].to_dict())
    return {key: sum(audit.to_dict()[key] for audit in audits) for key in keys}


def _fuse(
    tile_predictions: list[Prediction],
    tiles: list[TileRecord],
    *,
    image_id: int,
    width: int,
    height: int,
    pipeline: Any,
    candidate_floor: float,
    output_threshold: float | None,
    owner_logit_slack: float | None,
    score_threshold_by_fine: dict[int, float] | None = None,
    threshold_safe_category_ids: tuple[int, ...] | None = None,
    audit_score_threshold: float | None = None,
) -> tuple[Prediction, SafeFusionAudit]:
    result = fuse_safe_tile_predictions(
        tile_predictions,
        tiles,
        parent_image_id=image_id,
        image_width=width,
        image_height=height,
        score_threshold=candidate_floor,
        score_threshold_by_fine=score_threshold_by_fine,
        merge_iou=pipeline.merge_iou,
        merge_ios=pipeline.merge_ios,
        fine_nms_iou=pipeline.fine_nms_iou,
        border_margin=pipeline.border_margin,
        max_detections=pipeline.max_detections,
        output_score_threshold=output_threshold,
        owner_logit_slack=owner_logit_slack,
        threshold_safe_category_ids=threshold_safe_category_ids,
        audit_score_threshold=audit_score_threshold,
        return_audit=True,
    )
    if not isinstance(result, tuple):
        raise AssertionError("return_audit=True must return a tuple")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-floor", type=float, default=0.001)
    parser.add_argument("--output-threshold", type=float, default=0.536)
    parser.add_argument("--owner-logit-slack", type=float, default=0.20)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not 0.0 <= args.candidate_floor <= args.output_threshold <= 1.0:
        raise ValueError("require candidate_floor <= output_threshold")

    document = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    images = list(document["images"])
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        images = images[: args.limit]
    config = load_submission_config(args.config)
    if config.get("aircraft_classifier_model") is not None:
        raise ValueError(
            "ledger qualification must use detector-only config; D4 stays outside BATIS"
        )
    if (
        config.get("resolution_expert_model") is not None
        or config.get("agreement_model") is not None
    ):
        raise ValueError("ledger qualification requires a single P40 detector")
    config["device"] = args.device
    runtime = CompetitionDetector(config)
    pipeline = runtime.pipeline_config
    if pipeline.fusion != "safe":
        raise ValueError("BATIS replay requires fusion=safe")
    if not math.isclose(pipeline.score_threshold, args.candidate_floor, abs_tol=1e-12):
        raise ValueError("config score_threshold must equal the frozen candidate floor")

    output_predictions: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("h0_legacy", "h1a_prefilter", "h1b_threshold_safe", "h2_owner")
    }
    output_audits: dict[str, list[SafeFusionAudit]] = {name: [] for name in output_predictions}
    ledger_images = []
    durations = []
    max_det = int(config["model"]["max_detections"])
    saturated_tiles = 0
    started = time.time()
    batis_category_ids = (0, 1, 2, 3, 24)
    Image.MAX_IMAGE_PIXELS = None
    for offset, image_row in enumerate(images):
        image_id = int(image_row["id"])
        path = _resolve_image(args.image_root, image_row)
        before = time.perf_counter()
        with Image.open(path) as opened:
            rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
        height, width = rgb.shape[:2]
        tiles = generate_tiles(width, height, pipeline.tile_size, pipeline.overlap)
        for tile in tiles:
            tile.parent_image_id = image_id
        samples = [
            InferenceSample(
                tile.tile_id,
                _extract_tile_image(rgb, tile),
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
        saturated_tiles += sum(len(prediction.scores) >= max_det for prediction in tile_predictions)

        h1a_fine_thresholds = {
            category_id: (
                args.output_threshold if category_id in batis_category_ids else args.candidate_floor
            )
            for category_id in range(25)
        }
        variants = {
            "h0_legacy": (args.candidate_floor, None, None, None, None),
            "h1a_prefilter": (
                args.candidate_floor,
                None,
                None,
                h1a_fine_thresholds,
                None,
            ),
            "h1b_threshold_safe": (
                args.candidate_floor,
                args.output_threshold,
                None,
                None,
                batis_category_ids,
            ),
            "h2_owner": (
                args.candidate_floor,
                args.output_threshold,
                args.owner_logit_slack,
                None,
                batis_category_ids,
            ),
        }
        for name, (
            floor,
            output_threshold,
            owner_slack,
            fine_thresholds,
            safe_category_ids,
        ) in variants.items():
            prediction, audit = _fuse(
                tile_predictions,
                tiles,
                image_id=image_id,
                width=width,
                height=height,
                pipeline=pipeline,
                candidate_floor=floor,
                output_threshold=output_threshold,
                owner_logit_slack=owner_slack,
                score_threshold_by_fine=fine_thresholds,
                threshold_safe_category_ids=safe_category_ids,
                audit_score_threshold=args.output_threshold,
            )
            prediction = _filter_score(prediction, args.output_threshold)
            output_predictions[name].extend(_prediction_to_coco(prediction))
            output_audits[name].append(audit)

        ledger_images.append(
            {
                "image_id": image_id,
                "file_name": str(image_row["file_name"]),
                "fold": image_row.get("fold"),
                "width": width,
                "height": height,
                "tiles": [
                    {
                        "tile_id": int(tile.tile_id),
                        "x_offset": int(tile.x_offset),
                        "y_offset": int(tile.y_offset),
                        "width": int(tile.width),
                        "height": int(tile.height),
                        "detections": [
                            {
                                "bbox_xyxy_local": [float(value) for value in box],
                                "score": float(score),
                                "label": int(label),
                            }
                            for box, score, label in zip(
                                prediction.boxes_xyxy,
                                prediction.scores,
                                prediction.labels,
                                strict=True,
                            )
                        ],
                    }
                    for tile, prediction in zip(tiles, tile_predictions, strict=True)
                ],
            }
        )
        durations.append(time.perf_counter() - before)
        print(
            f"images={offset + 1}/{len(images)} tiles={len(tiles)} "
            f"mean_seconds={fmean(durations):.4f}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True)
    ledger_path = args.output_dir / "tile_ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "schema_version": "hera_guard_batis_tile_ledger_v1",
                "config": str(args.config),
                "candidate_floor": args.candidate_floor,
                "images": ledger_images,
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    prediction_sha = {}
    for name, rows in output_predictions.items():
        path = args.output_dir / f"{name}_predictions.json"
        path.write_text(json.dumps(rows, separators=(",", ":")) + "\n", encoding="utf-8")
        prediction_sha[name] = _sha256(path)
    ordered_durations = sorted(durations)
    summary = {
        "status": "complete",
        "role": "batis_detector_only_mechanism_qualification",
        "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "ground_truth": str(args.ground_truth),
        "ground_truth_sha256": _sha256(args.ground_truth),
        "images": len(images),
        "tiles": sum(len(row["tiles"]) for row in ledger_images),
        "max_det_saturated_tiles": saturated_tiles,
        "candidate_floor": args.candidate_floor,
        "output_threshold": args.output_threshold,
        "owner_logit_slack": args.owner_logit_slack,
        "batis_category_ids": list(batis_category_ids),
        "aircraft_fusion_frozen_legacy": True,
        "variant_prediction_count": {name: len(rows) for name, rows in output_predictions.items()},
        "variant_prediction_sha256": prediction_sha,
        "variant_audits": {name: _sum_audits(audits) for name, audits in output_audits.items()},
        "h1a_h1b_exact_prediction_parity": (
            prediction_sha["h1a_prefilter"] == prediction_sha["h1b_threshold_safe"]
        ),
        "mean_image_seconds": fmean(durations),
        "p50_image_seconds": ordered_durations[len(ordered_durations) // 2],
        "p95_image_seconds": ordered_durations[
            min(len(ordered_durations) - 1, int(0.95 * len(ordered_durations)))
        ],
        "wall_seconds": time.time() - started,
        "ledger_sha256": _sha256(ledger_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
