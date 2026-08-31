#!/usr/bin/env python3
"""Run fold-heldout YOLO or RT-DETR checkpoints on pseudo-10K mosaics.

This evaluator deliberately bypasses the competition Docker entry point because
that entry point is frozen to the admitted single YOLO deployment.  It reuses
the same tiling and safe-fusion implementation, while allowing the formal M3
RT-DETR checkpoints to be evaluated under the identical pseudo-10K geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.models.ultralytics_adapter import UltralyticsDetector
from rsdet.pipeline.large_image import PipelineConfig, run_pipeline


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_pipeline_config(*, batch_size: int, score_floor: float) -> PipelineConfig:
    """Return the frozen pseudo-10K tiling contract shared by Y5 and M3."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= score_floor <= 1.0:
        raise ValueError("score_floor must be in [0, 1]")
    return PipelineConfig(
        tile_size=1024,
        overlap=256,
        batch_size=batch_size,
        score_threshold=score_floor,
        fine_nms_iou=0.70,
        coarse_nms_iou=0.85,
        max_detections=4000,
        fusion="safe",
        merge_iou=0.50,
        merge_ios=0.75,
        border_margin=8.0,
    )


def prediction_to_coco(
    prediction: object, *, image_id: int, source_fold: int
) -> list[dict[str, object]]:
    """Convert the project xyxy prediction contract to COCO xywh rows."""
    rows: list[dict[str, object]] = []
    for box, score, label in zip(
        prediction.boxes_xyxy, prediction.scores, prediction.labels, strict=True
    ):
        x1, y1, x2, y2 = (float(value) for value in box)
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append(
            {
                "image_id": int(image_id),
                "category_id": int(label),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
                "source_fold": int(source_fold),
            }
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo-root", type=Path, required=True)
    parser.add_argument("--family", choices=("yolo", "rtdetr"), required=True)
    parser.add_argument("--weights", type=Path, nargs=3, required=True)
    parser.add_argument(
        "--agreement-adapters",
        type=Path,
        nargs=3,
        help="optional fold-aligned in-model agreement adapters",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        choices=(0, 1, 2),
        default=(0, 1, 2),
        help="evaluate only selected folds; default is all three",
    )
    parser.add_argument("--score-floor", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--coarse-label-space",
        action="store_true",
        help="map detector labels ship/aircraft/vehicle from 0/1/2 to 0/4/24 placeholders",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_gt = json.loads(
        (args.pseudo_root / "ground_truth.json").read_text(encoding="utf-8")
    )
    image_id_by_name = {
        str(item["file_name"]): int(item["id"]) for item in combined_gt["images"]
    }
    pipeline = build_pipeline_config(
        batch_size=args.batch_size, score_floor=args.score_floor
    )

    all_predictions: list[dict[str, object]] = []
    fold_summaries: list[dict[str, object]] = []
    selected_folds = tuple(sorted(set(args.folds)))
    for fold in selected_folds:
        weight = args.weights[fold]
        if not weight.is_file():
            raise FileNotFoundError(weight)
        input_dir = args.pseudo_root / f"fold_{fold}" / "images"
        image_paths = sorted(input_dir.glob("*.jpg"))
        expected_names = {
            str(item["file_name"])
            for item in combined_gt["images"]
            if str(item["file_name"]).startswith(f"fold{fold}_")
        }
        if {path.name for path in image_paths} != expected_names:
            raise RuntimeError(f"fold {fold} pseudo image inventory mismatch")

        agreement = None
        if args.agreement_adapters is not None and str(args.agreement_adapters[fold]) != "-":
            adapter = args.agreement_adapters[fold]
            if not adapter.is_file():
                raise FileNotFoundError(adapter)
            agreement = {
                "checkpoint": str(adapter),
                "expected_sha256": _sha256(adapter),
                "category_id": 24,
            }
        detector = UltralyticsDetector(
            family=args.family,
            imgsz=1024,
            confidence=args.score_floor,
            iou=0.70,
            max_detections=300 if args.family == "rtdetr" else 500,
            half=True,
            agnostic_nms=False,
            label_map={0: 0, 1: 4, 2: 24} if args.coarse_label_space else None,
            agreement=agreement,
        )
        detector.load(str(weight))
        detector.to(args.device)
        detector.eval()
        print(
            f"[pseudo-eval] fold={fold} family={args.family} "
            f"images={len(image_paths)} weight_sha256={_sha256(weight)}",
            flush=True,
        )

        started = time.perf_counter()
        fold_predictions: list[dict[str, object]] = []
        image_timings: list[dict[str, object]] = []
        for path in image_paths:
            with Image.open(path) as handle:
                rgb = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
            image_started = time.perf_counter()
            prediction, timing = run_pipeline(
                rgb,
                detector,
                config=pipeline,
                parent_image_id=image_id_by_name[path.name],
            )
            elapsed = time.perf_counter() - image_started
            fold_predictions.extend(
                prediction_to_coco(
                    prediction,
                    image_id=image_id_by_name[path.name],
                    source_fold=fold,
                )
            )
            image_timings.append(
                {
                    "file_name": path.name,
                    "wall_seconds": elapsed,
                    **timing.to_dict(),
                }
            )
            print(
                f"[pseudo-eval] fold={fold} image={path.name} "
                f"predictions={len(prediction.scores)} wall_seconds={elapsed:.3f}",
                flush=True,
            )

        run_dir = args.output_dir / f"fold_{fold}"
        run_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = run_dir / "predictions.json"
        prediction_path.write_text(
            json.dumps(fold_predictions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        all_predictions.extend(fold_predictions)
        fold_summaries.append(
            {
                "fold": fold,
                "weight": str(weight),
                "weight_sha256": _sha256(weight),
                "agreement_adapter": (
                    str(args.agreement_adapters[fold])
                    if args.agreement_adapters is not None
                    and str(args.agreement_adapters[fold]) != "-"
                    else None
                ),
                "agreement_adapter_sha256": (
                    _sha256(args.agreement_adapters[fold])
                    if args.agreement_adapters is not None
                    and str(args.agreement_adapters[fold]) != "-"
                    else None
                ),
                "images": len(image_paths),
                "predictions": len(fold_predictions),
                "wall_seconds": time.perf_counter() - started,
                "image_timings": image_timings,
            }
        )
        print(
            f"[pseudo-eval] fold={fold} complete "
            f"predictions={len(fold_predictions)}",
            flush=True,
        )
        del detector

    combined_path = args.output_dir / "predictions.json"
    combined_path.write_text(
        json.dumps(all_predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": "cv3_oof_pseudo_inference_complete",
        "protocol": "fold_heldout_multifamily_safe1024_pseudo10k_v1",
        "family": args.family,
        "score_floor": args.score_floor,
        "coarse_label_space": bool(args.coarse_label_space),
        "pipeline": vars(pipeline),
        "selected_folds": selected_folds,
        "predictions": len(all_predictions),
        "predictions_sha256": _sha256(combined_path),
        "folds": fold_summaries,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
