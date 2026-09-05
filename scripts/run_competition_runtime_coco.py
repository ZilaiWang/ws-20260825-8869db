#!/usr/bin/env python3
"""Run one exact competition runtime on a labelled pseudo set.

This is a diagnostic bridge between the Docker code path and the fixed proxy:
all branch thresholds, class ownership and Aircraft-D4 refinement are applied
by ``CompetitionDetector`` itself.  The emitted COCO list can then be evaluated
at threshold zero because the runtime has already applied its frozen workpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

from PIL import Image

from rsdet.submission.competition import CompetitionDetector, load_submission_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_image(root: Path, image: dict[str, Any]) -> Path:
    file_name = str(image["file_name"])
    candidates = []
    if "fold" in image:
        candidates.append(root / f"fold_{int(image['fold'])}" / "images" / file_name)
    candidates.extend((root / "images" / file_name, root / file_name))
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"expected one image for id={image.get('id')}, found={existing}, tried={candidates}"
        )
    return existing[0]


def _validate_runtime_object(row: dict[str, Any]) -> None:
    score = float(row["score"])
    box = [float(value) for value in row["bbox"]]
    if not 0 <= int(row["category_id"]) < 25:
        raise ValueError(f"invalid category: {row}")
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"invalid score: {row}")
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        raise ValueError(f"invalid box: {row}")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"degenerate xyxy box: {row}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pseudo-root", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--runtime-factory",
        choices=("competition", "sprint20"),
        default="competition",
    )
    args = parser.parse_args()
    for output in (args.predictions, args.summary):
        if output.exists():
            raise FileExistsError(output)
    gt_path = args.ground_truth or args.pseudo_root / "ground_truth.json"
    document = json.loads(gt_path.read_text(encoding="utf-8"))
    images = document.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("ground truth must contain a non-empty images list")
    ids = [int(image["id"]) for image in images]
    if len(ids) != len(set(ids)):
        raise ValueError("ground truth image ids are not unique")

    config = load_submission_config(args.config)
    config["device"] = args.device
    if args.runtime_factory == "sprint20":
        from sprint20.runtime import detector_factory

        detector = detector_factory(config)
    else:
        detector = CompetitionDetector(config)
    predictions: list[dict[str, Any]] = []
    durations: list[float] = []
    per_coarse: Counter[str] = Counter()
    started = time.time()
    Image.MAX_IMAGE_PIXELS = None
    for offset, image_row in enumerate(images):
        path = resolve_image(args.pseudo_root, image_row)
        before = time.perf_counter()
        with Image.open(path) as image:
            objects = detector.predict(image.convert("RGB"))
        durations.append(time.perf_counter() - before)
        for row in objects:
            _validate_runtime_object(row)
            x1, y1, x2, y2 = (float(value) for value in row["bbox"])
            category = int(row["category_id"])
            predictions.append(
                {
                    "image_id": int(image_row["id"]),
                    "category_id": category,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(row["score"]),
                }
            )
            coarse = "ship" if category < 4 else "aircraft" if category < 24 else "vehicle"
            per_coarse[coarse] += 1
        if (offset + 1) % 50 == 0:
            print(
                f"images={offset + 1}/{len(images)} mean_sec={fmean(durations):.4f}",
                flush=True,
            )
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    args.predictions.write_text(
        json.dumps(predictions, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    ordered = sorted(durations)
    summary = {
        "status": "complete",
        "role": "exact_runtime_fixed_proxy_diagnostic_not_hidden_score_prediction",
        "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "ground_truth": str(gt_path),
        "ground_truth_sha256": _sha256(gt_path),
        "images": len(images),
        "predictions": len(predictions),
        "per_coarse_prediction_count": dict(per_coarse),
        "device": args.device,
        "runtime_factory": args.runtime_factory,
        "mean_image_seconds": fmean(durations),
        "p50_image_seconds": ordered[len(ordered) // 2],
        "p95_image_seconds": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "wall_seconds": time.time() - started,
        "predictions_sha256": _sha256(args.predictions),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
