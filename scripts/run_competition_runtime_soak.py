#!/usr/bin/env python3
"""Shard a frozen competition runtime over real images for engineering QA.

This does not select thresholds or estimate a hidden-set score.  It verifies
that one already-frozen runtime loads once, finishes every assigned image, and
emits finite contract-valid objects.  Deterministic hashes make repeated runs
and multi-GPU shard merges auditable without retaining bulky prediction files.
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

from rsdet.submission.competition import (
    CompetitionDetector,
    discover_images,
    load_submission_config,
)


def _canonical_object(row: dict[str, Any]) -> list[Any]:
    return [
        int(row["category_id"]),
        str(row["category_name"]),
        round(float(row["score"]), 10),
        *[round(float(value), 6) for value in row["bbox"]],
    ]


def _validate_objects(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if not 0 <= int(row["category_id"]) < 25:
            raise ValueError(f"invalid category_id: {row}")
        score = float(row["score"])
        box = [float(value) for value in row["bbox"]]
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid score: {row}")
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError(f"invalid bbox: {row}")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"degenerate bbox: {row}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    args.output.mkdir(parents=True)

    config = load_submission_config(args.config)
    config["device"] = args.device
    images = discover_images(args.image_root)
    assigned = [
        path for index, path in enumerate(images) if index % args.shard_count == args.shard_index
    ]
    if not assigned:
        raise ValueError("empty shard")

    detector = CompetitionDetector(config)
    digest = hashlib.sha256()
    durations: list[float] = []
    object_count = 0
    per_fine: Counter[int] = Counter()
    started = time.time()
    for index, path in enumerate(assigned):
        before = time.perf_counter()
        with Image.open(path) as image:
            rows = detector.predict(image.convert("RGB"))
        durations.append(time.perf_counter() - before)
        _validate_objects(rows)
        object_count += len(rows)
        per_fine.update(int(row["category_id"]) for row in rows)
        canonical = [path.name, [_canonical_object(row) for row in rows]]
        digest.update(json.dumps(canonical, separators=(",", ":")).encode())
        if (index + 1) % 100 == 0:
            print(
                f"shard={args.shard_index} images={index + 1}/{len(assigned)} "
                f"mean_sec={fmean(durations):.4f}",
                flush=True,
            )

    ordered = sorted(durations)
    summary = {
        "status": "pass",
        "role": "engineering_runtime_soak_not_score_estimation",
        "config": str(args.config),
        "image_root": str(args.image_root),
        "device": args.device,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "image_count": len(assigned),
        "object_count": object_count,
        "per_fine_object_count": dict(sorted(per_fine.items())),
        "prediction_digest_sha256": digest.hexdigest(),
        "mean_image_seconds": fmean(durations),
        "p50_image_seconds": ordered[len(ordered) // 2],
        "p95_image_seconds": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "wall_seconds": time.time() - started,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
