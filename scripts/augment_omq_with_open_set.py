#!/usr/bin/env python3
"""Append V5 open-set OOF evidence to an aligned crop-only OMQ cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--open-set-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    with np.load(args.open_set_scores, allow_pickle=False) as scores:
        index = scores["candidate_index"].astype(np.int64)
        probabilities = scores["probabilities"].astype(np.float32)
        score_folds = scores["fold"].astype(np.int64)
        image_id = scores["image_id"].astype(np.int64)
        category_id = scores["category_id"].astype(np.int64)
        bbox_xyxy = scores["bbox_xyxy"].astype(np.float32)
    n = arrays["features"].shape[0]
    if (
        index.tolist() != list(range(n))
        or probabilities.shape != (n, 3)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
        or not np.array_equal(score_folds, arrays["fold"].astype(np.int64))
        or not np.array_equal(image_id, arrays["image_id"].astype(np.int64))
        or not np.array_equal(category_id, arrays["category_id"].astype(np.int64))
        or not np.allclose(bbox_xyxy, arrays["bbox_xyxy"].astype(np.float32), atol=1e-3)
    ):
        raise ValueError("open-set scores are not aligned with the OMQ cache")
    open_features = np.column_stack(
        (
            probabilities,
            probabilities[:, 0] - np.maximum(probabilities[:, 1], probabilities[:, 2]),
        )
    ).astype(np.float16)
    arrays["features"] = np.concatenate(
        (arrays["features"].astype(np.float16), open_features), axis=1
    )
    if not np.isfinite(arrays["features"]).all():
        raise RuntimeError("augmented OMQ cache contains NaN/Inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    summary = {
        "status": "complete",
        "protocol": "v5_crop_omq_plus_three_way_open_set_v1",
        "rows": n,
        "base_dimension": int(arrays["features"].shape[1] - 4),
        "output_dimension": int(arrays["features"].shape[1]),
        "added_features": [
            "open_foreground_probability",
            "open_structured_background_probability",
            "open_ordinary_background_probability",
            "open_foreground_margin",
        ],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
