#!/usr/bin/env python3
"""Append open-set probabilities for a selective candidate route only."""

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
        len(index) == 0
        or len(np.unique(index)) != len(index)
        or index.min() < 0
        or index.max() >= n
        or probabilities.shape != (len(index), 3)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
        or not np.array_equal(score_folds, arrays["fold"][index].astype(np.int64))
        or not np.array_equal(image_id, arrays["image_id"][index].astype(np.int64))
        or not np.array_equal(category_id, arrays["category_id"][index].astype(np.int64))
        or not np.allclose(
            bbox_xyxy, arrays["bbox_xyxy"][index].astype(np.float32), atol=1e-3
        )
    ):
        raise ValueError("selective open-set scores are not aligned with the OMQ cache")
    selective = np.zeros((n, 5), dtype=np.float16)
    selective[index, 0] = 1.0
    selective[index, 1:4] = probabilities.astype(np.float16)
    selective[index, 4] = (
        probabilities[:, 0] - np.maximum(probabilities[:, 1], probabilities[:, 2])
    ).astype(np.float16)
    base_dimension = int(arrays["features"].shape[1])
    arrays["features"] = np.concatenate(
        (arrays["features"].astype(np.float16), selective), axis=1
    )
    if not np.isfinite(arrays["features"]).all():
        raise RuntimeError("augmented OMQ cache contains NaN/Inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    summary = {
        "status": "complete",
        "protocol": "v5_crop_omq_plus_selective_three_way_open_set_v1",
        "rows": n,
        "selected_rows": int(len(index)),
        "base_dimension": base_dimension,
        "output_dimension": int(arrays["features"].shape[1]),
        "added_features": [
            "selective_route_active",
            "selective_foreground_probability",
            "selective_structured_background_probability",
            "selective_ordinary_background_probability",
            "selective_foreground_margin",
        ],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
