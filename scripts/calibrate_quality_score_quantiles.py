#!/usr/bin/env python3
"""Fit label-free fold/coarse quantile maps for quality-head score stability.

For outer fold ``k``, the map is fitted only on Normal-CV3 rows from the other
two folds.  It maps the quality-head marginal CDF back to the incumbent score
marginal CDF within each coarse class.  GT labels and pressure benchmarks are
never read.  The fixed 257-quantile grid is an implementation constant, not a
searched hyperparameter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

QUANTILE_COUNT = 257


def quantile_map(values: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Apply a monotone quantile map, robust to repeated source knots."""

    unique_x, inverse = np.unique(x.astype(np.float64), return_inverse=True)
    unique_y = np.zeros(len(unique_x), dtype=np.float64)
    counts = np.zeros(len(unique_x), dtype=np.int64)
    np.add.at(unique_y, inverse, y.astype(np.float64))
    np.add.at(counts, inverse, 1)
    unique_y /= np.maximum(counts, 1)
    unique_y = np.maximum.accumulate(unique_y)
    return np.interp(values, unique_x, unique_y).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model-pattern", required=True, help="must contain {fold}")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if "{fold}" not in args.model_pattern:
        raise ValueError("--model-pattern must contain {fold}")

    import torch

    with np.load(args.cache, allow_pickle=False) as payload:
        features = payload["features"].astype(np.float32)
        detector_score = payload["detector_score"].astype(np.float32)
        folds = payload["fold"].astype(np.int64)
        coarse = payload["coarse_id"].astype(np.int64)
        candidate_index = payload["candidate_index"].astype(np.int64)
    if features.shape[1] != 63:
        raise ValueError("this registered calibrator requires base_crop 63-D features")
    quantiles = np.linspace(0.0, 1.0, QUANTILE_COUNT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibrators: dict[str, dict[str, dict[str, list[float]]]] = {}

    for held_out in (0, 1, 2):
        model = torch.jit.load(
            args.model_pattern.format(fold=held_out), map_location="cpu"
        ).eval()
        with torch.inference_mode():
            quality = model(
                torch.from_numpy(features), torch.from_numpy(detector_score)
            )[:, 0].numpy()
        calibrated = quality.copy()
        calibrators[str(held_out)] = {}
        train_fold = folds != held_out
        for coarse_id in (0, 1, 2):
            fit_mask = train_fold & (coarse == coarse_id)
            apply_mask = (folds == held_out) & (coarse == coarse_id)
            if int(fit_mask.sum()) < QUANTILE_COUNT or not apply_mask.any():
                raise ValueError("insufficient rows for fold/coarse quantile calibration")
            x = np.quantile(quality[fit_mask], quantiles).astype(np.float32)
            y = np.quantile(detector_score[fit_mask], quantiles).astype(np.float32)
            calibrated[apply_mask] = quantile_map(quality[apply_mask], x, y)
            calibrators[str(held_out)][str(coarse_id)] = {
                "quality_knots": x.tolist(),
                "detector_knots": y.tolist(),
                "fit_rows": int(fit_mask.sum()),
            }
        held_mask = folds == held_out
        np.savez_compressed(
            args.output_dir / f"quality_fold{held_out}.scores.npz",
            candidate_index=candidate_index[held_mask],
            score=calibrated[held_mask],
        )
    payload = {
        "version": "normal_trainfold_coarse_quantile_calibration_v1",
        "metric_protocol": "platform_observed_20260831",
        "uses_labels": False,
        "quantile_count": QUANTILE_COUNT,
        "feature_contract": "base_crop_63d_v1",
        "calibrators": calibrators,
    }
    (args.output_dir / "calibrators.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "calibrators"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
