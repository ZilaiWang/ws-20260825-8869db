#!/usr/bin/env python3
"""Verify optimized D4 output equivalence and report paired proxy latency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--optimized-predictions", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--optimized-summary", type=Path, required=True)
    parser.add_argument("--optimization-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_predictions = _load(args.reference_predictions)
    optimized_predictions = _load(args.optimized_predictions)
    exact = reference_predictions == optimized_predictions
    if not exact:
        raise AssertionError("tensorized/channels-last D4 changed emitted predictions")
    reference = _load(args.reference_summary)
    optimized = _load(args.optimized_summary)
    reference_time = float(reference["mean_image_seconds"])
    optimized_time = float(optimized["mean_image_seconds"])
    payload = {
        "status": "complete",
        "protocol": "d4_runtime_optimization_exact_output_paired_proxy_v1",
        "optimization": args.optimization_label,
        "predictions_exact": exact,
        "images": int(reference["images"]),
        "predictions": int(reference["predictions"]),
        "reference_mean_image_seconds": reference_time,
        "optimized_mean_image_seconds": optimized_time,
        "optimized_minus_reference_seconds": optimized_time - reference_time,
        "optimized_speedup": reference_time / optimized_time,
        "timing_is_proxy_diagnostic_not_official_measurement": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
