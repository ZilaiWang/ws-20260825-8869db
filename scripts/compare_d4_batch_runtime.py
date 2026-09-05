#!/usr/bin/env python3
"""Verify D4 batch-size output parity and report paired proxy latency.

The historical command compared batch 32 with batch 64.  Keep those option
names as aliases, but allow a later, explicitly labelled equivalent-batch
check without duplicating the audit implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-predictions", "--batch32-predictions", dest="reference_predictions",
        type=Path, required=True,
    )
    parser.add_argument(
        "--candidate-predictions", "--batch64-predictions", dest="candidate_predictions",
        type=Path, required=True,
    )
    parser.add_argument(
        "--reference-summary", "--batch32-summary", dest="reference_summary",
        type=Path, required=True,
    )
    parser.add_argument(
        "--candidate-summary", "--batch64-summary", dest="candidate_summary",
        type=Path, required=True,
    )
    parser.add_argument("--reference-batch", type=int, default=32)
    parser.add_argument("--candidate-batch", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.reference_batch <= 0 or args.candidate_batch <= 0:
        raise ValueError("batch labels must be positive")
    if args.reference_batch == args.candidate_batch:
        raise ValueError("reference and candidate batch labels must differ")
    reference_predictions = _load(args.reference_predictions)
    candidate_predictions = _load(args.candidate_predictions)
    reference_summary = _load(args.reference_summary)
    candidate_summary = _load(args.candidate_summary)
    exact = reference_predictions == candidate_predictions
    if not exact:
        raise AssertionError(
            f"D4 batch={args.candidate_batch} changed the emitted predictions"
        )

    reference_time = float(reference_summary["mean_image_seconds"])
    candidate_time = float(candidate_summary["mean_image_seconds"])
    payload = {
        "status": "complete",
        "protocol": (
            f"d4_batch{args.reference_batch}_vs_batch{args.candidate_batch}_"
            "exact_output_paired_proxy_v1"
        ),
        "predictions_exact": exact,
        "images": int(reference_summary["images"]),
        "predictions": int(reference_summary["predictions"]),
        "reference_batch": args.reference_batch,
        "candidate_batch": args.candidate_batch,
        "reference_mean_image_seconds": reference_time,
        "candidate_mean_image_seconds": candidate_time,
        "candidate_minus_reference_seconds": candidate_time - reference_time,
        "candidate_speedup": reference_time / candidate_time,
        "timing_is_proxy_diagnostic_not_official_measurement": True,
    }
    # Preserve the historical batch32/batch64 field contract while exposing
    # generic names for later equivalent-batch audits.
    payload[f"batch{args.reference_batch}_mean_image_seconds"] = reference_time
    payload[f"batch{args.candidate_batch}_mean_image_seconds"] = candidate_time
    payload[
        f"batch{args.candidate_batch}_minus_batch{args.reference_batch}_seconds"
    ] = candidate_time - reference_time
    payload[f"batch{args.candidate_batch}_speedup"] = reference_time / candidate_time
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
