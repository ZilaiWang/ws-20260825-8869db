#!/usr/bin/env python3
"""Generate three read-only split views and a frozen CV3 OOF run plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rsdet.experiments.cv3_oof import prepare_oof_run_plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 M1/M3 正式 CV3 OOF 三折输入")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-key", choices=("M1", "M3"), required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-size", type=int, required=True)
    parser.add_argument("--foundation-epochs", type=int, required=True)
    parser.add_argument("--low-score-threshold", type=float, default=0.001)
    parser.add_argument("--max-detections", type=int, required=True)
    parser.add_argument("--pretrained-weight", required=True)
    parser.add_argument("--pretrained-weight-sha256", required=True)
    parser.add_argument("--detection-data-lock", type=Path, required=True)
    parser.add_argument("--detection-data-lock-sha256", required=True)
    parser.add_argument("--expected-image-count", type=int, default=4481)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = prepare_oof_run_plan(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            model_key=args.model_key,
            model_family=args.model_family,
            model_name=args.model_name,
            seed=args.seed,
            input_size=args.input_size,
            foundation_epochs=args.foundation_epochs,
            low_score_threshold=args.low_score_threshold,
            max_detections=args.max_detections,
            pretrained_weight=args.pretrained_weight,
            pretrained_weight_sha256=args.pretrained_weight_sha256,
            detection_data_lock=str(args.detection_data_lock),
            detection_data_lock_sha256=args.detection_data_lock_sha256,
            expected_manifest_sha256=args.manifest_sha256,
            expected_image_count=args.expected_image_count,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"CV3_OOF_PREPARE_FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "CV3_OOF_PREPARE_PASS "
        f"model={plan['model_key']} images={plan['image_count']} "
        f"folds={plan['fold_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
