#!/usr/bin/env python3
"""Validate and freeze one completed CV3 OOF fold delivery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rsdet.experiments.cv3_oof import finalize_fold_delivery


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="冻结单折 OOF 小型交付元数据")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--train-summary", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--data-lock-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        metadata = finalize_fold_delivery(
            plan_path=args.plan,
            held_out_fold=args.fold,
            train_config_path=args.train_config,
            train_summary_path=args.train_summary,
            infer_config_path=args.infer_config,
            environment_path=args.environment,
            checkpoint_path=args.checkpoint,
            predictions_path=args.predictions,
            runtime_path=args.runtime,
            data_lock_verification_path=args.data_lock_verification,
            output_path=args.output,
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"CV3_OOF_FOLD_FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "CV3_OOF_FOLD_PASS "
        f"model={metadata['model_key']} fold={metadata['held_out_fold']} "
        f"proposals={metadata['inference']['proposal_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
