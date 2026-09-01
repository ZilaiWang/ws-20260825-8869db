#!/usr/bin/env python3
"""Evaluate COCO predictions on a frozen Background-100MP manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.background_stress import evaluate_background_stress
from rsdet.evaluation.coco import load_coco_predictions
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()
    manifest = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    result = evaluate_background_stress(
        manifest,
        load_coco_predictions(args.predictions),
        category_mapping=protocol.category_mapping,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.__dict__, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
