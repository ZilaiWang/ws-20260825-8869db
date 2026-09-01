#!/usr/bin/env python3
"""Apply the frozen Ship/Vehicle same-architecture support rule to COCO rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsdet.submission.same_arch_support import rescore_same_fine_support


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--specialist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    specialist = json.loads(args.specialist.read_text(encoding="utf-8"))
    output, audit = rescore_same_fine_support(
        primary,
        specialist,
        label_iou_thresholds={0: 0.50, 1: 0.50, 2: 0.50, 3: 0.50, 24: 0.35},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False) + "\n", encoding="utf-8")
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
