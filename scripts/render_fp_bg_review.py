#!/usr/bin/env python3
"""Render a blinded N0-4 FP_BG review package (no automatic labels)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rsdet.analysis.fp_bg_review import (
    load_formal_image_index,
    load_review_cards,
    write_review_outputs,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--audit-csv", type=Path, required=True)
    value.add_argument("--formal-crop-manifest", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--cards-per-sheet", type=int, default=4)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.cards_per_sheet <= 0:
        raise ValueError("cards-per-sheet 必须为正整数")
    formal_index = load_formal_image_index(args.formal_crop_manifest)
    cards = load_review_cards(args.audit_csv, formal_index)
    summary = write_review_outputs(
        cards,
        args.data_root,
        args.output_dir,
        cards_per_sheet=args.cards_per_sheet,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
