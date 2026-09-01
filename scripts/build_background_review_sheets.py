#!/usr/bin/env python3
"""Build deterministic visual-audit sheets for Background-100MP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=96)
    parser.add_argument("--seed", default="background-visual-20260901")
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        help="If supplied, review only candidate keys absent from this earlier manifest.",
    )
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    if args.baseline_manifest is not None:
        baseline_keys = {
            json.loads(line)["candidate_key"]
            for line in args.baseline_manifest.read_text().splitlines()
            if line
        }
        rows = [row for row in rows if row["candidate_key"] not in baseline_keys]
    if not rows:
        raise ValueError("review selection is empty")
    selected = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{args.seed}|{row.get('candidate_key', row['sha256'])}".encode()
        ).hexdigest(),
    )[: args.sample_count]
    args.output.mkdir(parents=True, exist_ok=True)
    review_rows = []
    for sheet_index in range((len(selected) + 15) // 16):
        canvas = Image.new("RGB", (1024, 1024), "white")
        draw = ImageDraw.Draw(canvas)
        for local_index, row in enumerate(selected[sheet_index * 16 : (sheet_index + 1) * 16]):
            x = (local_index % 4) * 256
            y = (local_index // 4) * 256
            path = args.root / row["file_name"]
            with Image.open(path) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((256, 232))
                canvas.paste(thumb, (x, y + 24))
            draw.rectangle((x, y, x + 255, y + 23), fill="black")
            draw.text((x + 4, y + 4), f"id={row['image_id']} src={row['source_image_id']}", fill="white")
            review_rows.append(
                {
                    "image_id": row["image_id"],
                    "source_image_id": row["source_image_id"],
                    "candidate_key": row["candidate_key"],
                    "sheet": sheet_index,
                    "object_free": "",
                    "ambiguous": "",
                    "notes": "",
                }
            )
        canvas.save(args.output / f"background_review_{sheet_index:02d}.jpg", quality=92)
    with (args.output / "background_visual_review.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
