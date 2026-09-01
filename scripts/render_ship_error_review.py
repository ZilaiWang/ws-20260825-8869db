#!/usr/bin/env python3
"""Render deterministic context crops for the Ship error-review queue."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-scale", type=float, default=2.0)
    args = parser.parse_args()
    with args.review_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    coco = json.loads(args.gt.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in coco["images"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sheet_index in range((len(rows) + 15) // 16):
        canvas = Image.new("RGB", (1024, 1024), "white")
        for local_index, row in enumerate(rows[sheet_index * 16 : (sheet_index + 1) * 16]):
            image_row = images[int(row["image_id"])]
            path = args.image_root / str(image_row["file_name"])
            box = [float(value) for value in row["bbox_xyxy"].split()]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            width = max(96.0, (box[2] - box[0]) * args.context_scale)
            height = max(96.0, (box[3] - box[1]) * args.context_scale)
            side = max(width, height)
            with Image.open(path) as source:
                source = source.convert("RGB")
                crop_box = (
                    max(0, int(cx - side / 2)),
                    max(0, int(cy - side / 2)),
                    min(source.width, int(cx + side / 2)),
                    min(source.height, int(cy + side / 2)),
                )
                crop = source.crop(crop_box).resize((256, 232))
            draw = ImageDraw.Draw(crop)
            scale_x = 256 / max(1, crop_box[2] - crop_box[0])
            scale_y = 232 / max(1, crop_box[3] - crop_box[1])
            draw.rectangle(
                (
                    (box[0] - crop_box[0]) * scale_x,
                    (box[1] - crop_box[1]) * scale_y,
                    (box[2] - crop_box[0]) * scale_x,
                    (box[3] - crop_box[1]) * scale_y,
                ),
                outline="red" if row["case_side"] == "prediction" else "lime",
                width=3,
            )
            x, y = (local_index % 4) * 256, (local_index // 4) * 256
            canvas.paste(crop, (x, y + 24))
            header = ImageDraw.Draw(canvas)
            header.rectangle((x, y, x + 255, y + 23), fill="black")
            header.text(
                (x + 3, y + 4),
                f"{row['reason']} c{row['category_id']} i{row['image_id']}",
                fill="white",
            )
        canvas.save(args.output_dir / f"ship_error_review_{sheet_index:02d}.jpg", quality=92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
