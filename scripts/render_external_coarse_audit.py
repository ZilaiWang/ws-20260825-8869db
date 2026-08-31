#!/usr/bin/env python3
"""Render deterministic coarse-dataset annotation cards and contact sheets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLORS = {0: "#ff3b30", 1: "#00a7ff", 2: "#34c759", 3: "#ffcc00"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-images", type=int, default=24)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()
    payload = json.loads(args.coco.read_text(encoding="utf-8"))
    by_image: dict[int, list[dict]] = defaultdict(list)
    for row in payload["annotations"]:
        by_image[int(row["image_id"])].append(row)
    names = {int(row["id"]): row["name"] for row in payload["categories"]}
    ranked = sorted(
        payload["images"],
        key=lambda row: (-len(by_image[int(row["id"])]), int(row["id"])),
    )[: args.maximum_images]
    cards = args.output_dir / "cards"
    sheets = args.output_dir / "contact_sheets"
    cards.mkdir(parents=True, exist_ok=True)
    sheets.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    card_paths = []
    for rank, row in enumerate(ranked, 1):
        source = Image.open(args.image_root / row["file_name"]).convert("RGB")
        draw = ImageDraw.Draw(source)
        for annotation in by_image[int(row["id"])]:
            x, y, width, height = (float(value) for value in annotation["bbox"])
            category_id = int(annotation["category_id"])
            draw.rectangle((x, y, x + width, y + height), outline=COLORS[category_id], width=3)
        resized = source.copy()
        resized.thumbnail((480, 480), Image.Resampling.LANCZOS)
        card = Image.new("RGB", (500, 535), "white")
        card.paste(resized, ((500 - resized.width) // 2, 35))
        caption = f"#{rank} {row['file_name']} boxes={len(by_image[int(row['id'])])}"
        ImageDraw.Draw(card).text((8, 8), caption, fill="black", font=font)
        card_path = cards / f"{rank:03d}.jpg"
        card.save(card_path, quality=92)
        card_paths.append(card_path)
    rows_per_sheet = 3
    per_sheet = args.columns * rows_per_sheet
    for sheet_index in range(0, len(card_paths), per_sheet):
        group = card_paths[sheet_index : sheet_index + per_sheet]
        sheet = Image.new("RGB", (args.columns * 500, rows_per_sheet * 535), "white")
        for index, path in enumerate(group):
            card = Image.open(path).convert("RGB")
            sheet.paste(card, ((index % args.columns) * 500, (index // args.columns) * 535))
        sheet.save(sheets / f"sheet_{sheet_index // per_sheet + 1:03d}.jpg", quality=92)
    summary = {
        "status": "waiting_for_visual_audit",
        "rendered_image_count": len(card_paths),
        "contact_sheet_count": (len(card_paths) + per_sheet - 1) // per_sheet,
        "legend": {names[key]: COLORS[key] for key in sorted(names)},
    }
    (args.output_dir / "render_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
