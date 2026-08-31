#!/usr/bin/env python3
"""Render full-image and zoom cards for a missing-label review CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CARD_SIZE = (1000, 520)
PANEL_SIZE = (480, 440)


def _resample() -> int:
    return int(getattr(Image.Resampling, "LANCZOS", Image.LANCZOS))


def _fit(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, float, int, int]:
    scale = min(size[0] / image.width, size[1] / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    resized = image.resize((width, height), _resample())
    return resized, scale, (size[0] - width) // 2, (size[1] - height) // 2


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    *,
    scale: float,
    offset_x: float,
    offset_y: float,
    color: str,
    width: int,
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle(
        [
            offset_x + x1 * scale,
            offset_y + y1 * scale,
            offset_x + x2 * scale,
            offset_y + y2 * scale,
        ],
        outline=color,
        width=width,
    )


def _render_card(row: dict[str, str], image_root: Path) -> Image.Image:
    source = image_root / row["file_name"]
    if not source.is_file():
        raise FileNotFoundError(source)
    image = Image.open(source).convert("RGB")
    primary = [float(row[f"bbox_{key}"]) for key in ("x1", "y1", "x2", "y2")]
    support = [
        float(row[f"support_bbox_{key}"]) for key in ("x1", "y1", "x2", "y2")
    ]

    card = Image.new("RGB", CARD_SIZE, "white")
    full, full_scale, full_dx, full_dy = _fit(image, PANEL_SIZE)
    card.paste(full, (10 + full_dx, 55 + full_dy))
    draw = ImageDraw.Draw(card)
    _draw_box(
        draw,
        support,
        scale=full_scale,
        offset_x=10 + full_dx,
        offset_y=55 + full_dy,
        color="#00c8ff",
        width=3,
    )
    _draw_box(
        draw,
        primary,
        scale=full_scale,
        offset_x=10 + full_dx,
        offset_y=55 + full_dy,
        color="#ff3030",
        width=4,
    )

    x1, y1, x2, y2 = primary
    box_width = max(8.0, x2 - x1)
    box_height = max(8.0, y2 - y1)
    margin = max(box_width, box_height) * 1.5
    crop_box = (
        max(0, math.floor(x1 - margin)),
        max(0, math.floor(y1 - margin)),
        min(image.width, math.ceil(x2 + margin)),
        min(image.height, math.ceil(y2 + margin)),
    )
    zoom_source = image.crop(crop_box)
    zoom, zoom_scale, zoom_dx, zoom_dy = _fit(zoom_source, PANEL_SIZE)
    card.paste(zoom, (510 + zoom_dx, 55 + zoom_dy))
    _draw_box(
        draw,
        [
            support[0] - crop_box[0],
            support[1] - crop_box[1],
            support[2] - crop_box[0],
            support[3] - crop_box[1],
        ],
        scale=zoom_scale,
        offset_x=510 + zoom_dx,
        offset_y=55 + zoom_dy,
        color="#00c8ff",
        width=3,
    )
    _draw_box(
        draw,
        [x1 - crop_box[0], y1 - crop_box[1], x2 - crop_box[0], y2 - crop_box[1]],
        scale=zoom_scale,
        offset_x=510 + zoom_dx,
        offset_y=55 + zoom_dy,
        color="#ff3030",
        width=4,
    )

    font = ImageFont.load_default()
    title = (
        f"#{row['global_rank']} {row['candidate_id']}  "
        f"prod={float(row['agreement_product']):.3f} "
        f"Y5={float(row['primary_score']):.3f} "
        f"D={float(row['support_score']):.3f} "
        f"IoU={float(row['support_iou']):.3f}"
    )
    draw.text((12, 12), title, fill="black", font=font)
    draw.text(
        (12, 34),
        f"red=Y5 candidate, cyan=D-FINE support | {row['file_name']}",
        fill="black",
        font=font,
    )
    return card


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cards-per-sheet", type=int, default=6)
    args = parser.parse_args()
    if args.cards_per_sheet <= 0:
        raise ValueError("cards-per-sheet must be positive")

    rows = list(csv.DictReader(args.review_csv.open(encoding="utf-8")))
    cards_dir = args.output_dir / "cards"
    sheets_dir = args.output_dir / "contact_sheets"
    cards_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    cards: list[Image.Image] = []
    for row in rows:
        card = _render_card(row, args.image_root)
        card.save(cards_dir / f"{int(row['global_rank']):04d}_{row['candidate_id']}.jpg", quality=92)
        cards.append(card)

    for start in range(0, len(cards), args.cards_per_sheet):
        subset = cards[start : start + args.cards_per_sheet]
        sheet = Image.new("RGB", (CARD_SIZE[0], CARD_SIZE[1] * len(subset)), "white")
        for index, card in enumerate(subset):
            sheet.paste(card, (0, index * CARD_SIZE[1]))
        sheet.save(
            sheets_dir / f"sheet_{start // args.cards_per_sheet + 1:03d}.jpg",
            quality=90,
        )
    print(f"rendered {len(cards)} cards and {math.ceil(len(cards) / args.cards_per_sheet)} sheets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
