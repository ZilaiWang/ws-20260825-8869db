#!/usr/bin/env python3
"""Build deterministic boxed contact sheets for a MacroExpert materialized view."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

NAMES = ["HM", "LQS", "QHS", "MS", "VEH", "AIR_REJECT"]
COLORS = ["#ff4040", "#ff9f1c", "#ffe600", "#00d084", "#00a6ff", "#bd5cff"]


def labels_for(image: Path) -> tuple[Path, list[tuple[int, float, float, float, float]]]:
    label = Path(str(image).replace("/images/", "/labels/")).with_suffix(".txt")
    rows = []
    for line in label.read_text(encoding="utf-8").splitlines():
        category, x, y, w, h = line.split()
        rows.append((int(category), float(x), float(y), float(w), float(h)))
    return label, rows


def choose(images: list[Path], count: int, seed: int) -> list[Path]:
    by_class: dict[int, list[Path]] = defaultdict(list)
    rows_by_image = {}
    for image in images:
        _, rows = labels_for(image)
        rows_by_image[image] = rows
        for category in {row[0] for row in rows}:
            by_class[category].append(image)
    rng = random.Random(seed)
    for values in by_class.values():
        rng.shuffle(values)
    selected: list[Path] = []
    selected_set: set[Path] = set()
    quota = max(1, count // len(NAMES))
    for category in range(len(NAMES)):
        for image in by_class[category]:
            if image not in selected_set:
                selected.append(image)
                selected_set.add(image)
            if sum(category in {row[0] for row in rows_by_image[item]} for item in selected) >= quota:
                break
    remaining = list(images)
    rng.shuffle(remaining)
    selected.extend(item for item in remaining if item not in selected_set)
    return selected[:count]


def render_card(image_path: Path, index: int, size: int = 320) -> Image.Image:
    _, rows = labels_for(image_path)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    scale = min(size / width, (size - 28) / height)
    resized = image.resize((round(width * scale), round(height * scale)))
    card = Image.new("RGB", (size, size), "#202020")
    x_offset = (size - resized.width) // 2
    y_offset = 28 + (size - 28 - resized.height) // 2
    card.paste(resized, (x_offset, y_offset))
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default()
    labels = sorted({row[0] for row in rows})
    draw.text((5, 7), f"{index:03d} {image_path.stem} " + ",".join(NAMES[v] for v in labels),
              fill="white", font=font)
    for category, x, y, box_w, box_h in rows:
        x1 = x_offset + (x - box_w / 2) * width * scale
        y1 = y_offset + (y - box_h / 2) * height * scale
        x2 = x_offset + (x + box_w / 2) * width * scale
        y2 = y_offset + (y + box_h / 2) * height * scale
        draw.rectangle((x1, y1, x2, y2), outline=COLORS[category], width=2)
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    images = sorted((args.view / "images/train").glob("*_r00.*"))
    selected = choose(images, args.count, args.seed)
    if len(selected) != args.count:
        raise ValueError(f"requested {args.count} unique images, found {len(selected)}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for start in range(0, len(selected), 20):
        sheet = Image.new("RGB", (5 * 320, 4 * 320), "black")
        for offset, image in enumerate(selected[start : start + 20]):
            sheet.paste(render_card(image, start + offset), ((offset % 5) * 320, (offset // 5) * 320))
            _, rows = labels_for(image)
            manifest.append({"review_index": start + offset, "image": str(image.resolve()),
                             "classes": sorted({row[0] for row in rows})})
        sheet.save(args.output / f"contact_sheet_{start // 20:02d}.jpg", quality=92)
    (args.output / "review_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
