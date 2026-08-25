#!/usr/bin/env python3
"""Build deterministic overlays for the MAR20 v1.2 patch-mask audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from rsdet.grouping.contracts import MASKED_PATCH_PROTOCOL_VERSION, atomic_write_json, sha256_file
from rsdet.grouping.masks import render_masked_patch_inputs
from rsdet.grouping.registry import load_annotations, load_registry


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAR20 feature-level patch-mask audit")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--mar20-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=120)
    parser.add_argument("--dilation-ratios", default="0.10,0.15,0.20")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--maximum-patch-foreground-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-valid-patch-fraction", type=float, default=0.25)
    parser.add_argument("--cards-per-sheet", type=int, default=4)
    parser.add_argument("--seed", type=int, default=202625)
    return parser.parse_args(argv)


def _stable_sample(rows: list[dict[str, str]], count: int, seed: int) -> list[dict[str, str]]:
    target = [row for row in rows if row["is_target"] == "1"]
    hashed = sorted(
        target,
        key=lambda row: hashlib.sha256(
            f"{MASKED_PATCH_PROTOCOL_VERSION}|{seed}|{row['node_uid']}".encode()
        ).hexdigest(),
    )
    if count <= 0 or count > len(target):
        raise ValueError("sample-count outside target range")
    selected: dict[str, dict[str, str]] = {}
    # Cover all 20 aircraft classes before filling by stable hash.
    for class_id in range(4, 24):
        if len(selected) >= count:
            break
        candidate = next(
            (row for row in hashed if str(class_id) in json.loads(row["fine_class_hist_json"])),
            None,
        )
        if candidate is not None:
            selected[candidate["node_uid"]] = candidate
    # Force dense scenes into the audit because they are the main excessive-loss risk.
    if len(selected) < count:
        for row in sorted(
            target,
            key=lambda value: (-int(value["bbox_count"]), int(value["mar20_number"])),
        )[: min(20, count - len(selected))]:
            selected[row["node_uid"]] = row
    for row in hashed:
        if len(selected) >= count:
            break
        selected[row["node_uid"]] = row
    return sorted(selected.values(), key=lambda row: int(row["mar20_number"]))


def _overlay(image: Image.Image, valid: np.ndarray, patch_size: int) -> Image.Image:
    base = image.convert("RGBA")
    tint = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(tint, "RGBA")
    grid = int(round(valid.size**0.5))
    if grid * grid != valid.size:
        raise ValueError("patch mask is not square")
    mask = valid.reshape(grid, grid)
    for y in range(grid):
        for x in range(grid):
            x1, y1 = x * patch_size, y * patch_size
            x2, y2 = x1 + patch_size - 1, y1 + patch_size - 1
            if not mask[y, x]:
                draw.rectangle((x1, y1, x2, y2), fill=(255, 0, 0, 88))
                draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0, 190), width=1)
    return Image.alpha_composite(base, tint).convert("RGB")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dilations = tuple(float(value) for value in args.dilation_ratios.split(","))
    if dilations != (0.10, 0.15, 0.20):
        raise ValueError("00B audit freezes dilation ratios at 0.10,0.15,0.20")
    registry = load_registry(args.registry)
    annotations = load_annotations(args.annotations)
    sample = _stable_sample(registry, args.sample_count, args.seed)
    mar20_root = Path(args.mar20_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    overlay_dir = output / "overlays"
    sheet_dir = output / "contact_sheets"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    rendered: list[tuple[str, list[Image.Image]]] = []
    automatic_failures: list[str] = []
    for row in sample:
        uid = row["node_uid"]
        path = (mar20_root / row["original_relative_path"]).resolve()
        path.relative_to(mar20_root)
        with Image.open(path) as source:
            source.load()
            rgb = ImageOps.exif_transpose(source).convert("RGB")
        views: list[Image.Image] = []
        audit: dict[str, object] = {"node_uid": uid}
        for dilation in dilations:
            item = render_masked_patch_inputs(
                node_uid=uid,
                image=rgb,
                boxes=[box["xyxy"] for box in annotations[uid]["boxes"]],
                rotations=(0,),
                input_size=args.input_size,
                patch_size=args.patch_size,
                dilation_ratio=dilation,
                maximum_patch_foreground_fraction=args.maximum_patch_foreground_fraction,
            )[0]
            key = f"dilation_{dilation:.2f}".replace(".", "p")
            audit[f"{key}_valid_patch_fraction"] = item.valid_patch_fraction
            audit[f"{key}_valid_patch_count"] = item.valid_patch_count
            audit[f"{key}_patch_mask_sha256"] = item.patch_mask_sha256
            if item.valid_patch_fraction < args.minimum_valid_patch_fraction:
                automatic_failures.append(f"{uid}:{key}:valid_patch_fraction")
            overlay = _overlay(item.image, item.valid_patch_mask, args.patch_size)
            overlay_path = overlay_dir / f"{uid.replace(':', '-')}-{key}.png"
            overlay.save(overlay_path, optimize=False)
            views.append(overlay)
        audit_rows.append(audit)
        review_rows.append(
            {
                "node_uid": uid,
                "valid": "",
                "dilation_0p10_aircraft_covered": "",
                "dilation_0p15_aircraft_covered": "",
                "dilation_0p20_aircraft_covered": "",
                "dilation_0p15_excessive_background_loss": "",
                "notes": "",
            }
        )
        rendered.append((uid, views))
    audit_path = output / "patch_mask_audit.csv"
    review_path = output / "manual_patch_mask_review.csv"
    with audit_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    with review_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    width = 3 * args.input_size
    row_height = args.input_size + 28
    sheets = []
    for sheet_index, offset in enumerate(range(0, len(rendered), args.cards_per_sheet)):
        subset = rendered[offset : offset + args.cards_per_sheet]
        canvas = Image.new("RGB", (width, row_height * len(subset)), "white")
        draw = ImageDraw.Draw(canvas)
        for local, (uid, images) in enumerate(subset):
            y = local * row_height
            for column, (label, image) in enumerate(
                zip(("d=0.10", "d=0.15", "d=0.20"), images, strict=True)
            ):
                x = column * args.input_size
                canvas.paste(image, (x, y + 28))
                draw.text((x + 5, y + 6), f"{uid} | {label}", fill="black")
        path = sheet_dir / f"patch-mask-sheet-{sheet_index:03d}.jpg"
        canvas.save(path, quality=92, optimize=False, progressive=False)
        sheets.append({"path": path.relative_to(output).as_posix(), "sha256": sha256_file(path)})
    summary = {
        "status": "waiting_for_manual_patch_mask_review",
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "sample_count": len(sample),
        "automatic_geometry_gate": "pass" if not automatic_failures else "fail",
        "automatic_failures": automatic_failures,
        "formal_patch_mask_admission": False,
        "configuration": {
            "dilation_ratios": list(dilations),
            "primary_dilation_ratio": 0.15,
            "input_size": args.input_size,
            "patch_size": args.patch_size,
            "maximum_patch_foreground_fraction": args.maximum_patch_foreground_fraction,
            "minimum_valid_patch_fraction": args.minimum_valid_patch_fraction,
        },
        "artifacts": {
            "patch_mask_audit.csv": sha256_file(audit_path),
            "manual_patch_mask_review.csv": sha256_file(review_path),
            "contact_sheets": sheets,
        },
    }
    atomic_write_json(output / "patch_mask_audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not automatic_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
