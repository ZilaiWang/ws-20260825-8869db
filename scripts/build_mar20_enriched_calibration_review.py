#!/usr/bin/env python3
"""Build a blinded, positive-enriched review batch from 00B retrieval candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from rsdet.grouping.contracts import MASKED_PATCH_PROTOCOL_VERSION, atomic_write_json, sha256_file
from rsdet.grouping.registry import load_registry


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MAR20 00B enriched blind review")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--mar20-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pair-count", type=int, default=240)
    parser.add_argument("--duplicate-fraction", type=float, default=0.10)
    parser.add_argument("--minimum-target-target-fraction", type=float, default=0.75)
    parser.add_argument("--cards-per-sheet", type=int, default=4)
    return parser.parse_args(argv)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _number(row: dict[str, str], key: str, default: float) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def _stable(key: str) -> str:
    return hashlib.sha256(f"{MASKED_PATCH_PROTOCOL_VERSION}|{key}".encode()).hexdigest()


def _thumb(image: Image.Image, size: int = 320) -> Image.Image:
    value = ImageOps.contain(image.convert("RGB"), (size, size), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(value, ((size - value.width) // 2, (size - value.height) // 2))
    return canvas


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidate_path = Path(args.candidates).expanduser().resolve()
    summary_path = Path(args.candidate_summary).expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "pass" or summary.get("output_sha256") != sha256_file(
        candidate_path
    ):
        raise ValueError("candidate mining artifacts not admitted")
    candidates = _read(candidate_path)
    if len(candidates) < args.pair_count:
        raise ValueError("not enough enriched candidates")
    candidates.sort(
        key=lambda row: (
            0 if "exact_pixel" in row["routes"].split("+") else 1,
            -int(_number(row, "sift_inliers", 0)),
            -_number(row, "sift_inlier_ratio", 0),
            -min(_number(row, "sift_coverage_u", 0), _number(row, "sift_coverage_v", 0)),
            int(_number(row, "best_rank", 999999)),
            -_number(row, "best_similarity", -1),
            _stable(row["pair_uid"]),
        )
    )
    required_tt = round(args.pair_count * args.minimum_target_target_fraction)
    selected = [row for row in candidates if row["target_relation"] == "target_target"][
        :required_tt
    ]
    selected_ids = {row["pair_uid"] for row in selected}
    for row in candidates:
        if len(selected) == args.pair_count:
            break
        if row["pair_uid"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["pair_uid"])
    target_target_count = sum(row["target_relation"] == "target_target" for row in selected)
    if len(selected) != args.pair_count or target_target_count < required_tt:
        raise ValueError("candidate pool cannot satisfy pair-count/target-target contract")
    selected.sort(key=lambda row: _stable(f"base|{row['pair_uid']}"))
    duplicate_count = round(args.pair_count * args.duplicate_fraction)
    minimum_gap = min(30, max(args.pair_count // 2, 1))
    sources = list(enumerate(selected[: max(len(selected) - minimum_gap, 1)]))
    sources.sort(key=lambda item: _stable(f"duplicate|{item[1]['pair_uid']}"))
    scheduled: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source_index, row in sources[:duplicate_count]:
        scheduled[source_index + minimum_gap].append(
            {**row, "node_u": row["node_v"], "node_v": row["node_u"], "swapped": 1}
        )
    cards = []
    for index, row in enumerate(selected):
        cards.append({**row, "swapped": 0})
        cards.extend(scheduled.get(index, []))
    registry = {row["node_uid"]: row for row in load_registry(args.registry)}
    mar20_root = Path(args.mar20_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    sheet_dir = output / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    mapping_rows = []
    decision_rows = []
    rendered = []
    for index, card in enumerate(cards):
        card_id = f"ENR-{index + 1:04d}"
        images = []
        for side in ("node_u", "node_v"):
            path = (mar20_root / registry[card[side]]["original_relative_path"]).resolve()
            path.relative_to(mar20_root)
            with Image.open(path) as image:
                image.load()
                rgb = ImageOps.exif_transpose(image).convert("RGB")
            edge = rgb.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")
            images.append((rgb, edge))
        rendered.append((card_id, images))
        mapping_rows.append(
            {
                "card_id": card_id,
                "pair_uid": card["pair_uid"],
                "node_u": card["node_u"],
                "node_v": card["node_v"],
                "route": card["routes"],
                "duplicate_group": card["pair_uid"],
                "swapped": card["swapped"],
                "target_relation": card["target_relation"],
                "cross_official_side": card["cross_official_side"],
                "sift_inliers": card["sift_inliers"],
                "sift_inlier_ratio": card["sift_inlier_ratio"],
                "sift_coverage_u": card["sift_coverage_u"],
                "sift_coverage_v": card["sift_coverage_v"],
            }
        )
        decision_rows.append(
            {
                "card_id": card_id,
                "label": "",
                "confidence": "",
                "supporting_evidence": "",
                "counter_evidence": "",
                "notes": "",
            }
        )
    mapping_path = output / "blind_card_mapping.csv"
    decisions_path = output / "manual_enriched_decisions.csv"
    with mapping_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(mapping_rows[0]))
        writer.writeheader()
        writer.writerows(mapping_rows)
    with decisions_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(decision_rows[0]))
        writer.writeheader()
        writer.writerows(decision_rows)
    cell, row_height = 320, 350
    sheets = []
    for sheet_index, offset in enumerate(range(0, len(rendered), args.cards_per_sheet)):
        subset = rendered[offset : offset + args.cards_per_sheet]
        canvas = Image.new("RGB", (4 * cell, len(subset) * row_height), "white")
        draw = ImageDraw.Draw(canvas)
        for local, (card_id, images) in enumerate(subset):
            y = local * row_height
            values = (images[0][0], images[1][0], images[0][1], images[1][1])
            labels = ("A original", "B original", "A edges", "B edges")
            for column, (label, image) in enumerate(zip(labels, values, strict=True)):
                x = column * cell
                canvas.paste(_thumb(image, cell), (x, y + 30))
                draw.text((x + 4, y + 6), f"{card_id} | {label}", fill="black")
        path = sheet_dir / f"enriched-sheet-{sheet_index:03d}.jpg"
        canvas.save(path, quality=92, optimize=False, progressive=False)
        sheets.append({"path": path.relative_to(output).as_posix(), "sha256": sha256_file(path)})
    result = {
        "status": "waiting_for_blind_manual_review",
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "unique_pair_count": len(selected),
        "card_count": len(cards),
        "blind_duplicate_count": duplicate_count,
        "target_target_count": target_target_count,
        "target_bridge_count": len(selected) - target_target_count,
        "cross_official_side_count": sum(int(row["cross_official_side"]) for row in selected),
        "selected_route_counts": {
            route: sum(route in row["routes"].split("+") for row in selected)
            for route in sorted({name for row in selected for name in row["routes"].split("+")})
        },
        "formal_descriptor_admission": False,
        "artifacts": {
            "blind_card_mapping.csv": sha256_file(mapping_path),
            "manual_enriched_decisions.csv": sha256_file(decisions_path),
            "contact_sheets": sheets,
        },
    }
    atomic_write_json(output / "enriched_review_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
