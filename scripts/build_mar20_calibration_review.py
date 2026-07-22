#!/usr/bin/env python3
"""MG01：生成背景地点描述子校准用的盲化 pair 复核包。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageOps

from rsdet.grouping.contracts import (
    PROTOCOL_VERSION,
    atomic_write_json,
    canonical_pair_uid,
    sha256_file,
)
from rsdet.grouping.masks import apply_mask_fill, build_foreground_mask
from rsdet.grouping.registry import load_annotations, load_registry


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 MAR20 校准 pair 盲评包")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--mar20-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--near-duplicate-json")
    parser.add_argument("--pair-count", type=int, default=360)
    parser.add_argument("--duplicate-fraction", type=float, default=0.08)
    parser.add_argument("--dilation-ratio", type=float, default=0.15)
    parser.add_argument("--fill-method", choices=("telea", "blur", "local_mean"), default="telea")
    parser.add_argument("--cards-per-sheet", type=int, default=4)
    return parser.parse_args(argv)


def _dominant_class(row: dict[str, str]) -> int:
    hist = json.loads(row["fine_class_hist_json"])
    maximum = max(hist.values())
    return min(int(key) for key, value in hist.items() if value == maximum)


def _stable_order(route: str, pair_uid: str) -> str:
    return hashlib.sha256(f"{PROTOCOL_VERSION}|{route}|{pair_uid}".encode()).hexdigest()


def _candidate_pairs(
    rows: list[dict[str, str]], near_duplicate_json: str | None, pair_count: int
) -> list[dict[str, str]]:
    by_uid = {row["node_uid"]: row for row in rows if row["is_target"] == "1"}
    by_class: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in by_uid.values():
        by_class[_dominant_class(row)].append(row)
    for values in by_class.values():
        values.sort(key=lambda row: int(row["mar20_number"]))
    candidates: dict[str, dict[str, str]] = {}

    def add(node_u: str, node_v: str, route: str) -> None:
        if node_u not in by_uid or node_v not in by_uid or node_u == node_v:
            return
        pair_uid = canonical_pair_uid(node_u, node_v)
        existing = candidates.get(pair_uid)
        if existing is None:
            left, right = pair_uid.split("--")
            candidates[pair_uid] = {
                "pair_uid": pair_uid,
                "node_u": left,
                "node_v": right,
                "route": route,
            }
        elif route not in existing["route"].split("+"):
            existing["route"] = "+".join(sorted(existing["route"].split("+") + [route]))

    # 完整像素相同是唯一自动 H0；仍放入少量人工包用于流程审计。
    by_pixel: dict[str, list[str]] = defaultdict(list)
    for row in by_uid.values():
        by_pixel[row["original_pixel_sha256"]].append(row["node_uid"])
    for group in by_pixel.values():
        for index, node_u in enumerate(group):
            for node_v in group[index + 1 :]:
                add(node_u, node_v, "h0_pixel_equal")
    if near_duplicate_json:
        payload = json.loads(Path(near_duplicate_json).expanduser().resolve().read_text(encoding="utf-8"))
        for group in payload.get("duplicate_groups", []):
            normalized = [f"mar20:{int(value.removeprefix('MAR20_'))}" for value in group]
            for index, node_u in enumerate(normalized):
                for node_v in normalized[index + 1 :]:
                    add(node_u, node_v, "legacy_dhash_candidate")

    # 编号相邻只是 hard-negative/潜在同源候选，不给予任何正标签。
    ordered = sorted(by_uid.values(), key=lambda row: int(row["mar20_number"]))
    for left, right in zip(ordered, ordered[1:], strict=False):
        if int(right["mar20_number"]) - int(left["mar20_number"]) <= 2:
            add(left["node_uid"], right["node_uid"], "adjacent_id_audit")

    # 同主类的跨官方侧/同侧候选，按稳定哈希取样；不把官方侧当机场标签。
    route_pools: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for values in by_class.values():
        for index, left in enumerate(values):
            for right in values[index + 1 :]:
                route = (
                    "same_class_cross_official_side"
                    if left["official_side"] != right["official_side"]
                    else "same_class_same_official_side"
                )
                route_pools[route].append((left["node_uid"], right["node_uid"]))
    quota = max(pair_count // 3, 1)
    for route, pairs in sorted(route_pools.items()):
        ordered_pairs = sorted(
            pairs,
            key=lambda pair: _stable_order(route, canonical_pair_uid(*pair)),
        )
        for node_u, node_v in ordered_pairs[:quota]:
            add(node_u, node_v, route)

    # 不同主类普通负例只用于检查描述子是否被通用风格主导。
    classes = sorted(by_class)
    cross_class = []
    for left_class, right_class in zip(classes, classes[1:], strict=False):
        for left in by_class[left_class][:40]:
            for right in by_class[right_class][:40]:
                pair_uid = canonical_pair_uid(left["node_uid"], right["node_uid"])
                cross_class.append((pair_uid, left["node_uid"], right["node_uid"]))
    cross_class.sort(key=lambda item: _stable_order("different_class_audit", item[0]))
    for _, node_u, node_v in cross_class[:quota]:
        add(node_u, node_v, "different_class_audit")

    priority = {
        "h0_pixel_equal": 0,
        "legacy_dhash_candidate": 1,
        "adjacent_id_audit": 2,
        "same_class_cross_official_side": 3,
        "same_class_same_official_side": 4,
        "different_class_audit": 5,
    }
    values = list(candidates.values())
    values.sort(
        key=lambda item: (
            min(priority.get(route, 99) for route in item["route"].split("+")),
            _stable_order(item["route"], item["pair_uid"]),
        )
    )
    if len(values) < pair_count:
        raise ValueError(f"只能构造 {len(values)} 对，少于请求的 {pair_count}")
    return values[:pair_count]


def _thumb(image: Image.Image, size: int = 300) -> Image.Image:
    value = ImageOps.contain(image.convert("RGB"), (size, size), method=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(value, ((size - value.width) // 2, (size - value.height) // 2))
    return canvas


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.pair_count <= 0 or args.cards_per_sheet <= 0:
        raise ValueError("pair-count/cards-per-sheet 必须大于 0")
    if not 0 <= args.duplicate_fraction <= 0.2:
        raise ValueError("duplicate-fraction 必须在 [0,0.2]")
    registry_rows = load_registry(args.registry)
    rows_by_uid = {row["node_uid"]: row for row in registry_rows}
    annotations = load_annotations(args.annotations)
    pairs = _candidate_pairs(registry_rows, args.near_duplicate_json, args.pair_count)
    duplicate_count = round(args.pair_count * args.duplicate_fraction)
    minimum_gap = min(30, max(args.pair_count // 2, 1))
    base_cards = sorted(
        pairs,
        key=lambda item: _stable_order("blind_base", item["pair_uid"]),
    )
    eligible_sources = list(enumerate(base_cards[: max(len(base_cards) - minimum_gap, 1)]))
    eligible_sources.sort(
        key=lambda item: _stable_order("blind_duplicate", item[1]["pair_uid"])
    )
    duplicate_sources = eligible_sources[:duplicate_count]
    scheduled: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source_position, duplicate in duplicate_sources:
        scheduled[source_position + minimum_gap].append(
            {
                **duplicate,
                "node_u": duplicate["node_v"],
                "node_v": duplicate["node_u"],
                "duplicate_group": duplicate["pair_uid"],
                "swapped": True,
                "source_index": source_position,
            }
        )
    cards: list[dict[str, Any]] = []
    for index, pair in enumerate(base_cards):
        cards.append(
            {
                **pair,
                "duplicate_group": pair["pair_uid"],
                "swapped": False,
                "source_index": index,
            }
        )
        cards.extend(scheduled.get(index, []))
    output_dir = Path(args.output_dir).expanduser().resolve()
    sheets_dir = output_dir / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    mapping_rows = []
    decision_rows = []
    rendered_cards = []
    mar20_root = Path(args.mar20_root).expanduser().resolve()
    for card_index, card in enumerate(cards):
        card_id = f"CAL-{card_index + 1:04d}"
        images = []
        for side in ("node_u", "node_v"):
            node_uid = card[side]
            row = rows_by_uid[node_uid]
            path = (mar20_root / row["original_relative_path"]).resolve()
            try:
                path.relative_to(mar20_root)
            except ValueError as error:
                raise ValueError(f"图像路径逃逸: {path}") from error
            with Image.open(path) as source:
                source.load()
                rgb = ImageOps.exif_transpose(source).convert("RGB")
            mask = build_foreground_mask(
                rgb.size,
                [item["xyxy"] for item in annotations[node_uid]["boxes"]],
                dilation_ratio=args.dilation_ratio,
            )
            masked = apply_mask_fill(rgb, mask, method=args.fill_method)
            images.append((rgb, masked))
        rendered_cards.append((card_id, images))
        mapping_rows.append(
            {
                "card_id": card_id,
                "pair_uid": card["pair_uid"],
                "node_u": card["node_u"],
                "node_v": card["node_v"],
                "route": card["route"],
                "duplicate_group": card["duplicate_group"],
                "swapped": int(card["swapped"]),
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
    sheet_artifacts = []
    cell = 300
    row_height = 330
    for sheet_index, offset in enumerate(range(0, len(rendered_cards), args.cards_per_sheet)):
        subset = rendered_cards[offset : offset + args.cards_per_sheet]
        canvas = Image.new("RGB", (4 * cell, len(subset) * row_height), "white")
        draw = ImageDraw.Draw(canvas)
        for local_index, (card_id, images) in enumerate(subset):
            y = local_index * row_height
            labels = ("A original", "B original", "A masked", "B masked")
            values = (images[0][0], images[1][0], images[0][1], images[1][1])
            for column, (label, image) in enumerate(zip(labels, values, strict=True)):
                x = column * cell
                canvas.paste(_thumb(image, cell), (x, y + 30))
                draw.text((x + 4, y + 5), f"{card_id} | {label}", fill="black")
        path = sheets_dir / f"calibration-sheet-{sheet_index:03d}.jpg"
        canvas.save(path, quality=92, optimize=False, progressive=False)
        sheet_artifacts.append({"path": path.relative_to(output_dir).as_posix(), "sha256": sha256_file(path)})
    mapping_path = output_dir / "blind_card_mapping.csv"
    with mapping_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(mapping_rows[0]))
        writer.writeheader()
        writer.writerows(mapping_rows)
    decisions_path = output_dir / "manual_calibration_decisions.csv"
    with decisions_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(decision_rows[0]))
        writer.writeheader()
        writer.writerows(decision_rows)
    node_list = sorted(
        {row["node_u"] for row in mapping_rows} | {row["node_v"] for row in mapping_rows},
        key=lambda value: int(value.split(":")[1]),
    )
    node_list_path = output_dir / "blind_calibration_node_uids.txt"
    node_list_path.write_text("\n".join(node_list) + "\n", encoding="utf-8")
    summary = {
        "status": "waiting_for_blind_manual_review",
        "protocol_version": PROTOCOL_VERSION,
        "unique_pair_count": len(pairs),
        "unique_node_count": len(node_list),
        "blind_duplicate_count": len(duplicate_sources),
        "minimum_duplicate_card_gap": minimum_gap,
        "card_count": len(cards),
        "formal_descriptor_admission": False,
        "instructions": {
            "allowed_labels": [
                "same_frame",
                "geometric_overlap",
                "same_local_site",
                "likely_same_airport",
                "not_same_local_site",
                "different_airport",
                "uncertain",
            ],
            "do_not_open_mapping_before_review": True,
        },
        "artifacts": {
            "blind_card_mapping.csv": sha256_file(mapping_path),
            "manual_calibration_decisions.csv": sha256_file(decisions_path),
            "blind_calibration_node_uids.txt": sha256_file(node_list_path),
            "contact_sheets": sheet_artifacts,
        },
    }
    atomic_write_json(output_dir / "calibration_review_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
