#!/usr/bin/env python3
"""Build the bounded, blinded pair review pack that precedes core construction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from rsdet.grouping.contracts import atomic_write_json, sha256_file
from rsdet.grouping.masks import build_protocol_foreground_mask
from rsdet.grouping.registry import load_annotations, load_registry


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MAR20 TASK-01 blind review pack")
    parser.add_argument("--pair-evidence", required=True)
    parser.add_argument("--geometry-summary", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--geometry-decision", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--mar20-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--new-pair-count", type=int, default=300)
    parser.add_argument("--control-pair-count", type=int, default=48)
    parser.add_argument("--duplicate-fraction", type=float, default=0.08)
    parser.add_argument("--cards-per-sheet", type=int, default=4)
    return parser.parse_args(argv)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _stable(text: str) -> str:
    return hashlib.sha256(f"mar20-task01-review-v1|{text}".encode()).hexdigest()


def _number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def _thumb(image: Image.Image, width: int = 240, height: int = 240) -> Image.Image:
    value = ImageOps.contain(image.convert("RGB"), (width, height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(value, ((width - value.width) // 2, (height - value.height) // 2))
    return canvas


def _background_view(image: Image.Image, boxes: list[list[float]]) -> Image.Image:
    mask = build_protocol_foreground_mask(image.size, boxes, dilation_ratio=0.15)
    background = image.copy()
    background.paste(Image.new("RGB", image.size, (128, 128, 128)), mask=mask)
    return background


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.new_pair_count < 100 or args.control_pair_count < 20:
        raise ValueError("review budget is too small for formal TASK-01")
    evidence_path = Path(args.pair_evidence).expanduser().resolve()
    geometry_summary_path = Path(args.geometry_summary).expanduser().resolve()
    assignment_path = Path(args.assignments).expanduser().resolve()
    decision_path = Path(args.geometry_decision).expanduser().resolve()
    geometry_summary = json.loads(geometry_summary_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if geometry_summary.get("pair_evidence_sha256") != sha256_file(evidence_path):
        raise ValueError("geometry evidence SHA mismatch")
    if (
        decision.get("status") != "ready_for_blind_review_pack"
        or decision.get("assignment_sha256") != sha256_file(assignment_path)
        or decision.get("selection_uses_heldout") is not False
    ):
        raise ValueError("geometry calibration decision is not admitted")
    evidence = {row["pair_uid"]: row for row in _read(evidence_path)}
    assignments = _read(assignment_path)
    if set(evidence) != {row["pair_uid"] for row in assignments}:
        raise ValueError("assignment/evidence pair sets differ")

    grade_order = {"Q0": 0, "Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    new_pool = [
        {**evidence[row["pair_uid"]], **row}
        for row in assignments
        if row["queue_source"] == "formal_k50_candidate" and row["queue_grade"] != "Q4"
    ]
    new_pool.sort(
        key=lambda row: (
            grade_order[row["queue_grade"]],
            -_number(row, "review_score"),
            -int(row["cross_official_side"]),
            -int(row["target_target"]),
            _stable(row["pair_uid"]),
        )
    )
    grade_targets = {"Q0": args.new_pair_count, "Q1": 180, "Q2": 80, "Q3": 40}
    selected_new = []
    selected_ids = set()
    for grade in ("Q0", "Q1", "Q2", "Q3"):
        limit = min(grade_targets[grade], args.new_pair_count - len(selected_new))
        for row in (item for item in new_pool if item["queue_grade"] == grade):
            if limit <= 0:
                break
            selected_new.append(row)
            selected_ids.add(row["pair_uid"])
            limit -= 1
    for row in new_pool:
        if len(selected_new) == args.new_pair_count:
            break
        if row["pair_uid"] not in selected_ids:
            selected_new.append(row)
            selected_ids.add(row["pair_uid"])
    if len(selected_new) < args.new_pair_count:
        raise ValueError(
            f"only {len(selected_new)} Q0-Q3 new pairs available, need {args.new_pair_count}"
        )

    controls = [
        row
        for row in evidence.values()
        if row.get("queue_source") == "calibration_control"
        and row.get("known_binary_role") in {"positive", "negative"}
    ]
    per_role = args.control_pair_count // 2
    selected_controls = []
    for role in ("positive", "negative"):
        values = [row for row in controls if row["known_binary_role"] == role]
        values.sort(key=lambda row: _stable(f"control|{role}|{row['pair_uid']}"))
        selected_controls.extend(values[:per_role])
    if len(selected_controls) < 2 * per_role:
        raise ValueError("insufficient balanced hidden controls")

    base_cards = [{**row, "is_control": 0, "expected_role": ""} for row in selected_new] + [
        {**row, "is_control": 1, "expected_role": row["known_binary_role"]}
        for row in selected_controls
    ]
    base_cards.sort(key=lambda row: _stable(f"base|{row['pair_uid']}|{row['is_control']}"))
    duplicate_count = round(len(base_cards) * args.duplicate_fraction)
    minimum_gap = min(30, max(len(base_cards) // 3, 1))
    sources = list(enumerate(base_cards[: max(len(base_cards) - minimum_gap, 1)]))
    sources.sort(key=lambda item: _stable(f"duplicate|{item[1]['pair_uid']}"))
    scheduled: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source_index, row in sources[:duplicate_count]:
        scheduled[source_index + minimum_gap].append(
            {**row, "node_u": row["node_v"], "node_v": row["node_u"], "swapped": 1}
        )
    cards = []
    for index, row in enumerate(base_cards):
        cards.append({**row, "swapped": 0})
        cards.extend(scheduled.get(index, []))

    registry_path = Path(args.registry).expanduser().resolve()
    annotations_path = Path(args.annotations).expanduser().resolve()
    registry = {row["node_uid"]: row for row in load_registry(registry_path)}
    annotations = load_annotations(annotations_path)
    root = Path(args.mar20_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    sheet_dir = output / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    hidden_rows = []
    decision_rows = []
    rendered = []
    image_cache: dict[str, tuple[Image.Image, Image.Image, Image.Image]] = {}

    def views(uid: str) -> tuple[Image.Image, Image.Image, Image.Image]:
        if uid not in image_cache:
            path = (root / registry[uid]["original_relative_path"]).resolve()
            path.relative_to(root)
            with Image.open(path) as image:
                image.load()
                original = ImageOps.exif_transpose(image).convert("RGB")
            background = _background_view(
                original, [box["xyxy"] for box in annotations[uid]["boxes"]]
            )
            edge = original.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")
            image_cache[uid] = original, background, edge
        return image_cache[uid]

    for index, card in enumerate(cards, 1):
        card_id = f"MG01-{index:04d}"
        rendered.append((card_id, views(card["node_u"]), views(card["node_v"])))
        hidden_rows.append(
            {
                "card_id": card_id,
                "pair_uid": card["pair_uid"],
                "node_u": card["node_u"],
                "node_v": card["node_v"],
                "duplicate_group": card["pair_uid"],
                "swapped": card["swapped"],
                "is_control": card["is_control"],
                "expected_role": card["expected_role"],
                "queue_grade": card.get("queue_grade", "CONTROL"),
                "review_score": card.get("review_score", ""),
                "target_relation": "target_target"
                if card["target_target"] == "1"
                else "target_bridge"
                if card["target_bridge"] == "1"
                else "bridge_bridge",
                "cross_official_side": card["cross_official_side"],
            }
        )
        decision_rows.append(
            {
                "card_id": card_id,
                "label": "",
                "confidence": "",
                "supporting_evidence": "",
                "counter_evidence": "",
                "fixed_structure_types": "",
                "style_only": "",
                "notes": "",
            }
        )
    hidden_path = output / "blind_mapping_private.csv"
    manual_path = output / "manual_review_decisions.csv"
    with hidden_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(hidden_rows[0]))
        writer.writeheader()
        writer.writerows(hidden_rows)
    with manual_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(decision_rows[0]))
        writer.writeheader()
        writer.writerows(decision_rows)

    panel, header, row_height = 240, 26, 266
    sheets = []
    for sheet_index, offset in enumerate(range(0, len(rendered), args.cards_per_sheet)):
        subset = rendered[offset : offset + args.cards_per_sheet]
        canvas = Image.new("RGB", (6 * panel, len(subset) * row_height), "white")
        draw = ImageDraw.Draw(canvas)
        for local, (card_id, left, right) in enumerate(subset):
            y = local * row_height
            values = (left[0], right[0], left[1], right[1], left[2], right[2])
            labels = (
                "A original",
                "B original",
                "A background",
                "B background",
                "A edge",
                "B edge",
            )
            for column, (label, image) in enumerate(zip(labels, values, strict=True)):
                x = column * panel
                canvas.paste(_thumb(image, panel, panel), (x, y + header))
                draw.text((x + 4, y + 5), f"{card_id} | {label}", fill="black")
        path = sheet_dir / f"task01-sheet-{sheet_index:03d}.jpg"
        canvas.save(path, quality=90, optimize=False, progressive=False)
        sheets.append({"path": path.relative_to(output).as_posix(), "sha256": sha256_file(path)})
    review_summary = {
        "status": "waiting_for_blind_pair_review",
        "new_pair_count": len(selected_new),
        "control_pair_count": len(selected_controls),
        "base_pair_count": len(base_cards),
        "blind_duplicate_count": duplicate_count,
        "card_count": len(cards),
        "contact_sheet_count": len(sheets),
        "new_pair_grade_counts": dict(
            sorted(Counter(row["queue_grade"] for row in selected_new).items())
        ),
        "formal_grouping_admission": False,
        "artifacts": {
            "manual_review_decisions.csv": sha256_file(manual_path),
            "blind_mapping_private.csv": sha256_file(hidden_path),
            "contact_sheets": sheets,
        },
    }
    atomic_write_json(output / "blind_review_summary.json", review_summary)
    atomic_write_json(
        output / "task_decision.json",
        {
            "status": "waiting_for_blind_pair_review",
            "retrieval_admission": True,
            "geometry_evidence_admission": True,
            "blind_review_pack_admission": True,
            "formal_grouping_admission": False,
            "next_action": "complete_manual_review_then_compile_strict_core",
            "geometry_decision_sha256": sha256_file(decision_path),
            "blind_review_summary_sha256": sha256_file(output / "blind_review_summary.json"),
        },
    )
    print(json.dumps(review_summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
