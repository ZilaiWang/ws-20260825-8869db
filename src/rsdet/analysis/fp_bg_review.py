"""Render blinded review cards for the N0-4 FP_BG semantic audit.

The existing N0-4 sampler deliberately stores only coordinates and audit
metadata.  A reviewer needs both local appearance and known annotations to
separate clear background from plausible missing objects, localization
failures, and duplicate fragments.  This module builds that visual evidence
without assigning any semantic label automatically.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class KnownObject:
    annotation_uid: str
    category_id: int
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class ReviewCard:
    card_id: str
    audit_uid: str
    proposal_uid: str
    image_id: int
    relative_path: str
    category_id: int
    class_name: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    known_objects: tuple[KnownObject, ...]


def _parse_bbox(value: str | Sequence[float]) -> tuple[float, float, float, float]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    result = tuple(float(item) for item in parsed)
    if len(result) != 4 or not all(math.isfinite(item) for item in result):
        raise ValueError(f"bbox 非法: {value!r}")
    x0, y0, x1, y1 = result
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"bbox 面积非正: {result}")
    return result


def load_formal_image_index(
    path: str | Path,
) -> dict[int, tuple[str, tuple[KnownObject, ...]]]:
    """Load one source path and the de-duplicated GT list for every image."""

    sources: dict[int, str] = {}
    objects: dict[int, dict[str, KnownObject]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "formal_image_id",
            "source_relative_path",
            "annotation_uid",
            "class_id",
            "gt_x0",
            "gt_y0",
            "gt_x1",
            "gt_y1",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"formal crop manifest 缺列: {sorted(missing)}")
        for row in reader:
            image_id = int(row["formal_image_id"])
            source = row["source_relative_path"].strip()
            previous = sources.setdefault(image_id, source)
            if previous != source:
                raise ValueError(f"image_id={image_id} 对应多个源路径")
            annotation_uid = row["annotation_uid"].strip()
            item = KnownObject(
                annotation_uid=annotation_uid,
                category_id=int(row["class_id"]),
                bbox_xyxy=(
                    float(row["gt_x0"]),
                    float(row["gt_y0"]),
                    float(row["gt_x1"]),
                    float(row["gt_y1"]),
                ),
            )
            bucket = objects.setdefault(image_id, {})
            existing = bucket.setdefault(annotation_uid, item)
            if existing != item:
                raise ValueError(f"annotation_uid={annotation_uid} 几何不一致")
    if not sources:
        raise ValueError("formal crop manifest 为空")
    return {
        image_id: (sources[image_id], tuple(objects.get(image_id, {}).values()))
        for image_id in sorted(sources)
    }


def load_review_cards(
    audit_csv: str | Path,
    formal_index: Mapping[int, tuple[str, tuple[KnownObject, ...]]],
) -> list[ReviewCard]:
    """Join blinded audit rows with source paths and known annotations."""

    cards: list[ReviewCard] = []
    with Path(audit_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "audit_uid",
            "proposal_uid",
            "image_id",
            "category_id",
            "class_name",
            "score",
            "bbox_xyxy",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"audit CSV 缺列: {sorted(missing)}")
        for index, row in enumerate(reader):
            image_id = int(row["image_id"])
            if image_id not in formal_index:
                raise ValueError(f"audit image_id 不在 formal manifest: {image_id}")
            relative_path, known = formal_index[image_id]
            cards.append(
                ReviewCard(
                    card_id=f"card-{index:04d}",
                    audit_uid=row["audit_uid"].strip(),
                    proposal_uid=row["proposal_uid"].strip(),
                    image_id=image_id,
                    relative_path=relative_path,
                    category_id=int(row["category_id"]),
                    class_name=row["class_name"].strip(),
                    score=float(row["score"]),
                    bbox_xyxy=_parse_bbox(row["bbox_xyxy"]),
                    known_objects=known,
                )
            )
    if not cards:
        raise ValueError("audit CSV 为空")
    return cards


def _expanded_square(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
    scale: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    width, height = image_size
    side = max(x1 - x0, y1 - y0) * scale
    side = max(side, 64.0)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    left = max(0.0, min(cx - side / 2.0, width - side))
    top = max(0.0, min(cy - side / 2.0, height - side))
    right = min(float(width), left + side)
    bottom = min(float(height), top + side)
    return left, top, right, bottom


def _draw_box(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[float, float, float, float],
    crop: tuple[float, float, float, float],
    output_size: tuple[int, int],
    color: tuple[int, int, int],
    width: int,
) -> None:
    left, top, right, bottom = crop
    sx = output_size[0] / max(right - left, 1e-6)
    sy = output_size[1] / max(bottom - top, 1e-6)
    x0, y0, x1, y1 = bbox
    transformed = (
        (x0 - left) * sx,
        (y0 - top) * sy,
        (x1 - left) * sx,
        (y1 - top) * sy,
    )
    draw.rectangle(transformed, outline=color, width=width)


def render_review_card(
    card: ReviewCard,
    image: Image.Image,
    *,
    panel_size: int = 320,
    context_scale: float = 4.0,
) -> Image.Image:
    """Render full/context/tight panels; red=prediction, green=known GT."""

    source = image.convert("RGB")
    full_crop = (0.0, 0.0, float(source.width), float(source.height))
    context_crop = _expanded_square(card.bbox_xyxy, source.size, context_scale)
    tight_crop = _expanded_square(card.bbox_xyxy, source.size, 1.35)
    panels: list[Image.Image] = []
    for crop, show_gt in ((full_crop, True), (context_crop, True), (tight_crop, False)):
        panel = source.crop(tuple(int(round(item)) for item in crop)).resize(
            (panel_size, panel_size), Image.Resampling.BICUBIC
        )
        draw = ImageDraw.Draw(panel)
        _draw_box(draw, card.bbox_xyxy, crop, panel.size, (255, 48, 48), 4)
        if show_gt:
            for known in card.known_objects:
                _draw_box(draw, known.bbox_xyxy, crop, panel.size, (40, 220, 90), 3)
        panels.append(panel)
    header = 52
    result = Image.new("RGB", (panel_size * 3, panel_size + header), "white")
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    draw.text(
        (8, 6),
        f"{card.card_id}  pred={card.class_name}/{card.category_id}  score={card.score:.3f}",
        fill="black",
        font=font,
    )
    draw.text(
        (8, 26),
        "full | context (red prediction, green known GT) | tight",
        fill=(55, 55, 55),
        font=font,
    )
    for index, panel in enumerate(panels):
        result.paste(panel, (panel_size * index, header))
    return result


def write_review_outputs(
    cards: Sequence[ReviewCard],
    data_root: str | Path,
    output_dir: str | Path,
    *,
    cards_per_sheet: int = 4,
) -> dict[str, Any]:
    """Render individual cards, contact sheets, and a blank decision contract."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(destination)
    cards_dir = destination / "cards"
    sheets_dir = destination / "contact_sheets"
    cards_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)
    root = Path(data_root).expanduser().resolve()
    rendered: list[Path] = []
    mapping_rows: list[dict[str, Any]] = []
    for card in cards:
        source_path = (root / card.relative_path).resolve()
        try:
            source_path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"源路径逃逸 data root: {source_path}") from error
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        with Image.open(source_path) as image:
            rendered_card = render_review_card(card, image)
        card_path = cards_dir / f"{card.card_id}.jpg"
        rendered_card.save(card_path, quality=92, subsampling=0)
        rendered.append(card_path)
        mapping_rows.append(
            {
                "card_id": card.card_id,
                "audit_uid": card.audit_uid,
                "proposal_uid": card.proposal_uid,
                "image_id": card.image_id,
                "relative_path": card.relative_path,
                "label": "",
                "labeler": "",
                "notes": "",
            }
        )
    for sheet_index, start in enumerate(range(0, len(rendered), cards_per_sheet)):
        paths = rendered[start : start + cards_per_sheet]
        images = [Image.open(path).convert("RGB") for path in paths]
        width = max(image.width for image in images)
        height = sum(image.height for image in images)
        sheet = Image.new("RGB", (width, height), "white")
        offset = 0
        for image in images:
            sheet.paste(image, (0, offset))
            offset += image.height
            image.close()
        sheet.save(sheets_dir / f"sheet-{sheet_index:03d}.jpg", quality=90, subsampling=0)
    mapping_fields = [
        "card_id",
        "audit_uid",
        "proposal_uid",
        "image_id",
        "relative_path",
    ]
    mapping_path = destination / "sealed_card_mapping.csv"
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=mapping_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(mapping_rows)
    decision_fields = ["card_id", "label", "labeler", "notes"]
    decision_path = destination / "manual_review_decisions.csv"
    with decision_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=decision_fields)
        writer.writeheader()
        writer.writerows(
            {"card_id": row["card_id"], "label": "", "labeler": "", "notes": ""}
            for row in mapping_rows
        )
    digest = hashlib.sha256()
    for path in rendered:
        digest.update(path.read_bytes())
    summary = {
        "status": "waiting_for_manual_review",
        "card_count": len(cards),
        "sheet_count": math.ceil(len(cards) / cards_per_sheet),
        "cards_per_sheet": cards_per_sheet,
        "individual_cards_ordered_sha256": digest.hexdigest(),
        "review_csv_is_blinded": True,
        "sealed_mapping_must_not_be_opened_during_review": True,
        "automatic_background_admission": False,
    }
    (destination / "render_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def label_counts(rows: Iterable[Mapping[str, str]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        value = row.get("label", "").strip()
        if value:
            result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


__all__ = [
    "KnownObject",
    "ReviewCard",
    "label_counts",
    "load_formal_image_index",
    "load_review_cards",
    "render_review_card",
    "write_review_outputs",
]
