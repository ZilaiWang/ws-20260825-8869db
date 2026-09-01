"""Leakage-safe construction of low-intensity background regularization views."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_source_name(value: str | Path) -> str:
    """Return the dataset-relative source name used by Background-100MP."""

    normalized = str(value).replace("\\", "/")
    marker = "images/train/"
    if marker not in normalized:
        raise ValueError(f"path does not contain {marker!r}: {value}")
    return marker + normalized.split(marker, 1)[1]


def load_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_training_background_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_images: Iterable[str | Path],
    val_images: Iterable[str | Path],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select frozen backgrounds whose source belongs exclusively to train."""

    train_names = {canonical_source_name(value) for value in train_images}
    val_names = {canonical_source_name(value) for value in val_images}
    overlap = train_names & val_names
    if overlap:
        raise ValueError(f"train/val source overlap: {len(overlap)}")

    selected: list[dict[str, Any]] = []
    val_rows = 0
    unknown_rows = 0
    for raw in rows:
        row = dict(raw)
        source = canonical_source_name(str(row["source_file_name"]))
        row["source_file_name"] = source
        if source in train_names:
            selected.append(row)
        elif source in val_names:
            val_rows += 1
        else:
            unknown_rows += 1
    if unknown_rows:
        raise ValueError(f"background manifest has {unknown_rows} sources outside fold view")
    selected.sort(key=lambda row: (str(row["candidate_key"]), int(row["image_id"])))
    return selected, {
        "train_row_count": len(selected),
        "val_row_count": val_rows,
        "unknown_row_count": unknown_rows,
    }


def make_source_diverse_groups(
    rows: Sequence[Mapping[str, Any]], group_size: int = 4
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Build deterministic groups while avoiding repeated sources per mosaic."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    by_source: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for raw in rows:
        row = dict(raw)
        by_source[str(row["source_file_name"])].append(row)
    for source, values in by_source.items():
        by_source[source] = deque(sorted(values, key=lambda row: str(row["candidate_key"])))

    groups: list[list[dict[str, Any]]] = []
    while sum(bool(values) for values in by_source.values()) >= group_size:
        ranked = sorted(
            ((-len(values), source) for source, values in by_source.items() if values),
            key=lambda item: (item[0], item[1]),
        )
        sources = [source for _, source in ranked[:group_size]]
        groups.append([by_source[source].popleft() for source in sources])
    leftovers = sorted(
        (row for values in by_source.values() for row in values),
        key=lambda row: (str(row["source_file_name"]), str(row["candidate_key"])),
    )
    return groups, leftovers


def materialize_background_mosaics(
    groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    background_root: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    """Place four audited 512px crops into one native-scale 1024px empty image."""

    image_dir = output_root / "images/background_regularization"
    label_dir = output_root / "labels/background_regularization"
    image_dir.mkdir(parents=True, exist_ok=False)
    label_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    positions = ((0, 0), (512, 0), (0, 512), (512, 512))
    for index, group in enumerate(groups):
        if len(group) != 4:
            raise ValueError("each background mosaic must contain exactly four crops")
        source_names = [str(row["source_file_name"]) for row in group]
        if len(source_names) != len(set(source_names)):
            raise ValueError("a mosaic contains repeated source images")
        canvas = Image.new("RGB", (1024, 1024))
        input_rows = []
        for row, position in zip(group, positions, strict=True):
            source = background_root / str(row["file_name"])
            if not source.is_file():
                raise FileNotFoundError(source)
            actual_sha = sha256_file(source)
            if actual_sha != str(row["sha256"]):
                raise ValueError(f"background SHA mismatch: {source}")
            with Image.open(source) as image:
                rgb = image.convert("RGB")
                if rgb.size != (512, 512):
                    raise ValueError(f"expected 512x512 background crop: {source}")
                canvas.paste(rgb, position)
            input_rows.append(
                {
                    "candidate_key": row["candidate_key"],
                    "source_file_name": row["source_file_name"],
                    "crop_xyxy": row["crop_xyxy"],
                    "sha256": actual_sha,
                }
            )
        image_path = image_dir / f"bg_mosaic_{index:04d}.png"
        label_path = label_dir / f"bg_mosaic_{index:04d}.txt"
        canvas.save(image_path, optimize=True)
        label_path.write_text("", encoding="utf-8")
        records.append(
            {
                "mosaic_index": index,
                "image_path": str(image_path.resolve()),
                "label_path": str(label_path.resolve()),
                "image_sha256": sha256_file(image_path),
                "inputs": input_rows,
            }
        )
    return records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
