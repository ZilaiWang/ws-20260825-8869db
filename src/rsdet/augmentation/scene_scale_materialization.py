"""Materialize context-preserving object-scale scenes for detector training."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .jitter_hard_negative import Box
from .object_scale_refinement import transform_box_to_crop


def reflect_crop(image: Image.Image, crop: Box, output_size: int) -> Image.Image:
    """Crop a possibly out-of-bounds square with reflect padding and resize it."""

    if crop.width <= 0 or crop.height <= 0 or output_size <= 0:
        raise ValueError("invalid crop geometry")
    array = np.asarray(image.convert("RGB"))
    height, width = array.shape[:2]
    left = max(0, int(math.ceil(-crop.x1)))
    top = max(0, int(math.ceil(-crop.y1)))
    right = max(0, int(math.ceil(crop.x2 - width)))
    bottom = max(0, int(math.ceil(crop.y2 - height)))
    # NumPy reflect requires an axis of at least two pixels. Competition scenes
    # are much larger, but fail explicitly instead of silently changing policy.
    if min(width, height) < 2:
        raise ValueError("reflect padding requires both image axes >= 2")
    padded = np.pad(array, ((top, bottom), (left, right), (0, 0)), mode="reflect")
    shifted = (crop.x1 + left, crop.y1 + top, crop.x2 + left, crop.y2 + top)
    result = Image.fromarray(padded).crop(shifted)
    return result.resize((output_size, output_size), Image.Resampling.LANCZOS)


def yolo_rows_for_crop(
    boxes: Sequence[Box],
    category_ids: Sequence[int],
    kept_indices: Sequence[int],
    crop: Box,
    output_size: int,
) -> list[str]:
    """Transform retained annotations into normalized YOLO rows."""

    if len(boxes) != len(category_ids):
        raise ValueError("boxes and category_ids must have equal length")
    rows: list[str] = []
    for index in kept_indices:
        box = transform_box_to_crop(boxes[index], crop, output_size)
        if box.width <= 0 or box.height <= 0:
            raise ValueError("retained annotation became empty")
        cx = (box.x1 + box.x2) / (2.0 * output_size)
        cy = (box.y1 + box.y2) / (2.0 * output_size)
        width = box.width / output_size
        height = box.height / output_size
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)):
            raise ValueError("transformed annotation is outside normalized bounds")
        rows.append(
            f"{int(category_ids[index])} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}"
        )
    return rows


def paired_label_path(image_path: Path) -> Path:
    """Return the YOLO label path for an image path using directory semantics."""

    parts = list(image_path.parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as error:
        raise ValueError(f"image path has no images directory: {image_path}") from error
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")
