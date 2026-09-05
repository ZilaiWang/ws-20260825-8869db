"""Lightweight style retrieval for Domain-RAG-lite real backgrounds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BackgroundDescriptor:
    key: str
    vector: np.ndarray
    semantic_tags: frozenset[str]
    source_group: str


def _unit(vector: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= epsilon:
        raise ValueError("descriptor must be finite and non-zero")
    return value / norm


def describe_image(
    image_or_path: Image.Image | str | Path,
    *,
    key: str,
    semantic_tags: Iterable[str],
    source_group: str,
    size: int = 128,
) -> BackgroundDescriptor:
    if size <= 1:
        raise ValueError("size must exceed one pixel")
    if isinstance(image_or_path, Image.Image):
        image = image_or_path.convert("RGB")
    else:
        with Image.open(image_or_path) as source:
            image = source.convert("RGB")
    array = (
        np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    )
    luminance = 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
    histogram, _ = np.histogram(luminance, bins=16, range=(0, 1), density=True)
    gx, gy = np.diff(luminance, axis=1), np.diff(luminance, axis=0)
    vector = _unit(
        np.concatenate(
            [
                array.mean(axis=(0, 1)),
                array.std(axis=(0, 1)),
                histogram.astype(np.float32),
                np.asarray([abs(gx).mean(), abs(gy).mean(), (gx * gx).mean() + (gy * gy).mean()]),
            ]
        )
    )
    tags = frozenset(str(tag).strip() for tag in semantic_tags if str(tag).strip())
    if not key or not tags or not source_group:
        raise ValueError("key, semantic_tags and source_group must be non-empty")
    return BackgroundDescriptor(str(key), vector, tags, str(source_group))


def retrieve_backgrounds(
    query: BackgroundDescriptor,
    candidates: Sequence[BackgroundDescriptor],
    *,
    required_tags: Iterable[str],
    top_k: int = 8,
    exclude_same_source_group: bool = True,
) -> list[tuple[str, float]]:
    required = frozenset(str(tag) for tag in required_tags)
    if top_k <= 0 or not required:
        raise ValueError("top_k and required_tags must be non-empty")
    rows: list[tuple[str, float]] = []
    for candidate in candidates:
        if exclude_same_source_group and candidate.source_group == query.source_group:
            continue
        if not required <= candidate.semantic_tags:
            continue
        if candidate.vector.shape != query.vector.shape:
            raise ValueError("descriptor dimensions differ")
        rows.append((candidate.key, float(query.vector @ candidate.vector)))
    rows.sort(key=lambda row: (-row[1], row[0]))
    return rows[:top_k]
