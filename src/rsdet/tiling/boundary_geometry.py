"""Geometry helpers for boundary-aware large-image tiling.

The production slicer remains edge aligned.  This module gives experiments a
single, tested definition of tile ownership, context margin and phase-shifted
diagnostic grids without changing that production default.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rsdet.contracts import TileRecord

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class VirtualTile:
    """A possibly padded tile used only by geometry/phase diagnostics."""

    tile_id: int
    x_start: int
    y_start: int
    tile_size: int
    image_width: int
    image_height: int

    @property
    def box(self) -> Box:
        return (
            float(self.x_start),
            float(self.y_start),
            float(self.x_start + self.tile_size),
            float(self.y_start + self.tile_size),
        )


def _validate_box(box: Sequence[float]) -> Box:
    if len(box) != 4:
        raise ValueError("box must contain four values")
    values = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("box must contain finite values")
    if values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError("box must have positive area")
    return values


def _area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = _validate_box(box)
    return (x2 - x1) * (y2 - y1)


def edge_aligned_axis_starts(length: int, tile_size: int, stride: int) -> tuple[int, ...]:
    """Match the production slicer's edge-aligned starts exactly."""
    if length <= 0 or tile_size <= 0 or stride <= 0 or stride > tile_size:
        raise ValueError("require length/tile_size/stride > 0 and stride <= tile_size")
    if length <= tile_size:
        return (0,)
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return tuple(starts)


def padded_phase_axis_starts(
    length: int,
    tile_size: int,
    stride: int,
    phase: int,
) -> tuple[int, ...]:
    """Return a regular shifted grid whose outside portion requires padding."""
    if length <= 0 or tile_size <= 0 or stride <= 0 or stride > tile_size:
        raise ValueError("invalid grid geometry")
    start = -(int(phase) % stride)
    values: list[int] = []
    while start < length:
        values.append(start)
        start += stride
    if values[-1] + tile_size < length:
        values.append(values[-1] + stride)
    return tuple(values)


def build_virtual_tiles(
    image_width: int,
    image_height: int,
    tile_size: int,
    overlap: int,
    *,
    phase_x: int = 0,
    phase_y: int = 0,
    padded_phase: bool = False,
) -> tuple[VirtualTile, ...]:
    """Build the production grid or a reflection-padded diagnostic phase."""
    if not 0 <= overlap < tile_size:
        raise ValueError("require 0 <= overlap < tile_size")
    stride = tile_size - overlap
    if padded_phase:
        xs = padded_phase_axis_starts(image_width, tile_size, stride, phase_x)
        ys = padded_phase_axis_starts(image_height, tile_size, stride, phase_y)
    else:
        if phase_x or phase_y:
            raise ValueError("non-zero phase requires padded_phase=True")
        xs = edge_aligned_axis_starts(image_width, tile_size, stride)
        ys = edge_aligned_axis_starts(image_height, tile_size, stride)
    return tuple(
        VirtualTile(index, x, y, tile_size, image_width, image_height)
        for index, (y, x) in enumerate((y, x) for y in ys for x in xs)
    )


def visible_fraction(box: Sequence[float], tile: VirtualTile) -> float:
    x1, y1, x2, y2 = _validate_box(box)
    tx1, ty1, tx2, ty2 = tile.box
    intersection = max(0.0, min(x2, tx2) - max(x1, tx1)) * max(
        0.0, min(y2, ty2) - max(y1, ty1)
    )
    return intersection / _area((x1, y1, x2, y2))


def fully_contained(box: Sequence[float], tile: VirtualTile, eps: float = 1e-9) -> bool:
    x1, y1, x2, y2 = _validate_box(box)
    tx1, ty1, tx2, ty2 = tile.box
    return x1 >= tx1 - eps and y1 >= ty1 - eps and x2 <= tx2 + eps and y2 <= ty2 + eps


def normalized_context_margin(box: Sequence[float], tile: VirtualTile) -> float:
    """Minimum box-to-tile edge distance divided by the object's largest side."""
    x1, y1, x2, y2 = _validate_box(box)
    tx1, ty1, tx2, ty2 = tile.box
    scale = max(x2 - x1, y2 - y1)
    return min(x1 - tx1, y1 - ty1, tx2 - x2, ty2 - y2) / scale


def best_geometry(box: Sequence[float], tiles: Iterable[VirtualTile]) -> dict[str, float | int | bool]:
    values = tuple(tiles)
    if not values:
        raise ValueError("tiles must be non-empty")
    fractions = [visible_fraction(box, tile) for tile in values]
    full = [tile for tile in values if fully_contained(box, tile)]
    margins = [normalized_context_margin(box, tile) for tile in full]
    return {
        "max_visible_fraction": max(fractions),
        "fully_contained_count": len(full),
        "has_fully_contained_view": bool(full),
        "best_normalized_context_margin": max(margins) if margins else float("-inf"),
    }


def guaranteed_full_containment(
    box_width: float,
    box_height: float,
    *,
    tile_size: int,
    overlap: int,
) -> bool:
    """Return the infinite-grid containment guarantee ``w,h <= overlap``."""
    if box_width <= 0.0 or box_height <= 0.0:
        raise ValueError("box dimensions must be positive")
    if not 0 <= overlap < tile_size:
        raise ValueError("invalid tile/overlap")
    return box_width <= overlap and box_height <= overlap


def axis_voronoi_cores(
    starts_and_sizes: Sequence[tuple[int, int]],
    length: int,
) -> tuple[tuple[float, float], ...]:
    """Partition one image axis at midpoints between actual tile centres."""
    if not starts_and_sizes or length <= 0:
        raise ValueError("starts_and_sizes must be non-empty and length positive")
    normalized = tuple(sorted(set((int(start), int(size)) for start, size in starts_and_sizes)))
    if normalized != tuple((int(start), int(size)) for start, size in starts_and_sizes):
        raise ValueError("starts_and_sizes must be unique and sorted")
    centres = [start + size / 2.0 for start, size in normalized]
    boundaries = [0.0]
    boundaries.extend((left + right) / 2.0 for left, right in zip(centres, centres[1:]))
    boundaries.append(float(length))
    return tuple((boundaries[index], boundaries[index + 1]) for index in range(len(normalized)))


def tile_owner_lookup(
    tiles: Sequence[TileRecord],
    *,
    image_width: int,
    image_height: int,
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...], dict[tuple[int, int], int]]:
    """Build Voronoi cores and an ``(x-index,y-index) -> tile_id`` lookup."""
    if not tiles:
        raise ValueError("tiles must be non-empty")
    xs = sorted({(int(tile.x_offset), int(tile.width)) for tile in tiles})
    ys = sorted({(int(tile.y_offset), int(tile.height)) for tile in tiles})
    x_index = {start: index for index, (start, _) in enumerate(xs)}
    y_index = {start: index for index, (start, _) in enumerate(ys)}
    lookup = {
        (x_index[int(tile.x_offset)], y_index[int(tile.y_offset)]): int(tile.tile_id)
        for tile in tiles
    }
    if len(lookup) != len(xs) * len(ys):
        raise ValueError("tiles must form a complete rectangular grid")
    return axis_voronoi_cores(xs, image_width), axis_voronoi_cores(ys, image_height), lookup


def locate_owner_tile(
    x: float,
    y: float,
    *,
    x_cores: Sequence[tuple[float, float]],
    y_cores: Sequence[tuple[float, float]],
    lookup: dict[tuple[int, int], int],
) -> int:
    """Return the unique tile that owns a full-image point."""

    def locate(value: float, cores: Sequence[tuple[float, float]]) -> int:
        for index, (left, right) in enumerate(cores):
            if left <= value < right or (index == len(cores) - 1 and value == right):
                return index
        raise ValueError(f"point coordinate {value} lies outside ownership cores")

    return lookup[(locate(float(x), x_cores), locate(float(y), y_cores))]


__all__ = [
    "VirtualTile",
    "axis_voronoi_cores",
    "best_geometry",
    "build_virtual_tiles",
    "edge_aligned_axis_starts",
    "fully_contained",
    "guaranteed_full_containment",
    "locate_owner_tile",
    "normalized_context_margin",
    "padded_phase_axis_starts",
    "tile_owner_lookup",
    "visible_fraction",
]
