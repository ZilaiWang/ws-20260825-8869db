"""Object-centric scale refinement without isolated object stretching."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Sequence

from .jitter_hard_negative import Box


def intersection_area(first: Box, second: Box) -> float:
    return max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1)) * max(
        0.0, min(first.y2, second.y2) - max(first.y1, second.y1)
    )


def visibility(box: Box, crop: Box) -> float:
    return intersection_area(box, crop) / box.area if box.area > 0 else 0.0


@dataclass(frozen=True)
class ScaleCropPolicy:
    network_size: int = 1024
    target_network_side: int = 48
    target_visibility: float = 0.95
    keep_visibility: float = 0.70
    reject_partial_visibility: float = 0.05
    center_jitter_fraction: float = 0.06
    context_side_min: float = 96.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.network_size <= 0 or not 0 < self.target_network_side <= self.network_size:
            raise ValueError("invalid network sizes")
        if not (
            0
            <= self.reject_partial_visibility
            < self.keep_visibility
            <= self.target_visibility
            <= 1
        ):
            raise ValueError("invalid visibility thresholds")
        if self.center_jitter_fraction < 0 or self.context_side_min <= 0:
            raise ValueError("invalid crop geometry")


def _rng(policy: ScaleCropPolicy, key: str) -> random.Random:
    digest = hashlib.sha256(f"{policy.seed}|{key}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def build_scale_crop(
    target: Box,
    *,
    image_width: int,
    image_height: int,
    all_boxes: Sequence[Box],
    policy: ScaleCropPolicy,
    stable_key: str,
) -> tuple[Box, list[int]] | None:
    """Return a square scene crop and the fully retained annotation indices."""

    del image_width, image_height  # Materialization handles reflect padding.
    if target.area <= 0:
        raise ValueError("target must have positive area")
    side = max(
        policy.context_side_min,
        max(target.width, target.height) * policy.network_size / policy.target_network_side,
    )
    rng = _rng(policy, stable_key)
    cx, cy = (target.x1 + target.x2) / 2, (target.y1 + target.y2) / 2
    cx += rng.uniform(-policy.center_jitter_fraction, policy.center_jitter_fraction) * side
    cy += rng.uniform(-policy.center_jitter_fraction, policy.center_jitter_fraction) * side
    crop = Box(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    if visibility(target, crop) < policy.target_visibility:
        return None
    kept: list[int] = []
    for index, box in enumerate(all_boxes):
        visible = visibility(box, crop)
        if visible >= policy.keep_visibility:
            kept.append(index)
        elif visible > policy.reject_partial_visibility:
            return None
    return (crop, kept) if kept else None


def transform_box_to_crop(box: Box, crop: Box, output_size: int) -> Box:
    if crop.width <= 0 or crop.height <= 0 or output_size <= 0:
        raise ValueError("invalid crop or output size")
    return Box(
        (box.x1 - crop.x1) * output_size / crop.width,
        (box.y1 - crop.y1) * output_size / crop.height,
        (box.x2 - crop.x1) * output_size / crop.width,
        (box.y2 - crop.y1) * output_size / crop.height,
    ).clip(output_size, output_size)
