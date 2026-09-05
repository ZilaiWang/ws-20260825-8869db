"""Deterministic LMP-style jitter regions for proposal classification only."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def clip(self, width: int, height: int) -> "Box":
        return Box(
            min(max(self.x1, 0.0), float(width)),
            min(max(self.y1, 0.0), float(height)),
            min(max(self.x2, 0.0), float(width)),
            min(max(self.y2, 0.0), float(height)),
        )

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


def iou(first: Box, second: Box) -> float:
    ix1, iy1 = max(first.x1, second.x1), max(first.y1, second.y1)
    ix2, iy2 = min(first.x2, second.x2), min(first.y2, second.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = first.area + second.area - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass(frozen=True)
class JitterPolicy:
    iou_low: float
    iou_high: float
    count: int = 3
    center_shift_fraction: float = 0.75
    log_scale_limit: float = 0.55
    log_aspect_limit: float = 0.30
    max_attempts_per_box: int = 300
    minimum_side: float = 2.0
    other_gt_iou_limit: float = 0.10
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 <= self.iou_low < self.iou_high < 1.0:
            raise ValueError("require 0 <= iou_low < iou_high < 1")
        if self.count <= 0 or self.max_attempts_per_box <= 0:
            raise ValueError("count and max_attempts_per_box must be positive")
        if any(
            float(getattr(self, name)) < 0.0
            for name in (
                "center_shift_fraction",
                "log_scale_limit",
                "log_aspect_limit",
                "minimum_side",
            )
        ):
            raise ValueError("jitter geometry parameters must be non-negative")
        if not 0.0 <= self.other_gt_iou_limit < 1.0:
            raise ValueError("other_gt_iou_limit must be in [0, 1)")


def _stable_rng(policy: JitterPolicy, stable_key: str) -> random.Random:
    digest = hashlib.sha256(f"{policy.seed}|{stable_key}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _candidate(gt: Box, rng: random.Random, policy: JitterPolicy) -> Box:
    cx, cy = (gt.x1 + gt.x2) / 2.0, (gt.y1 + gt.y2) / 2.0
    cx += rng.uniform(-policy.center_shift_fraction, policy.center_shift_fraction) * gt.width
    cy += rng.uniform(-policy.center_shift_fraction, policy.center_shift_fraction) * gt.height
    scale = rng.uniform(-policy.log_scale_limit, policy.log_scale_limit)
    aspect = rng.uniform(-policy.log_aspect_limit, policy.log_aspect_limit)
    width = gt.width * math.exp(scale + aspect)
    height = gt.height * math.exp(scale - aspect)
    return Box(cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0)


def sample_hard_negative_boxes(
    gt: Box,
    *,
    image_width: int,
    image_height: int,
    policy: JitterPolicy,
    stable_key: str,
    other_ground_truth: Sequence[Box] = (),
) -> list[Box]:
    """Return unique jitter boxes inside the requested IoU band.

    These rows must never be written back as detector background labels.
    """

    if gt.area <= 0.0 or image_width <= 0 or image_height <= 0:
        raise ValueError("invalid GT or image dimensions")
    rng = _stable_rng(policy, stable_key)
    selected: list[Box] = []
    signatures: set[tuple[int, int, int, int]] = set()
    for _ in range(policy.max_attempts_per_box * policy.count):
        if len(selected) == policy.count:
            break
        candidate = _candidate(gt, rng, policy).clip(image_width, image_height)
        if candidate.width < policy.minimum_side or candidate.height < policy.minimum_side:
            continue
        target_iou = iou(candidate, gt)
        if not policy.iou_low <= target_iou <= policy.iou_high:
            continue
        if any(iou(candidate, other) > policy.other_gt_iou_limit for other in other_ground_truth):
            continue
        signature = tuple(round(value * 4.0) for value in candidate.as_list())
        if signature in signatures:
            continue
        signatures.add(signature)
        selected.append(candidate)
    return selected


def nearest_iou(box: Box, candidates: Iterable[Box]) -> float:
    return max((iou(box, candidate) for candidate in candidates), default=0.0)


def soft_match_target(overlap: float, threshold: float, temperature: float = 0.04) -> float:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    value = (float(overlap) - float(threshold)) / temperature
    if value >= 40.0:
        return 1.0
    if value <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))
