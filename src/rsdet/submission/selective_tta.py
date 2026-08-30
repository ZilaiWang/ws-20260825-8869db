"""Class-routed sparse 90-degree TTA for the HERA-Guard deployment chain.

The incumbent identity view always runs.  A lightweight tile router selects a
small subset of tiles for a second 90-degree view.  Aircraft proposals from the
second view are dropped by default because the official v2 model is already
near saturation there; ship/vehicle proposals must either receive geometric
support from the identity view or pass a high quality threshold under a strict
novel-proposal budget.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

SHIP_IDS = frozenset({0, 1, 2, 3})
AIRCRAFT_IDS = frozenset(range(4, 24))
VEHICLE_IDS = frozenset({24})


@dataclass(frozen=True)
class TileRouteDecision:
    tile_index: int
    probability: float
    rank: int


def _xyxy(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
    if "bbox_xyxy" in row:
        values = row["bbox_xyxy"]
    else:
        x, y, width, height = (float(value) for value in row["bbox"])
        values = (x, y, x + width, y + height)
    box = tuple(float(value) for value in values)
    if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"invalid bbox: {box}")
    return box


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x0 = max(float(first[0]), float(second[0]))
    y0 = max(float(first[1]), float(second[1]))
    x1 = min(float(first[2]), float(second[2]))
    y1 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_first = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    area_second = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = area_first + area_second - intersection
    return intersection / union if union > 0.0 else 0.0


def tile_summary_features(
    records: Sequence[Mapping[str, Any]],
    *,
    width: int,
    height: int,
    low_score: float = 0.08,
    high_score: float = 0.30,
    edge_fraction: float = 0.08,
) -> list[float]:
    """Build deployable identity-view features for one tile.

    The summary intentionally emphasizes ship/vehicle uncertainty, small boxes,
    and internal-border candidates.  It contains no ground-truth or source ID.
    """

    if width <= 0 or height <= 0:
        raise ValueError("tile width/height must be positive")
    if not 0.0 <= low_score < high_score <= 1.0:
        raise ValueError("score band must satisfy 0 <= low < high <= 1")
    ship_scores: list[float] = []
    vehicle_scores: list[float] = []
    ship_band = vehicle_band = 0
    small_ship_vehicle = 0
    edge_ship_vehicle = 0
    for row in records:
        category = int(row["category_id"])
        score = float(row["score"])
        box = _xyxy(row)
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        relevant = category in SHIP_IDS or category in VEHICLE_IDS
        if category in SHIP_IDS:
            ship_scores.append(score)
            ship_band += int(low_score <= score <= high_score)
        elif category in VEHICLE_IDS:
            vehicle_scores.append(score)
            vehicle_band += int(low_score <= score <= high_score)
        if relevant:
            small_ship_vehicle += int(min(box_width, box_height) <= 48.0)
            edge = min(box[0], box[1], width - box[2], height - box[3])
            edge_ship_vehicle += int(edge <= edge_fraction * min(width, height))

    total = max(1, len(records))
    relevant_count = len(ship_scores) + len(vehicle_scores)
    return [
        max(ship_scores, default=0.0),
        max(vehicle_scores, default=0.0),
        float(ship_band),
        float(vehicle_band),
        float(len(ship_scores)),
        float(len(vehicle_scores)),
        float(small_ship_vehicle),
        float(edge_ship_vehicle),
        float(relevant_count) / total,
        math.log1p(total),
    ]


class SparseTTARouter(nn.Module):
    """Small tile-level utility router; output is probability of running 90°."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("router features must have shape [tiles, features]")
        return torch.sigmoid(self.network(features).squeeze(1))


def select_tta_tiles(
    probabilities: Sequence[float],
    *,
    max_fraction: float = 0.25,
    minimum_probability: float = 0.55,
    max_tiles: int | None = None,
) -> list[TileRouteDecision]:
    """Select a deterministic top utility subset under a hard compute budget."""

    if not 0.0 <= max_fraction <= 1.0:
        raise ValueError("max_fraction must be within [0, 1]")
    if not 0.0 <= minimum_probability <= 1.0:
        raise ValueError("minimum_probability must be within [0, 1]")
    if max_tiles is not None and max_tiles < 0:
        raise ValueError("max_tiles must be non-negative or None")
    n = len(probabilities)
    budget = math.ceil(n * max_fraction)
    if max_tiles is not None:
        budget = min(budget, max_tiles)
    ordered = sorted(
        ((index, float(value)) for index, value in enumerate(probabilities)),
        key=lambda item: (-item[1], item[0]),
    )
    selected = [item for item in ordered if item[1] >= minimum_probability][:budget]
    return [
        TileRouteDecision(tile_index=index, probability=probability, rank=rank)
        for rank, (index, probability) in enumerate(selected)
    ]


def accept_rotated_candidates(
    identity: Sequence[Mapping[str, Any]],
    rotated: Sequence[Mapping[str, Any]],
    *,
    support_iou: float = 0.25,
    novel_quality_threshold: float = 0.90,
    novel_budget_by_coarse: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Conservatively admit ship/vehicle proposals from the selected TTA view.

    Supported candidates are accepted when the identity view contains a
    same-fine proposal with sufficient IoU.  Unsupported candidates require a
    model-supplied ``official_match_quality`` and consume a small coarse-class
    novelty budget.  Aircraft are always excluded from the second view here.
    """

    if not 0.0 <= support_iou <= 1.0:
        raise ValueError("support_iou must be within [0, 1]")
    if not 0.0 <= novel_quality_threshold <= 1.0:
        raise ValueError("novel_quality_threshold must be within [0, 1]")
    budgets = {"ship": 8, "vehicle": 12}
    if novel_budget_by_coarse is not None:
        budgets.update({key: int(value) for key, value in novel_budget_by_coarse.items()})
    if any(value < 0 for value in budgets.values()):
        raise ValueError("novel budgets must be non-negative")

    identity_by_class: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for row in identity:
        identity_by_class[int(row["category_id"])].append(_xyxy(row))

    supported: list[dict[str, Any]] = []
    novel: dict[str, list[dict[str, Any]]] = {"ship": [], "vehicle": []}
    for raw in rotated:
        row = dict(raw)
        category = int(row["category_id"])
        if category in AIRCRAFT_IDS:
            continue
        coarse = "ship" if category in SHIP_IDS else "vehicle" if category == 24 else ""
        if not coarse:
            continue
        box = _xyxy(row)
        best_support = max(
            (box_iou(box, previous) for previous in identity_by_class.get(category, ())),
            default=0.0,
        )
        row["tta_same_fine_iou"] = best_support
        row["source_view"] = "rot90"
        if best_support >= support_iou:
            row["tta_admission"] = "same_fine_support"
            supported.append(row)
            continue
        quality = float(row.get("official_match_quality", 0.0))
        if quality >= novel_quality_threshold:
            row["tta_admission"] = "high_quality_novel"
            novel[coarse].append(row)

    output = supported
    for coarse in ("ship", "vehicle"):
        novel[coarse].sort(
            key=lambda row: (
                -float(row.get("official_match_quality", 0.0)),
                -float(row["score"]),
                int(row.get("stable_order", 0)),
            )
        )
        output.extend(novel[coarse][: budgets[coarse]])
    return output


__all__ = [
    "AIRCRAFT_IDS",
    "SHIP_IDS",
    "VEHICLE_IDS",
    "SparseTTARouter",
    "TileRouteDecision",
    "accept_rotated_candidates",
    "box_iou",
    "select_tta_tiles",
    "tile_summary_features",
]
