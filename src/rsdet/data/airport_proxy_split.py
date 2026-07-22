"""Build a grouped development split from MAR20 airport-proxy assignments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class AirportProxyGroup:
    """Aggregated evidence needed to assign one indivisible proxy group."""

    group_id: str
    image_ids: tuple[int, ...]
    class_box_counts: Counter[int]
    old_train_images: int
    old_val_images: int

    @property
    def image_count(self) -> int:
        return len(self.image_ids)


def solve_grouped_validation_partition(
    groups: list[AirportProxyGroup],
    *,
    class_ids: tuple[int, ...],
    target_val_images: int,
    val_fraction: float,
    preservation_weight: float = 1e-4,
) -> tuple[set[str], dict[str, object]]:
    """Choose validation groups with class coverage and an exact image budget.

    The objective minimizes normalized per-class box-count deviation from the
    requested validation fraction. A small secondary term preserves as much of
    the previous development split as possible. Every class must remain in at
    least one training group; classes occurring in three or more source groups
    must appear in at least two validation groups.
    """

    if not groups:
        raise ValueError("groups must not be empty")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")

    ordered = sorted(groups, key=lambda item: item.group_id)
    group_count = len(ordered)
    class_count = len(class_ids)
    variable_count = group_count + class_count

    total_images = sum(group.image_count for group in ordered)
    if not 0 < target_val_images < total_images:
        raise ValueError("target_val_images must leave non-empty train and val")

    class_totals: Counter[int] = Counter()
    for group in ordered:
        class_totals.update(group.class_box_counts)
    missing_classes = [class_id for class_id in class_ids if class_totals[class_id] == 0]
    if missing_classes:
        raise ValueError(f"classes absent from all groups: {missing_classes}")

    objective = np.zeros(variable_count, dtype=np.float64)
    for index, group in enumerate(ordered):
        # Constant old_val_images is omitted. This coefficient is the extra
        # number of changed images if the group is assigned to validation.
        objective[index] = preservation_weight * (
            group.old_train_images - group.old_val_images
        )
        # Stable, negligible tie-breaker independent of solver traversal order.
        objective[index] += (index + 1) * 1e-10
    for offset, class_id in enumerate(class_ids):
        target_boxes = class_totals[class_id] * val_fraction
        objective[group_count + offset] = 1.0 / max(target_boxes, 20.0)

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    image_budget = np.zeros(variable_count, dtype=np.float64)
    image_budget[:group_count] = [group.image_count for group in ordered]
    rows.append(image_budget)
    lower.append(float(target_val_images))
    upper.append(float(target_val_images))

    class_contract: dict[str, dict[str, int | float]] = {}
    for offset, class_id in enumerate(class_ids):
        presence = np.asarray(
            [float(group.class_box_counts[class_id] > 0) for group in ordered]
        )
        source_group_count = int(presence.sum())
        if source_group_count < 2:
            raise ValueError(
                f"class {class_id} occurs in fewer than two proxy groups; "
                "group-isolated train/val is impossible"
            )
        minimum_val_groups = 2 if source_group_count >= 3 else 1

        coverage = np.zeros(variable_count, dtype=np.float64)
        coverage[:group_count] = presence
        rows.append(coverage)
        lower.append(float(minimum_val_groups))
        upper.append(float(source_group_count - 1))

        box_counts = np.asarray(
            [float(group.class_box_counts[class_id]) for group in ordered]
        )
        target_boxes = class_totals[class_id] * val_fraction

        positive_deviation = np.zeros(variable_count, dtype=np.float64)
        positive_deviation[:group_count] = box_counts
        positive_deviation[group_count + offset] = -1.0
        rows.append(positive_deviation)
        lower.append(-np.inf)
        upper.append(float(target_boxes))

        negative_deviation = np.zeros(variable_count, dtype=np.float64)
        negative_deviation[:group_count] = -box_counts
        negative_deviation[group_count + offset] = -1.0
        rows.append(negative_deviation)
        lower.append(-np.inf)
        upper.append(float(-target_boxes))

        class_contract[str(class_id)] = {
            "total_boxes": int(class_totals[class_id]),
            "target_val_boxes": float(target_boxes),
            "source_group_count": source_group_count,
            "minimum_val_groups": minimum_val_groups,
            "minimum_train_groups": 1,
        }

    bounds = Bounds(
        np.zeros(variable_count, dtype=np.float64),
        np.concatenate(
            [
                np.ones(group_count, dtype=np.float64),
                np.full(class_count, np.inf, dtype=np.float64),
            ]
        ),
    )
    integrality = np.concatenate(
        [
            np.ones(group_count, dtype=np.int32),
            np.zeros(class_count, dtype=np.int32),
        ]
    )
    result = milp(
        objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(
            np.vstack(rows),
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
        ),
        options={"time_limit": 60.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"grouped split optimization failed: {result.message}")

    selected = np.rint(result.x[:group_count]).astype(np.int32)
    val_groups = {
        group.group_id for group, is_val in zip(ordered, selected, strict=True) if is_val
    }
    actual_val_images = sum(
        group.image_count for group in ordered if group.group_id in val_groups
    )
    if actual_val_images != target_val_images:
        raise RuntimeError(
            f"solver returned {actual_val_images} val images, expected {target_val_images}"
        )

    metadata: dict[str, object] = {
        "solver": "scipy.optimize.milp_highs",
        "status": int(result.status),
        "message": str(result.message),
        "objective": float(result.fun),
        "target_val_images": target_val_images,
        "selected_val_group_count": len(val_groups),
        "class_contract": class_contract,
    }
    return val_groups, metadata
