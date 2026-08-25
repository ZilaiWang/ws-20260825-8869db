"""Label-hierarchy definitions used by the BHCL objective.

The root node is deliberately not represented as a trainable hierarchy level,
matching the convention in Eq. (7) of the BHCL paper.  This module has no
PyTorch import at module load time so that dataset/configuration tooling can
use the hierarchy in environments where PyTorch is not installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import accumulate
from typing import Any


@dataclass(frozen=True)
class HierarchySpec:
    """A compact, immutable fine-label to ancestor-label mapping.

    ``fine_to_level`` contains one row per non-root level.  Entries in a row
    are local category indices for every fine label.  Public level indices are
    zero-based; level index ``0`` therefore corresponds to ``l=1`` in the
    paper.

    The final level is required to be the fine-label level and consequently
    maps fine label ``k`` to category ``k``.  ``to(device)`` is the only method
    that imports PyTorch and returns the mapping as a ``long`` tensor with
    shape ``[num_levels, num_fine_classes]``.
    """

    root_name: str
    level_names: tuple[str, ...]
    level_categories: tuple[tuple[str, ...], ...]
    fine_to_level: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.root_name, str) or not self.root_name.strip():
            raise ValueError("root_name must be a non-empty string")
        if not self.level_names:
            raise ValueError("the hierarchy must contain at least one non-root level")
        if len(self.level_names) != len(self.level_categories):
            raise ValueError("level_names and level_categories must have equal length")
        if len(self.level_names) != len(self.fine_to_level):
            raise ValueError("fine_to_level must contain one row per non-root level")
        if len(set(self.level_names)) != len(self.level_names):
            raise ValueError("level_names must be unique")

        fine_count = len(self.level_categories[-1])
        if fine_count == 0:
            raise ValueError("the fine level must contain at least one category")

        for level_index, (categories, mapping) in enumerate(
            zip(self.level_categories, self.fine_to_level, strict=True)
        ):
            if not categories:
                raise ValueError(f"level {level_index} has no categories")
            if len(set(categories)) != len(categories):
                raise ValueError(f"categories at level {level_index} must be unique")
            if len(mapping) != fine_count:
                raise ValueError(
                    f"fine_to_level row {level_index} must contain {fine_count} entries"
                )
            category_count = len(categories)
            if any(
                not isinstance(category_id, int)
                or isinstance(category_id, bool)
                or not 0 <= category_id < category_count
                for category_id in mapping
            ):
                raise ValueError(
                    f"fine_to_level row {level_index} contains an invalid category index"
                )

        expected_fine_mapping = tuple(range(fine_count))
        if self.fine_to_level[-1] != expected_fine_mapping:
            raise ValueError("the final hierarchy level must map each fine label to itself")

    @property
    def num_levels(self) -> int:
        """Number of non-root levels (``L`` in the paper)."""

        return len(self.level_names)

    @property
    def num_fine_classes(self) -> int:
        return len(self.level_categories[-1])

    @property
    def fine_names(self) -> tuple[str, ...]:
        return self.level_categories[-1]

    @property
    def num_categories_per_level(self) -> tuple[int, ...]:
        return tuple(len(categories) for categories in self.level_categories)

    @property
    def level_offsets(self) -> tuple[int, ...]:
        """Starting row of every level in a flattened prototype bank."""

        sizes = self.num_categories_per_level
        return (0, *tuple(accumulate(sizes[:-1])))

    @property
    def num_nodes(self) -> int:
        """Total non-root category nodes, hence number of prototypes."""

        return sum(self.num_categories_per_level)

    @property
    def level_weights(self) -> tuple[float, ...]:
        """Return the normalized level penalties :math:`lambda_l` from Eq. (7)."""

        level_count = self.num_levels
        unnormalized = tuple(
            math.exp(1.0 / (level_count + 1 - paper_level))
            for paper_level in range(1, level_count + 1)
        )
        denominator = sum(unnormalized)
        return tuple(weight / denominator for weight in unnormalized)

    def category_for_fine(self, fine_label: int, level: int) -> int:
        """Return the local ancestor category for one fine label."""

        if not 0 <= level < self.num_levels:
            raise IndexError(f"level must be in [0, {self.num_levels})")
        if not 0 <= fine_label < self.num_fine_classes:
            raise IndexError(f"fine_label must be in [0, {self.num_fine_classes})")
        return self.fine_to_level[level][fine_label]

    def flat_node_index(self, level: int, category: int) -> int:
        """Map a level-local category index to a flattened prototype row."""

        if not 0 <= level < self.num_levels:
            raise IndexError(f"level must be in [0, {self.num_levels})")
        category_count = self.num_categories_per_level[level]
        if not 0 <= category < category_count:
            raise IndexError(f"category must be in [0, {category_count})")
        return self.level_offsets[level] + category

    def to(self, device: Any = None):
        """Build the ``[L, K]`` fine-to-level mapping tensor lazily."""

        try:
            import torch
        except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
            raise ImportError("HierarchySpec.to() requires PyTorch") from exc
        return torch.tensor(self.fine_to_level, dtype=torch.long, device=device)


_XH_FINE_NAMES = (
    "HM",
    "LQS",
    "QHS",
    "MS",
    "A1_SU-35",
    "A2_C-130",
    "A3_C-17",
    "A4_C-5",
    "A5_F-16",
    "A6_TU-160",
    "A7_E-3",
    "A8_B-52",
    "A9_P-3C",
    "A10_B-1B",
    "A11_E-8",
    "A12_TU-22",
    "A13_F-15",
    "A14_KC-135",
    "A15_F-22",
    "A16_FA-18",
    "A17_TU-95",
    "A18_KC-10",
    "A19_SU-34",
    "A20_SU-24",
    "FSC",
)


XH_HIERARCHY = HierarchySpec(
    root_name="root",
    level_names=("coarse", "fine"),
    level_categories=(
        ("ship", "aircraft", "vehicle"),
        _XH_FINE_NAMES,
    ),
    fine_to_level=(
        (0,) * 4 + (1,) * 20 + (2,),
        tuple(range(25)),
    ),
)


__all__ = ["HierarchySpec", "XH_HIERARCHY"]
