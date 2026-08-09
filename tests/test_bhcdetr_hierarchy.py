"""Pure-Python tests for the XH label hierarchy."""

from __future__ import annotations

import math

import pytest

from rsdet.models.hierarchy import XH_HIERARCHY, HierarchySpec


def test_xh_hierarchy_excludes_root_and_maps_all_25_fine_classes() -> None:
    hierarchy = XH_HIERARCHY

    assert hierarchy.root_name == "root"
    assert hierarchy.level_names == ("coarse", "fine")
    assert hierarchy.num_levels == 2
    assert hierarchy.num_fine_classes == 25
    assert hierarchy.num_categories_per_level == (3, 25)
    assert hierarchy.num_nodes == 28
    assert hierarchy.level_categories[0] == ("ship", "aircraft", "vehicle")
    assert hierarchy.fine_to_level[0] == (0,) * 4 + (1,) * 20 + (2,)
    assert hierarchy.fine_to_level[1] == tuple(range(25))


def test_level_weights_follow_equation_7_and_favour_the_leaf_level() -> None:
    coarse_raw = math.exp(1.0 / 2.0)
    fine_raw = math.exp(1.0)
    denominator = coarse_raw + fine_raw

    assert XH_HIERARCHY.level_weights == pytest.approx(
        (coarse_raw / denominator, fine_raw / denominator)
    )
    assert sum(XH_HIERARCHY.level_weights) == pytest.approx(1.0)
    assert XH_HIERARCHY.level_weights[1] > XH_HIERARCHY.level_weights[0]


def test_flat_node_indices_are_level_local_and_root_is_not_allocated() -> None:
    hierarchy = XH_HIERARCHY

    assert hierarchy.level_offsets == (0, 3)
    assert hierarchy.flat_node_index(0, 2) == 2
    assert hierarchy.flat_node_index(1, 0) == 3
    assert hierarchy.flat_node_index(1, 24) == 27


def test_hierarchy_rejects_a_non_identity_leaf_mapping() -> None:
    with pytest.raises(ValueError, match="final hierarchy level"):
        HierarchySpec(
            root_name="root",
            level_names=("fine",),
            level_categories=(("a", "b"),),
            fine_to_level=((1, 0),),
        )
