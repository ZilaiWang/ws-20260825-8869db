from collections import Counter

from rsdet.data.airport_proxy_split import (
    AirportProxyGroup,
    solve_grouped_validation_partition,
)


def _group(
    group_id: str,
    size: int,
    classes: dict[int, int],
    *,
    old_train: int,
) -> AirportProxyGroup:
    return AirportProxyGroup(
        group_id=group_id,
        image_ids=tuple(range(size)),
        class_box_counts=Counter(classes),
        old_train_images=old_train,
        old_val_images=size - old_train,
    )


def test_solver_respects_budget_coverage_and_is_deterministic() -> None:
    groups = [
        _group("g1", 2, {4: 8, 5: 1}, old_train=2),
        _group("g2", 3, {4: 1, 5: 8}, old_train=2),
        _group("g3", 2, {4: 3, 5: 3}, old_train=1),
        _group("g4", 3, {4: 2, 5: 2}, old_train=3),
    ]
    selected_a, metadata_a = solve_grouped_validation_partition(
        groups,
        class_ids=(4, 5),
        target_val_images=5,
        val_fraction=0.5,
    )
    selected_b, metadata_b = solve_grouped_validation_partition(
        list(reversed(groups)),
        class_ids=(4, 5),
        target_val_images=5,
        val_fraction=0.5,
    )

    assert selected_a == selected_b
    assert sum(group.image_count for group in groups if group.group_id in selected_a) == 5
    for class_id in (4, 5):
        assert (
            sum(
                group.class_box_counts[class_id] > 0
                for group in groups
                if group.group_id in selected_a
            )
            >= 2
        )
        assert any(
            group.class_box_counts[class_id] > 0
            for group in groups
            if group.group_id not in selected_a
        )
    assert metadata_a["selected_val_group_count"] == len(selected_a)
    assert metadata_a["objective"] == metadata_b["objective"]


def test_solver_rejects_class_present_in_only_one_group() -> None:
    groups = [
        _group("g1", 2, {4: 2, 5: 1}, old_train=1),
        _group("g2", 2, {4: 2}, old_train=1),
    ]

    try:
        solve_grouped_validation_partition(
            groups,
            class_ids=(4, 5),
            target_val_images=2,
            val_fraction=0.5,
        )
    except ValueError as error:
        assert "fewer than two proxy groups" in str(error)
    else:
        raise AssertionError("expected an impossible grouped split to be rejected")
