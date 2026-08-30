import numpy as np
import pytest

from rsdet.evaluation.grouped_oof import (
    GroupedLedger,
    assign_balanced_group_folds,
    iter_inner_group_splits,
    iter_outer_splits,
    split_audit,
)


def _ledger() -> GroupedLedger:
    candidate_ids = []
    image_ids = []
    outer_folds = []
    group_ids = []
    for fold in range(3):
        for group_offset in range(3):
            group = f"f{fold}-g{group_offset}"
            for candidate_offset in range(2):
                candidate = f"{group}-c{candidate_offset}"
                for _action in range(2):
                    candidate_ids.append(candidate)
                    image_ids.append(f"{group}-image")
                    outer_folds.append(fold)
                    group_ids.append(group)
    return GroupedLedger(
        candidate_ids=np.asarray(candidate_ids, dtype=object),
        image_ids=np.asarray(image_ids, dtype=object),
        outer_folds=np.asarray(outer_folds),
        group_ids=np.asarray(group_ids, dtype=object),
    )


def test_outer_splits_keep_groups_and_candidate_actions_together() -> None:
    ledger = _ledger()
    for split in iter_outer_splits(ledger):
        assert not (
            set(ledger.group_ids[split.train_indices])
            & set(ledger.group_ids[split.validation_indices])
        )
        assert not (
            set(ledger.candidate_ids[split.train_indices])
            & set(ledger.candidate_ids[split.validation_indices])
        )
    audit = split_audit(ledger)
    assert all(row["group_overlap"] == 0 for row in audit["outer_splits"])
    assert all(row["candidate_overlap"] == 0 for row in audit["outer_splits"])


def test_inner_splits_are_grouped_and_deterministic() -> None:
    ledger = _ledger()
    outer = iter_outer_splits(ledger)[0]
    labels = np.asarray([index % 2 for index in range(len(ledger.group_ids))])
    first = iter_inner_group_splits(
        ledger=ledger,
        outer_train_indices=outer.train_indices,
        labels=labels,
        n_splits=3,
    )
    second = iter_inner_group_splits(
        ledger=ledger,
        outer_train_indices=outer.train_indices,
        labels=labels,
        n_splits=3,
    )
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.validation_indices, right.validation_indices)
        assert not (
            set(ledger.group_ids[left.train_indices])
            & set(ledger.group_ids[left.validation_indices])
        )


def test_conflicting_candidate_or_group_assignment_is_rejected() -> None:
    with pytest.raises(ValueError, match="candidate_id -> image_id"):
        GroupedLedger(
            candidate_ids=np.asarray(["c", "c"], dtype=object),
            image_ids=np.asarray([1, 2]),
            outer_folds=np.asarray([0, 0]),
            group_ids=np.asarray(["g", "g"], dtype=object),
        )
    with pytest.raises(ValueError, match="group_id -> outer_fold"):
        GroupedLedger(
            candidate_ids=np.asarray(["a", "b"], dtype=object),
            image_ids=np.asarray([1, 2]),
            outer_folds=np.asarray([0, 1]),
            group_ids=np.asarray(["g", "g"], dtype=object),
        )


def test_balancing_requires_enough_groups() -> None:
    with pytest.raises(ValueError, match="fewer groups"):
        assign_balanced_group_folds(
            group_ids=["a", "b"],
            labels=[0, 1],
            n_splits=3,
        )
