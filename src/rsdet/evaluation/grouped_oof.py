"""Leakage-safe outer OOF and inner grouped split contracts.

The formal CV3 fold is the only allowed outer test assignment.  Inner folds
are made from source groups inside the two outer-training folds.  A candidate
may have several counterfactual actions, but all of those rows inherit the
same image, source group, and outer fold and therefore can never be split.

This module deliberately contains no model code.  Training scripts consume
the integer indices and must record :func:`split_audit` beside their outputs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class GroupedLedger:
    """One row per training example/action with immutable provenance."""

    candidate_ids: np.ndarray
    image_ids: np.ndarray
    outer_folds: np.ndarray
    group_ids: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            self.candidate_ids,
            self.image_ids,
            self.outer_folds,
            self.group_ids,
        )
        lengths = {len(value) for value in arrays}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("grouped ledger arrays must have the same non-zero length")
        _validate_single_assignment(
            keys=self.candidate_ids,
            values=self.image_ids,
            name="candidate_id -> image_id",
        )
        _validate_single_assignment(
            keys=self.candidate_ids,
            values=self.outer_folds,
            name="candidate_id -> outer_fold",
        )
        _validate_single_assignment(
            keys=self.candidate_ids,
            values=self.group_ids,
            name="candidate_id -> group_id",
        )
        _validate_single_assignment(
            keys=self.image_ids,
            values=self.outer_folds,
            name="image_id -> outer_fold",
        )
        _validate_single_assignment(
            keys=self.image_ids,
            values=self.group_ids,
            name="image_id -> group_id",
        )
        _validate_single_assignment(
            keys=self.group_ids,
            values=self.outer_folds,
            name="group_id -> outer_fold",
        )


@dataclass(frozen=True)
class OuterSplit:
    held_out_fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray


@dataclass(frozen=True)
class InnerSplit:
    inner_fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray


def _validate_single_assignment(
    *, keys: Sequence[object], values: Sequence[object], name: str
) -> None:
    assignment: dict[object, object] = {}
    for key, value in zip(keys, values, strict=True):
        previous = assignment.setdefault(key, value)
        if previous != value:
            raise ValueError(f"leakage contract violated ({name}): {key!r}")


def iter_outer_splits(ledger: GroupedLedger) -> tuple[OuterSplit, ...]:
    """Return the frozen outer folds; no random resplitting is permitted."""

    result: list[OuterSplit] = []
    for held_out in sorted({int(value) for value in ledger.outer_folds.tolist()}):
        validation = np.flatnonzero(ledger.outer_folds == held_out)
        train = np.flatnonzero(ledger.outer_folds != held_out)
        if not len(train) or not len(validation):
            raise ValueError(f"outer fold {held_out} is empty")
        result.append(OuterSplit(held_out, train, validation))
    if len(result) < 2:
        raise ValueError("at least two outer folds are required")
    return tuple(result)


def _stable_tie(value: object, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def assign_balanced_group_folds(
    *,
    group_ids: Sequence[object],
    labels: Sequence[int] | None,
    n_splits: int,
    seed: int = 2026,
) -> dict[object, int]:
    """Deterministically balance whole groups by row and positive counts."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    groups = np.asarray(group_ids, dtype=object)
    if len(groups) == 0:
        raise ValueError("group_ids must not be empty")
    y = np.zeros(len(groups), dtype=np.int64)
    if labels is not None:
        y = np.asarray(labels, dtype=np.int64)
        if len(y) != len(groups) or not np.isin(y, (0, 1)).all():
            raise ValueError("labels must be binary and aligned with group_ids")

    stats: dict[object, tuple[int, int]] = {}
    for group in dict.fromkeys(groups.tolist()):
        mask = groups == group
        stats[group] = (int(mask.sum()), int(y[mask].sum()))
    if len(stats) < n_splits:
        raise ValueError("fewer groups than inner folds")

    ordered = sorted(
        stats,
        key=lambda group: (
            -stats[group][0],
            -stats[group][1],
            _stable_tie(group, seed),
        ),
    )
    row_totals = [0] * n_splits
    positive_totals = [0] * n_splits
    assignment: dict[object, int] = {}
    for group in ordered:
        fold = min(
            range(n_splits),
            key=lambda index: (
                row_totals[index],
                positive_totals[index],
                index,
            ),
        )
        assignment[group] = fold
        n_rows, n_positive = stats[group]
        row_totals[fold] += n_rows
        positive_totals[fold] += n_positive
    return assignment


def iter_inner_group_splits(
    *,
    ledger: GroupedLedger,
    outer_train_indices: Sequence[int],
    labels: Sequence[int] | None = None,
    n_splits: int = 3,
    seed: int = 2026,
) -> tuple[InnerSplit, ...]:
    """Make deterministic source-group inner folds within outer training data."""

    outer_train = np.asarray(outer_train_indices, dtype=np.int64)
    if len(outer_train) == 0 or len(np.unique(outer_train)) != len(outer_train):
        raise ValueError("outer_train_indices must be non-empty and unique")
    local_groups = ledger.group_ids[outer_train]
    local_labels = None
    if labels is not None:
        all_labels = np.asarray(labels, dtype=np.int64)
        if len(all_labels) != len(ledger.group_ids):
            raise ValueError("labels must align with the full ledger")
        local_labels = all_labels[outer_train]
    assignment = assign_balanced_group_folds(
        group_ids=local_groups,
        labels=local_labels,
        n_splits=n_splits,
        seed=seed,
    )

    result: list[InnerSplit] = []
    local_fold = np.asarray([assignment[value] for value in local_groups], dtype=np.int64)
    for fold in range(n_splits):
        validation = outer_train[local_fold == fold]
        train = outer_train[local_fold != fold]
        if not len(train) or not len(validation):
            raise ValueError(f"inner fold {fold} is empty")
        result.append(InnerSplit(fold, train, validation))
    return tuple(result)


def split_audit(
    ledger: GroupedLedger,
    *,
    outer: Iterable[OuterSplit] | None = None,
) -> dict[str, object]:
    """Return compact provenance that can be serialized with an experiment."""

    splits = tuple(outer or iter_outer_splits(ledger))
    records: list[dict[str, object]] = []
    for split in splits:
        train_groups = set(ledger.group_ids[split.train_indices].tolist())
        val_groups = set(ledger.group_ids[split.validation_indices].tolist())
        train_candidates = set(ledger.candidate_ids[split.train_indices].tolist())
        val_candidates = set(ledger.candidate_ids[split.validation_indices].tolist())
        records.append(
            {
                "held_out_fold": split.held_out_fold,
                "n_train_rows": len(split.train_indices),
                "n_validation_rows": len(split.validation_indices),
                "n_train_groups": len(train_groups),
                "n_validation_groups": len(val_groups),
                "group_overlap": len(train_groups & val_groups),
                "candidate_overlap": len(train_candidates & val_candidates),
            }
        )
    return {
        "contract": "formal_outer_fold_plus_inner_source_group_v1",
        "n_rows": len(ledger.group_ids),
        "n_images": len(set(ledger.image_ids.tolist())),
        "n_candidates": len(set(ledger.candidate_ids.tolist())),
        "n_groups": len(set(ledger.group_ids.tolist())),
        "outer_splits": records,
    }


__all__ = [
    "GroupedLedger",
    "InnerSplit",
    "OuterSplit",
    "assign_balanced_group_folds",
    "iter_inner_group_splits",
    "iter_outer_splits",
    "split_audit",
]
