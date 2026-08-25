from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class OOFRecord:
    sample_id: str
    group_id: str
    detector_model_id: str
    detector_train_groups: frozenset[str]
    router_model_id: str | None = None
    router_train_groups: frozenset[str] | None = None
    split_manifest_hash: str | None = None


def validate_cross_fit(records: Iterable[OOFRecord], *, require_router: bool = False) -> None:
    errors: list[str] = []
    for row in records:
        if row.group_id in row.detector_train_groups:
            errors.append(
                f"{row.sample_id}: detector saw evaluation group {row.group_id}"
            )
        if require_router:
            if row.router_train_groups is None:
                errors.append(f"{row.sample_id}: missing router_train_groups")
            elif row.group_id in row.router_train_groups:
                errors.append(
                    f"{row.sample_id}: router saw evaluation group {row.group_id}"
                )
    if errors:
        preview = "\n".join(errors[:20])
        raise AssertionError(f"Cross-fitting provenance violations ({len(errors)}):\n{preview}")
