"""Fixed-risk labels for HERA/PAV training and resolver auditing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rsdet.evaluation.official_frontier import OfficialFrontierResult

WorkpointRole = Literal["protected_tp", "active_fp", "inactive_tail"]


@dataclass(frozen=True)
class WorkpointLabel:
    candidate_id: int
    role: WorkpointRole
    is_selected: bool
    is_official_tp: bool


def build_workpoint_labels(
    result: OfficialFrontierResult,
    *,
    target_fdr: float,
) -> dict[int, WorkpointLabel]:
    """Label the active prefix without calling all tail FPs ``active_fp``."""

    if target_fdr not in result.points:
        raise ValueError(f"target FDR was not scanned: {target_fdr}")
    selected = set(result.selected_candidate_ids[target_fdr])
    active_tp = set(result.selected_tp_candidate_ids[target_fdr])
    active_fp = set(result.selected_fp_candidate_ids[target_fdr])
    if active_tp & active_fp or selected != active_tp | active_fp:
        raise RuntimeError("inconsistent fixed-risk candidate ledger")

    all_ids: set[int] = set()
    official_tp = {int(match.prediction_index) for match in result.trace.matches}
    for predictions in result.kept_predictions.values():
        all_ids.update(int(row["source_prediction_index"]) for row in predictions)
    labels: dict[int, WorkpointLabel] = {}
    for candidate_id in sorted(all_ids):
        if candidate_id in active_tp:
            role: WorkpointRole = "protected_tp"
        elif candidate_id in active_fp:
            role = "active_fp"
        else:
            role = "inactive_tail"
        labels[candidate_id] = WorkpointLabel(
            candidate_id=candidate_id,
            role=role,
            is_selected=candidate_id in selected,
            is_official_tp=candidate_id in official_tp,
        )
    return labels


__all__ = ["WorkpointLabel", "WorkpointRole", "build_workpoint_labels"]
