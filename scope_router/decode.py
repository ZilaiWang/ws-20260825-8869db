from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .actions import Action, ActionKind, Candidate, apply_action, assert_candidate_count_invariant


@dataclass(frozen=True, slots=True)
class ProposedAction:
    candidate_id: str
    action: Action
    lower_utility: float
    median_utility: float


PairwisePenalty = Callable[[ProposedAction, ProposedAction], float]


def safe_greedy_decode(
    baseline: Sequence[Candidate],
    proposals: Sequence[ProposedAction],
    *,
    pairwise_penalty: PairwisePenalty | None = None,
    min_lower_utility: float = 0.0,
    max_actions: int = 32,
    conflict_iou: Mapping[tuple[str, str], float] | None = None,
    conflict_iou_threshold: float = 0.80,
) -> tuple[list[Candidate], list[ProposedAction]]:
    """Monotone, abstaining decoder.

    The baseline is returned unchanged unless a proposal has positive calibrated
    lower-bound utility after interactions. Only one action per candidate is allowed.
    """

    ranked = sorted(
        proposals,
        key=lambda p: (p.lower_utility, p.median_utility),
        reverse=True,
    )
    selected: list[ProposedAction] = []
    selected_ids: set[str] = set()

    for proposal in ranked:
        if len(selected) >= max_actions:
            break
        if proposal.candidate_id in selected_ids:
            continue
        net = proposal.lower_utility
        if pairwise_penalty is not None:
            net -= sum(max(0.0, pairwise_penalty(proposal, other)) for other in selected)
        if conflict_iou is not None and proposal.action.kind is ActionKind.RELABEL:
            for other in selected:
                overlap = conflict_iou.get(
                    (proposal.candidate_id, other.candidate_id),
                    conflict_iou.get((other.candidate_id, proposal.candidate_id), 0.0),
                )
                if overlap >= conflict_iou_threshold:
                    net -= abs(proposal.lower_utility)  # conservative veto
                    break
        if net <= min_lower_utility:
            continue
        selected.append(proposal)
        selected_ids.add(proposal.candidate_id)

    output = list(baseline)
    for proposal in selected:
        output = apply_action(output, proposal.candidate_id, proposal.action)
    assert_candidate_count_invariant(baseline, output, allow_drop=True)
    return output, selected
