from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Sequence


class ActionKind(str, Enum):
    KEEP = "keep"
    DROP = "drop"
    RELABEL = "relabel"
    RESCORE = "rescore"


@dataclass(frozen=True, slots=True)
class Action:
    """A deployable action. No GT-derived fields are allowed here."""

    kind: ActionKind
    cls_id: int | None = None
    score: float | None = None

    def validate(self) -> None:
        if self.kind is ActionKind.RELABEL and self.cls_id is None:
            raise ValueError("RELABEL requires cls_id")
        if self.kind is ActionKind.RESCORE and self.score is None:
            raise ValueError("RESCORE requires score")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    box_xyxy: tuple[float, float, float, float]
    cls_id: int
    score: float
    payload: dict[str, object] | None = None


def apply_action(
    candidates: Sequence[Candidate],
    candidate_id: str,
    action: Action,
) -> list[Candidate]:
    """Pure, deterministic transformation used by both label building and deploy."""

    action.validate()
    out: list[Candidate] = []
    found = False
    for cand in candidates:
        if cand.candidate_id != candidate_id:
            out.append(cand)
            continue
        found = True
        if action.kind is ActionKind.KEEP:
            out.append(cand)
        elif action.kind is ActionKind.DROP:
            continue
        elif action.kind is ActionKind.RELABEL:
            out.append(replace(cand, cls_id=int(action.cls_id)))
        elif action.kind is ActionKind.RESCORE:
            out.append(replace(cand, score=float(action.score)))
        else:  # pragma: no cover
            raise AssertionError(f"Unhandled action: {action.kind}")
    if not found:
        raise KeyError(f"candidate_id={candidate_id!r} not found")
    return out


def assert_candidate_count_invariant(
    before: Iterable[Candidate],
    after: Iterable[Candidate],
    *,
    allow_drop: bool = True,
) -> None:
    before_n = sum(1 for _ in before)
    after_n = sum(1 for _ in after)
    if allow_drop:
        if after_n > before_n:
            raise AssertionError(
                f"candidate expansion is forbidden: {before_n} -> {after_n}"
            )
    elif after_n != before_n:
        raise AssertionError(f"candidate count changed: {before_n} -> {after_n}")
