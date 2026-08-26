from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol, Sequence

from .actions import Action, Candidate, apply_action


class ExactScorer(Protocol):
    """Adapter around the repository's exact competition evaluator."""

    def __call__(
        self,
        predictions: Sequence[Candidate],
        ground_truth: object,
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class CounterfactualRecord:
    group_id: str
    candidate_id: str
    action_key: str
    base_utility: float
    action_utility: float
    delta_utility: float
    detector_model_id: str
    detector_train_groups_hash: str
    prediction_group_id: str
    scorer_version: str


def action_key(action: Action) -> str:
    payload = {
        "kind": action.kind.value,
        "cls_id": action.cls_id,
        "score": action.score,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _hash_groups(groups: Iterable[str]) -> str:
    text = "\n".join(sorted(set(groups)))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CounterfactualLabelBuilder:
    """Build exact metric-delta labels from strictly out-of-fold predictions.

    GT is intentionally required here, but this module must never be imported by
    the deploy package. Each prediction group must be disjoint from the detector's
    training groups. The resulting labels can then train a GT-blind controller.
    """

    def __init__(
        self,
        scorer: ExactScorer,
        *,
        scorer_version: str,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.scorer = scorer
        self.scorer_version = scorer_version
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def build_group(
        self,
        *,
        group_id: str,
        predictions: Sequence[Candidate],
        ground_truth: object,
        actions_by_candidate: Mapping[str, Sequence[Action]],
        detector_model_id: str,
        detector_train_groups: Iterable[str],
    ) -> list[CounterfactualRecord]:
        train_groups = set(detector_train_groups)
        if group_id in train_groups:
            raise ValueError(
                f"OOF violation: prediction group {group_id!r} was used to train detector"
            )
        train_hash = _hash_groups(train_groups)
        base = float(self.scorer(predictions, ground_truth))
        rows: list[CounterfactualRecord] = []

        known_ids = {p.candidate_id for p in predictions}
        unknown = set(actions_by_candidate) - known_ids
        if unknown:
            raise KeyError(f"actions reference unknown candidates: {sorted(unknown)[:5]}")

        for candidate_id, actions in actions_by_candidate.items():
            for action in actions:
                key = action_key(action)
                cached = self._read_cache(group_id, candidate_id, key)
                if cached is None:
                    edited = apply_action(predictions, candidate_id, action)
                    utility = float(self.scorer(edited, ground_truth))
                    self._write_cache(group_id, candidate_id, key, utility)
                else:
                    utility = cached
                rows.append(
                    CounterfactualRecord(
                        group_id=group_id,
                        candidate_id=candidate_id,
                        action_key=key,
                        base_utility=base,
                        action_utility=utility,
                        delta_utility=utility - base,
                        detector_model_id=detector_model_id,
                        detector_train_groups_hash=train_hash,
                        prediction_group_id=group_id,
                        scorer_version=self.scorer_version,
                    )
                )
        return rows

    def build_pairwise_interactions(
        self,
        *,
        predictions: Sequence[Candidate],
        ground_truth: object,
        first: tuple[str, Action],
        second: tuple[str, Action],
    ) -> float:
        """Second-order interaction term Δ_ij.

        Use only for top ambiguous candidates; enumerating all pairs is unnecessary.
        """

        cid_i, action_i = first
        cid_j, action_j = second
        if cid_i == cid_j:
            raise ValueError("pairwise actions must target different candidates")
        u0 = float(self.scorer(predictions, ground_truth))
        pi = apply_action(predictions, cid_i, action_i)
        pj = apply_action(predictions, cid_j, action_j)
        pij = apply_action(pi, cid_j, action_j)
        ui = float(self.scorer(pi, ground_truth))
        uj = float(self.scorer(pj, ground_truth))
        uij = float(self.scorer(pij, ground_truth))
        return uij - ui - uj + u0

    def _cache_path(self, group_id: str, candidate_id: str, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(
            f"{self.scorer_version}|{group_id}|{candidate_id}|{key}".encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, group_id: str, candidate_id: str, key: str) -> float | None:
        path = self._cache_path(group_id, candidate_id, key)
        if path is None or not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return float(payload["utility"])

    def _write_cache(
        self,
        group_id: str,
        candidate_id: str,
        key: str,
        utility: float,
    ) -> None:
        path = self._cache_path(group_id, candidate_id, key)
        if path is None:
            return
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"utility": utility}, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(path)


def write_jsonl(records: Iterable[CounterfactualRecord], path: str | Path) -> None:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")
