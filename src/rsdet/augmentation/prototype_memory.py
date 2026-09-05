"""Source-safe visual prototype memory inspired by PET-DINO."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PrototypeEntry:
    vector: np.ndarray
    class_id: int
    role: str
    source_group: str
    scale_bin: str


class PrototypeMemoryBank:
    """FIFO memory with class/role/scale keys and source exclusion."""

    def __init__(
        self, *, dimension: int, capacity_per_key: int = 64, epsilon: float = 1e-12
    ) -> None:
        if dimension <= 0 or capacity_per_key <= 0 or epsilon <= 0:
            raise ValueError("invalid memory-bank parameters")
        self.dimension = int(dimension)
        self.capacity_per_key = int(capacity_per_key)
        self.epsilon = float(epsilon)
        self._entries: dict[tuple[int, str, str], deque[PrototypeEntry]] = defaultdict(
            lambda: deque(maxlen=self.capacity_per_key)
        )

    def _normalize(self, vector: np.ndarray | Iterable[float]) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float32).reshape(-1)
        if value.shape != (self.dimension,):
            raise ValueError(f"prototype has shape {value.shape}, expected {(self.dimension,)}")
        norm = float(np.linalg.norm(value))
        if not np.isfinite(norm) or norm <= self.epsilon:
            raise ValueError("prototype must be finite and non-zero")
        return value / norm

    def update(
        self,
        vector: np.ndarray | Iterable[float],
        *,
        class_id: int,
        role: str,
        source_group: str,
        scale_bin: str,
    ) -> None:
        if role not in {"positive", "hard_negative", "active_fp"}:
            raise ValueError(f"unsupported role: {role}")
        if not source_group or not scale_bin:
            raise ValueError("source_group and scale_bin must be non-empty")
        key = (int(class_id), role, scale_bin)
        self._entries[key].append(
            PrototypeEntry(
                vector=self._normalize(vector),
                class_id=int(class_id),
                role=role,
                source_group=str(source_group),
                scale_bin=str(scale_bin),
            )
        )

    def aggregate(
        self,
        *,
        class_id: int,
        role: str,
        scale_bin: str,
        exclude_source_group: str | None = None,
    ) -> np.ndarray | None:
        rows = [
            entry.vector
            for entry in self._entries.get((int(class_id), role, scale_bin), ())
            if exclude_source_group is None or entry.source_group != exclude_source_group
        ]
        if not rows:
            return None
        mean = np.stack(rows).mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return None if not np.isfinite(norm) or norm <= self.epsilon else mean / norm

    def boundary_score(
        self,
        vector: np.ndarray | Iterable[float],
        *,
        class_id: int,
        scale_bin: str,
        exclude_source_group: str | None = None,
        negative_roles: tuple[str, ...] = ("hard_negative", "active_fp"),
        temperature: float = 0.07,
    ) -> float | None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        query = self._normalize(vector)
        positive = self.aggregate(
            class_id=class_id,
            role="positive",
            scale_bin=scale_bin,
            exclude_source_group=exclude_source_group,
        )
        negatives = [
            candidate
            for role in negative_roles
            if (
                candidate := self.aggregate(
                    class_id=class_id,
                    role=role,
                    scale_bin=scale_bin,
                    exclude_source_group=exclude_source_group,
                )
            )
            is not None
        ]
        if positive is None or not negatives:
            return None
        logit = (
            float(query @ positive) - max(float(query @ row) for row in negatives)
        ) / temperature
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))

    def counts(self) -> dict[str, int]:
        return {
            f"{class_id}:{role}:{scale_bin}": len(entries)
            for (class_id, role, scale_bin), entries in sorted(self._entries.items())
        }
