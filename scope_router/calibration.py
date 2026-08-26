from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class _CalibrationBucket:
    correction: float
    n: int


class GroupwiseConformalLCB:
    """Split-conformal correction for predicted utility lower bounds.

    Fit on an untouched calibration fold. For each action, residual is
    predicted_lcb - realized_delta. Subtracting its (1-alpha) quantile creates a
    conservative lower bound. Small class/domain buckets fall back to global.
    """

    def __init__(self, *, alpha: float = 0.10, min_group_size: int = 80) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0,1)")
        self.alpha = alpha
        self.min_group_size = min_group_size
        self.global_bucket: _CalibrationBucket | None = None
        self.group_buckets: dict[Hashable, _CalibrationBucket] = {}

    @staticmethod
    def _higher_quantile(values: np.ndarray, q: float) -> float:
        if values.size == 0:
            raise ValueError("cannot calibrate on an empty array")
        try:
            return float(np.quantile(values, q, method="higher"))
        except TypeError:  # NumPy < 1.22
            return float(np.quantile(values, q, interpolation="higher"))

    def fit(
        self,
        predicted_lcb: Iterable[float],
        realized_delta: Iterable[float],
        groups: Iterable[Hashable] | None = None,
    ) -> "GroupwiseConformalLCB":
        pred = np.asarray(list(predicted_lcb), dtype=np.float64)
        target = np.asarray(list(realized_delta), dtype=np.float64)
        if pred.shape != target.shape:
            raise ValueError("predicted_lcb and realized_delta must have same shape")
        residual = pred - target
        # Finite-sample split-conformal quantile.
        n = residual.size
        q = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.global_bucket = _CalibrationBucket(
            correction=self._higher_quantile(residual, q), n=n
        )
        self.group_buckets.clear()

        if groups is None:
            return self
        group_arr = np.asarray(list(groups), dtype=object)
        if group_arr.shape != pred.shape:
            raise ValueError("groups must have same shape as predictions")
        for group in dict.fromkeys(group_arr.tolist()):
            idx = group_arr == group
            count = int(idx.sum())
            if count < self.min_group_size:
                continue
            qg = min(1.0, np.ceil((count + 1) * (1.0 - self.alpha)) / count)
            self.group_buckets[group] = _CalibrationBucket(
                correction=self._higher_quantile(residual[idx], qg), n=count
            )
        return self

    def transform(
        self,
        predicted_lcb: Iterable[float],
        groups: Iterable[Hashable] | None = None,
    ) -> np.ndarray:
        if self.global_bucket is None:
            raise RuntimeError("fit must be called first")
        pred = np.asarray(list(predicted_lcb), dtype=np.float64)
        if groups is None:
            return pred - self.global_bucket.correction
        group_arr = np.asarray(list(groups), dtype=object)
        if group_arr.shape != pred.shape:
            raise ValueError("groups must have same shape as predictions")
        out = np.empty_like(pred)
        for i, (value, group) in enumerate(zip(pred, group_arr, strict=True)):
            bucket = self.group_buckets.get(group, self.global_bucket)
            out[i] = value - bucket.correction
        return out

    def accept(
        self,
        predicted_lcb: Iterable[float],
        *,
        groups: Iterable[Hashable] | None = None,
        margin: float = 0.0,
    ) -> np.ndarray:
        return self.transform(predicted_lcb, groups) > margin
