"""Frozen-embedding APEX proposal validity head.

This is the fast-screen implementation used before any expensive end-to-end
fine-tuning.  It deliberately keeps detector boxes, fine labels and scores
immutable; the learned probability can only admit low-score tail proposals.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

SCALE_BINS = ("tiny", "small", "medium", "large")
CATEGORY_IDS = (0, 1, 2, 3, 24)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("embeddings must be a finite matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def _prototype_role(row: Mapping[str, Any], target: int) -> str:
    if int(target) == 1:
        return "positive"
    return "jitter_negative" if row.get("role") == "jitter_hard_negative" else "active_fp"


def _key(row: Mapping[str, Any], target: int) -> tuple[int, str, str]:
    return int(row["category_id"]), str(row["scale_bin"]), _prototype_role(row, target)


def _coarse_key(row: Mapping[str, Any], target: int) -> tuple[str, str, str]:
    return str(row["coarse"]), str(row["scale_bin"]), _prototype_role(row, target)


@dataclass
class PrototypeTable:
    fine_sums: dict[tuple[int, str, str], np.ndarray]
    fine_weights: dict[tuple[int, str, str], float]
    fine_source_sums: dict[tuple[tuple[int, str, str], str], np.ndarray]
    fine_source_weights: dict[tuple[tuple[int, str, str], str], float]
    coarse_sums: dict[tuple[str, str, str], np.ndarray]
    coarse_weights: dict[tuple[str, str, str], float]
    coarse_source_sums: dict[tuple[tuple[str, str, str], str], np.ndarray]
    coarse_source_weights: dict[tuple[tuple[str, str, str], str], float]

    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        rows: Sequence[Mapping[str, Any]],
        targets: Sequence[int],
        weights: Sequence[float],
    ) -> "PrototypeTable":
        normalized = _unit_rows(embeddings)
        fine_sums: dict[tuple[int, str, str], np.ndarray] = {}
        fine_weights: dict[tuple[int, str, str], float] = defaultdict(float)
        fine_source_sums: dict[tuple[tuple[int, str, str], str], np.ndarray] = {}
        fine_source_weights: dict[tuple[tuple[int, str, str], str], float] = defaultdict(float)
        coarse_sums: dict[tuple[str, str, str], np.ndarray] = {}
        coarse_weights: dict[tuple[str, str, str], float] = defaultdict(float)
        coarse_source_sums: dict[tuple[tuple[str, str, str], str], np.ndarray] = {}
        coarse_source_weights: dict[tuple[tuple[str, str, str], str], float] = defaultdict(float)

        def add(target_dict: dict[Any, np.ndarray], key: Any, value: np.ndarray) -> None:
            if key not in target_dict:
                target_dict[key] = np.zeros_like(value, dtype=np.float64)
            target_dict[key] += value

        for vector, row, target, weight in zip(normalized, rows, targets, weights, strict=True):
            numeric_weight = float(weight)
            source = str(row["source_group"])
            fine_key = _key(row, target)
            coarse_key = _coarse_key(row, target)
            add(fine_sums, fine_key, vector * numeric_weight)
            fine_weights[fine_key] += numeric_weight
            add(fine_source_sums, (fine_key, source), vector * numeric_weight)
            fine_source_weights[(fine_key, source)] += numeric_weight
            add(coarse_sums, coarse_key, vector * numeric_weight)
            coarse_weights[coarse_key] += numeric_weight
            add(coarse_source_sums, (coarse_key, source), vector * numeric_weight)
            coarse_source_weights[(coarse_key, source)] += numeric_weight
        return cls(
            fine_sums=fine_sums,
            fine_weights=dict(fine_weights),
            fine_source_sums=fine_source_sums,
            fine_source_weights=dict(fine_source_weights),
            coarse_sums=coarse_sums,
            coarse_weights=dict(coarse_weights),
            coarse_source_sums=coarse_source_sums,
            coarse_source_weights=dict(coarse_source_weights),
        )

    @staticmethod
    def _aggregate_excluding_source(
        sums: Mapping[Any, np.ndarray],
        weights: Mapping[Any, float],
        source_sums: Mapping[tuple[Any, str], np.ndarray],
        source_weights: Mapping[tuple[Any, str], float],
        key: Any,
        source: str,
    ) -> np.ndarray | None:
        if key not in sums:
            return None
        vector = sums[key] - source_sums.get((key, source), 0.0)
        weight = weights[key] - source_weights.get((key, source), 0.0)
        if weight <= 1e-12:
            return None
        mean = vector / weight
        norm = float(np.linalg.norm(mean))
        return None if norm <= 1e-12 or not np.isfinite(norm) else (mean / norm).astype(np.float32)

    def features(self, embeddings: np.ndarray, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        normalized = _unit_rows(embeddings)
        output = np.zeros((len(rows), 10), dtype=np.float32)
        for index, (vector, row) in enumerate(zip(normalized, rows, strict=True)):
            source = str(row["source_group"])
            role_rows = {
                "positive": {**row, "role": "positive"},
                "jitter_negative": {**row, "role": "jitter_hard_negative"},
                "active_fp": {**row, "role": "active_fp"},
            }
            for role_index, role in enumerate(("positive", "jitter_negative", "active_fp")):
                target = 1 if role == "positive" else 0
                role_row = role_rows[role]
                fine_key = _key(role_row, target)
                coarse_key = _coarse_key(role_row, target)
                fine = self._aggregate_excluding_source(
                    self.fine_sums,
                    self.fine_weights,
                    self.fine_source_sums,
                    self.fine_source_weights,
                    fine_key,
                    source,
                )
                coarse = self._aggregate_excluding_source(
                    self.coarse_sums,
                    self.coarse_weights,
                    self.coarse_source_sums,
                    self.coarse_source_weights,
                    coarse_key,
                    source,
                )
                output[index, role_index] = float(vector @ fine) if fine is not None else 0.0
                output[index, 3 + role_index] = (
                    float(vector @ coarse) if coarse is not None else 0.0
                )
            output[index, 6] = output[index, 0] - max(output[index, 1], output[index, 2])
            output[index, 7] = output[index, 3] - max(output[index, 4], output[index, 5])
            output[index, 8] = float(output[index, 1] != 0.0)
            output[index, 9] = float(output[index, 2] != 0.0)
        return output


class ApexBoundaryClassifier:
    """Serializable boundary head with optional prototype evidence.

    ``head="logistic"`` is the deliberately weak A0/A1 control.  The
    ``mlp_rank`` variant is still a frozen-backbone screen, but implements the
    proposal objective from plan 16: weighted BCE plus source-local positive /
    hard-negative ranking.  Trained torch parameters are converted to NumPy so
    proxy and Docker inference do not acquire a second runtime path.
    """

    def __init__(
        self,
        *,
        use_prototypes: bool,
        head: str = "logistic",
        random_state: int = 42,
    ) -> None:
        if head not in {"logistic", "mlp_rank"}:
            raise ValueError(f"unsupported APEX head: {head}")
        self.use_prototypes = bool(use_prototypes)
        self.head = head
        self.random_state = int(random_state)
        self.prototype_table: PrototypeTable | None = None
        self.scaler: Any = None
        self.classifier: Any = None
        self.feature_dimension: int | None = None

    @staticmethod
    def _metadata(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        output = np.zeros((len(rows), 2 + len(CATEGORY_IDS) + len(SCALE_BINS)), dtype=np.float32)
        for index, row in enumerate(rows):
            score = min(max(float(row["score"]), 1e-6), 1 - 1e-6)
            output[index, 0] = score
            output[index, 1] = math.log(score / (1 - score))
            category_id = int(row["category_id"])
            if category_id in CATEGORY_IDS:
                output[index, 2 + CATEGORY_IDS.index(category_id)] = 1.0
            scale = str(row["scale_bin"])
            if scale in SCALE_BINS:
                output[index, 2 + len(CATEGORY_IDS) + SCALE_BINS.index(scale)] = 1.0
        return output

    def _matrix(self, embeddings: np.ndarray, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        normalized = _unit_rows(embeddings)
        parts = [normalized, self._metadata(rows)]
        if self.use_prototypes:
            if self.prototype_table is None:
                raise RuntimeError("prototype table is not fitted")
            parts.append(self.prototype_table.features(normalized, rows))
        matrix = np.concatenate(parts, axis=1).astype(np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("model features contain non-finite values")
        return matrix

    def fit(
        self,
        embeddings: np.ndarray,
        rows: Sequence[Mapping[str, Any]],
        targets: Sequence[int],
        sample_weights: Sequence[float],
    ) -> "ApexBoundaryClassifier":
        from sklearn.preprocessing import StandardScaler

        labels = np.asarray(targets, dtype=np.int64)
        base_weights = np.asarray(sample_weights, dtype=np.float64)
        if len(rows) != len(labels) or embeddings.shape[0] != len(rows):
            raise ValueError("training inputs differ in length")
        if set(labels.tolist()) != {0, 1}:
            raise ValueError("training requires both binary targets")
        groups = Counter(str(row["source_group"]) for row in rows)
        robust_weights = np.asarray(
            [
                base_weights[index] / math.sqrt(groups[str(row["source_group"])])
                for index, row in enumerate(rows)
            ],
            dtype=np.float64,
        )
        for target in (0, 1):
            mask = labels == target
            robust_weights[mask] *= len(labels) / (2 * int(mask.sum()))
        if self.use_prototypes:
            self.prototype_table = PrototypeTable.fit(embeddings, rows, labels, robust_weights)
        matrix = self._matrix(embeddings, rows)
        self.scaler = StandardScaler().fit(matrix)
        transformed = self.scaler.transform(matrix)
        if self.head == "logistic":
            from sklearn.linear_model import LogisticRegression

            self.classifier = LogisticRegression(
                C=0.25,
                max_iter=1000,
                solver="liblinear",
                random_state=self.random_state,
            ).fit(transformed, labels, sample_weight=robust_weights)
        else:
            self.classifier = _fit_numpy_mlp_rank(
                transformed.astype(np.float32),
                labels,
                robust_weights,
                rows,
                random_state=self.random_state,
            )
        self.feature_dimension = int(matrix.shape[1])
        return self

    def predict_proba(
        self, embeddings: np.ndarray, rows: Sequence[Mapping[str, Any]]
    ) -> np.ndarray:
        if self.scaler is None or self.classifier is None:
            raise RuntimeError("classifier is not fitted")
        matrix = self._matrix(embeddings, rows)
        if matrix.shape[1] != self.feature_dimension:
            raise ValueError("feature dimension changed")
        return np.asarray(self.classifier.predict_proba(self.scaler.transform(matrix))[:, 1])


@dataclass
class _NumpyBinaryMLP:
    """Small deterministic inference representation for the ranking head."""

    weight1: np.ndarray
    bias1: np.ndarray
    weight2: np.ndarray
    bias2: float

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        hidden = np.maximum(matrix @ self.weight1.T + self.bias1, 0.0)
        logits = hidden @ self.weight2.reshape(-1) + float(self.bias2)
        positive = np.empty_like(logits, dtype=np.float64)
        mask = logits >= 0
        positive[mask] = 1.0 / (1.0 + np.exp(-logits[mask]))
        exponential = np.exp(logits[~mask])
        positive[~mask] = exponential / (1.0 + exponential)
        return np.column_stack((1.0 - positive, positive))


def _source_rank_pairs(
    rows: Sequence[Mapping[str, Any]], labels: np.ndarray, *, limit: int = 8192
) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic positive/negative pairs without crossing sources."""

    grouped: dict[str, dict[int, list[int]]] = defaultdict(lambda: {0: [], 1: []})
    for index, (row, label) in enumerate(zip(rows, labels, strict=True)):
        grouped[str(row["source_group"])][int(label)].append(index)
    pairs: list[tuple[int, int]] = []
    for source in sorted(grouped):
        positives = sorted(grouped[source][1], key=lambda index: str(rows[index]["row_id"]))
        negatives = grouped[source][0]
        if not positives or not negatives:
            continue
        width = min(4, len(negatives))
        for positive in positives:
            # The detector already separates easy background.  Ranking is
            # useful only around its present boundary, so select score-near
            # active FP / duplicate / jitter examples from the same source.
            nearby = sorted(
                negatives,
                key=lambda index: (
                    abs(float(rows[index]["score"]) - float(rows[positive]["score"])),
                    str(rows[index].get("role")) == "fp_bg",
                    str(rows[index]["row_id"]),
                ),
            )
            pairs.extend((positive, negative) for negative in nearby[:width])
    if len(pairs) > limit:
        stride = len(pairs) / limit
        pairs = [pairs[min(int(index * stride), len(pairs) - 1)] for index in range(limit)]
    if not pairs:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    positive, negative = zip(*pairs, strict=True)
    return np.asarray(positive, dtype=np.int64), np.asarray(negative, dtype=np.int64)


def _fit_numpy_mlp_rank(
    values: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    random_state: int,
) -> _NumpyBinaryMLP:
    """Fit compact BCE + source-local ranking head and freeze it to NumPy."""

    import torch
    from torch import nn

    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = torch.from_numpy(values).to(device)
    targets = torch.from_numpy(labels.astype(np.float32)).to(device)
    normalized_weights = weights / max(float(np.mean(weights)), 1e-12)
    sample_weights = torch.from_numpy(normalized_weights.astype(np.float32)).to(device)
    positive_pairs, negative_pairs = _source_rank_pairs(rows, labels)
    positive_pairs_tensor = torch.from_numpy(positive_pairs).to(device)
    negative_pairs_tensor = torch.from_numpy(negative_pairs).to(device)

    hidden_dimension = 64
    model = nn.Sequential(
        nn.Linear(values.shape[1], hidden_dimension),
        nn.ReLU(),
        nn.Dropout(p=0.10),
        nn.Linear(hidden_dimension, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    generator = torch.Generator(device="cpu").manual_seed(random_state)
    batch_size = 1024
    rank_batch_size = 1024
    for _ in range(18):
        model.train()
        permutation = torch.randperm(len(labels), generator=generator)
        for offset in range(0, len(labels), batch_size):
            indices = permutation[offset : offset + batch_size].to(device)
            logits = model(features[indices]).flatten()
            losses = nn.functional.binary_cross_entropy_with_logits(
                logits, targets[indices], reduction="none"
            )
            loss = (losses * sample_weights[indices]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        for offset in range(0, len(positive_pairs), rank_batch_size):
            positive_index = positive_pairs_tensor[offset : offset + rank_batch_size]
            negative_index = negative_pairs_tensor[offset : offset + rank_batch_size]
            positive_logits = model(features[positive_index]).flatten()
            negative_logits = model(features[negative_index]).flatten()
            rank_loss = nn.functional.softplus(0.50 - positive_logits + negative_logits).mean()
            optimizer.zero_grad(set_to_none=True)
            (0.50 * rank_loss).backward()
            optimizer.step()

    model.eval()
    first = model[0]
    final = model[3]
    return _NumpyBinaryMLP(
        weight1=first.weight.detach().cpu().numpy().astype(np.float32),
        bias1=first.bias.detach().cpu().numpy().astype(np.float32),
        weight2=final.weight.detach().cpu().numpy().astype(np.float32),
        bias2=float(final.bias.detach().cpu().item()),
    )


__all__ = ["ApexBoundaryClassifier", "PrototypeTable"]
