"""Deployable dual-view consistency quality features and logistic model."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

FEATURE_CONTRACTS = {"quality_features_v1_12d": 12, "quality_features_v2_22d": 22}


def blend_probability(
    raw_score: np.ndarray, quality_score: np.ndarray, alpha: float
) -> np.ndarray:
    """Blend detector and quality evidence in logit space."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("blend alpha must be within [0, 1]")
    raw = np.clip(np.asarray(raw_score, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    quality = np.clip(
        np.asarray(quality_score, dtype=np.float64), 1e-5, 1.0 - 1e-5
    )
    raw_logit = np.log(raw / (1.0 - raw))
    quality_logit = np.log(quality / (1.0 - quality))
    logit = (1.0 - alpha) * raw_logit + alpha * quality_logit
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))


def quality_features(
    rows: list[dict[str, Any]], contract: str
) -> np.ndarray:
    if contract not in FEATURE_CONTRACTS:
        raise ValueError(f"unsupported feature contract: {contract}")
    output: list[list[float]] = []
    for row in rows:
        score = float(np.clip(row["score"], 1e-5, 1.0 - 1e-5))
        nearby = float(
            np.clip(row["nearby_identity_score"], 1e-5, 1.0 - 1e-5)
        )
        support_iou = float(np.clip(row["novel_same_fine_iou"], 0.0, 1.0))
        logit_score = math.log(score / (1.0 - score))
        logit_nearby = math.log(nearby / (1.0 - nearby))
        x0, y0, x1, y1 = (float(value) for value in row["bbox_xyxy"])
        width = max(x1 - x0, 1e-3)
        height = max(y1 - y0, 1e-3)
        log_width = math.log1p(width)
        log_height = math.log1p(height)
        category = int(row["category_id"])
        category_slot = category if category <= 3 else 4
        base = [
            logit_score,
            logit_nearby,
            score - nearby,
            support_iou,
            log_width,
            log_height,
            math.log(width / height),
            *(1.0 if category_slot == index else 0.0 for index in range(5)),
        ]
        if contract == "quality_features_v2_22d":
            base.extend(
                [
                    score,
                    nearby,
                    min(score, nearby),
                    max(score, nearby),
                    abs(score - nearby),
                    logit_score * logit_nearby,
                    support_iou * logit_score,
                    support_iou * logit_nearby,
                    log_width + log_height,
                    abs(log_width - log_height),
                ]
            )
        output.append(base)
    return np.asarray(output, dtype=np.float64).reshape(
        len(rows), FEATURE_CONTRACTS[contract]
    )


def fit_logistic(
    rows: list[dict[str, Any]],
    labels: dict[int, int],
    contract: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    dimensions = FEATURE_CONTRACTS[contract]
    if not rows:
        return np.zeros(dimensions), np.ones(dimensions), np.zeros(dimensions), -20.0
    matrix = quality_features(rows, contract)
    target = np.asarray(
        [labels[int(row["source_prediction_index"])] for row in rows],
        dtype=np.float64,
    )
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (matrix - mean) / scale
    positives = float(target.sum())
    if positives == 0.0:
        return mean, scale, np.zeros(dimensions), -20.0
    positive_weight = min(30.0, (len(target) - positives) / positives)
    sample_weight = np.where(target > 0.5, positive_weight, 1.0)
    weight = np.zeros(dimensions, dtype=np.float64)
    bias = math.log((positives + 0.5) / (len(target) - positives + 0.5))
    for step in range(800):
        logit = np.clip(normalized @ weight + bias, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logit))
        error = (probability - target) * sample_weight
        learning_rate = 0.08 / math.sqrt(1.0 + step / 80.0)
        weight -= learning_rate * (
            normalized.T @ error / len(target) + 0.01 * weight
        )
        bias -= learning_rate * float(error.mean())
    return mean, scale, weight, float(bias)


def predict_logistic(
    rows: list[dict[str, Any]],
    model: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    contract: str,
) -> np.ndarray:
    if not rows:
        return np.empty(0, dtype=np.float64)
    mean, scale, weight, bias = model
    expected = FEATURE_CONTRACTS[contract]
    if mean.shape != (expected,) or scale.shape != (expected,) or weight.shape != (
        expected,
    ):
        raise ValueError("model and feature contract dimensions differ")
    logit = np.clip(
        ((quality_features(rows, contract) - mean) / scale) @ weight + bias,
        -30.0,
        30.0,
    )
    return 1.0 / (1.0 + np.exp(-logit))


def serialize_model(
    model: tuple[np.ndarray, np.ndarray, np.ndarray, float],
) -> dict[str, Any]:
    mean, scale, weight, bias = model
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weight": weight.tolist(),
        "bias": float(bias),
    }


def deserialize_model(
    payload: dict[str, Any], contract: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    required = {"mean", "scale", "weight", "bias"}
    if set(payload) != required:
        raise ValueError("invalid frozen logistic model schema")
    dimensions = FEATURE_CONTRACTS[contract]
    mean = np.asarray(payload["mean"], dtype=np.float64)
    scale = np.asarray(payload["scale"], dtype=np.float64)
    weight = np.asarray(payload["weight"], dtype=np.float64)
    if (
        mean.shape != (dimensions,)
        or scale.shape != (dimensions,)
        or weight.shape != (dimensions,)
    ):
        raise ValueError("frozen logistic model and feature contract dimensions differ")
    if (
        np.any(scale <= 0.0)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or not np.isfinite(weight).all()
    ):
        raise ValueError("frozen logistic model contains invalid values")
    bias = float(payload["bias"])
    if not np.isfinite(bias):
        raise ValueError("frozen logistic bias is not finite")
    return mean, scale, weight, bias
