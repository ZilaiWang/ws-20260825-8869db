#!/usr/bin/env python3
"""MG01：在冻结人工 pair 上比较 DINO 层、聚合和背景视图。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rsdet.grouping.cache import PlaceFeatureCache
from rsdet.grouping.contracts import PROTOCOL_VERSION, atomic_write_json, sha256_file


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 MAR20 descriptor bake-off")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--calibration-pairs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k-values", default="20,50,100")
    parser.add_argument("--minimum-heldout-positive-directions", type=int, default=10)
    parser.add_argument("--heldout-recall-target", type=float, default=0.95)
    return parser.parse_args(argv)


def _read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"pair_uid", "node_u", "node_v", "binary_role", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"calibration pairs 缺少列: {sorted(required - set(rows[0] if rows else {}))}")
    return rows


def _normalize(array: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(norm <= 0) or not np.isfinite(norm).all():
        raise ValueError("描述子含零范数或非有限范数")
    return array / norm


def _aggregate(
    payload: dict[str, np.ndarray], feature_name: str, view_type: str
) -> tuple[list[str], dict[str, np.ndarray]]:
    nodes = payload["row__node_uid"].astype(str)
    views = payload["row__view_type"].astype(str)
    rotations = payload["row__rotation"].astype(int)
    features = payload[f"feature__{feature_name}"].astype(np.float32)
    by_node_rotation: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for node, view, rotation, vector in zip(nodes, views, rotations, features, strict=True):
        if view == view_type:
            by_node_rotation[(node, int(rotation))].append(vector)
    by_node: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for (node, rotation), vectors in by_node_rotation.items():
        by_node[node].append((rotation, _normalize(np.mean(vectors, axis=0, keepdims=True))[0]))
    result = {}
    for node, values in by_node.items():
        values.sort(key=lambda item: item[0])
        result[node] = np.stack([item[1] for item in values])
    return sorted(result, key=lambda value: int(value.split(":")[1])), result


def _pair_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(left @ right.T))


def _auc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    if not positive or not negative:
        return None
    wins = 0.0
    total = len(positive) * len(negative)
    for left in positive:
        for right in negative:
            wins += float(left > right) + 0.5 * float(left == right)
    return wins / total


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _is_hard_negative(row: dict[str, str]) -> bool:
    route = row.get("route", "")
    return any(
        token in route
        for token in (
            "legacy_dhash_candidate",
            "adjacent_id_audit",
            "same_class_cross_official_side",
            "same_class_same_official_side",
        )
    )


def _ranking_metrics(
    node_order: list[str], descriptors: dict[str, np.ndarray], pairs: list[dict[str, str]], k_values: tuple[int, ...]
) -> dict[str, Any]:
    node_to_index = {node: index for index, node in enumerate(node_order)}
    n = len(node_order)
    score_matrix = np.full((n, n), -np.inf, dtype=np.float32)
    for i, node_u in enumerate(node_order):
        for j in range(i + 1, n):
            value = _pair_similarity(descriptors[node_u], descriptors[node_order[j]])
            score_matrix[i, j] = value
            score_matrix[j, i] = value
    ranks: dict[tuple[str, str], int] = {}
    for index, node in enumerate(node_order):
        ordering = np.argsort(-score_matrix[index], kind="stable")
        for rank, candidate_index in enumerate(ordering, 1):
            if candidate_index == index:
                continue
            ranks[(node, node_order[int(candidate_index)])] = rank
    result: dict[str, Any] = {"node_count": n}
    for split in ("calibration", "held_out_audit"):
        positives = []
        negatives = []
        positive_ranks = []
        negative_ranks = []
        hard_negative_ranks = []
        ordinary_negative_ranks = []
        missing = 0
        for row in pairs:
            if row["split"] != split or row["binary_role"] == "excluded_uncertain":
                continue
            node_u, node_v = row["node_u"], row["node_v"]
            if node_u not in node_to_index or node_v not in node_to_index:
                missing += 1
                continue
            similarity = float(score_matrix[node_to_index[node_u], node_to_index[node_v]])
            directed = [ranks[(node_u, node_v)], ranks[(node_v, node_u)]]
            if row["binary_role"] == "positive":
                positives.append(similarity)
                positive_ranks.extend(directed)
            elif row["binary_role"] == "negative":
                negatives.append(similarity)
                negative_ranks.extend(directed)
                if _is_hard_negative(row):
                    hard_negative_ranks.extend(directed)
                else:
                    ordinary_negative_ranks.extend(directed)
        metrics: dict[str, Any] = {
            "positive_pairs": len(positives),
            "negative_pairs": len(negatives),
            "positive_directions": len(positive_ranks),
            "missing_pairs": missing,
            "auc": _auc(positives, negatives),
            "positive_similarity_median": float(np.median(positives)) if positives else None,
            "negative_similarity_median": float(np.median(negatives)) if negatives else None,
            "positive_median_rank": float(np.median(positive_ranks)) if positive_ranks else None,
            "negative_median_rank": float(np.median(negative_ranks)) if negative_ranks else None,
            "hard_negative_directions": len(hard_negative_ranks),
            "ordinary_negative_directions": len(ordinary_negative_ranks),
        }
        for k in k_values:
            effective_k = min(k, max(n - 1, 1))
            positive_hits = sum(rank <= effective_k for rank in positive_ranks)
            metrics[f"positive_recall_at_{k}"] = (
                positive_hits / len(positive_ranks) if positive_ranks else None
            )
            interval = _wilson(positive_hits, len(positive_ranks))
            metrics[f"positive_recall_at_{k}_wilson_low"] = interval[0]
            metrics[f"positive_recall_at_{k}_wilson_high"] = interval[1]
            metrics[f"negative_top_at_{k}_rate"] = (
                sum(rank <= effective_k for rank in negative_ranks) / len(negative_ranks)
                if negative_ranks
                else None
            )
            metrics[f"hard_negative_top_at_{k}_rate"] = (
                sum(rank <= effective_k for rank in hard_negative_ranks)
                / len(hard_negative_ranks)
                if hard_negative_ranks
                else None
            )
            metrics[f"ordinary_negative_top_at_{k}_rate"] = (
                sum(rank <= effective_k for rank in ordinary_negative_ranks)
                / len(ordinary_negative_ranks)
                if ordinary_negative_ranks
                else None
            )
        result[split] = metrics
    return result


def _neighbor_jaccard(
    left_order: list[str], left: dict[str, np.ndarray], right_order: list[str], right: dict[str, np.ndarray], k: int
) -> float | None:
    common = sorted(set(left_order) & set(right_order), key=lambda value: int(value.split(":")[1]))
    if len(common) <= 1:
        return None
    scores = []
    effective_k = min(k, len(common) - 1)
    for node in common:
        left_rank = sorted(
            (candidate for candidate in common if candidate != node),
            key=lambda candidate: (-_pair_similarity(left[node], left[candidate]), candidate),
        )[:effective_k]
        right_rank = sorted(
            (candidate for candidate in common if candidate != node),
            key=lambda candidate: (-_pair_similarity(right[node], right[candidate]), candidate),
        )[:effective_k]
        left_set, right_set = set(left_rank), set(right_rank)
        scores.append(len(left_set & right_set) / len(left_set | right_set))
    return float(np.mean(scores))


def _foreground_influence(
    original: dict[str, np.ndarray],
    masked: dict[str, np.ndarray],
    pairs: list[dict[str, str]],
) -> dict[str, float | int | None]:
    """量化目标前景对 pair 相似度的贡献。

    FI = cosine(original pair) - cosine(masked pair)。正值表示移除飞机后
    相似度下降，说明该 pair 的相似性至少部分来自目标前景，而非地点背景。
    """
    result: dict[str, float | int | None] = {}
    for split in ("calibration", "held_out_audit"):
        for role in ("positive", "negative"):
            values = []
            for row in pairs:
                if row["split"] != split or row["binary_role"] != role:
                    continue
                node_u, node_v = row["node_u"], row["node_v"]
                if not all(node in original and node in masked for node in (node_u, node_v)):
                    continue
                values.append(
                    _pair_similarity(original[node_u], original[node_v])
                    - _pair_similarity(masked[node_u], masked[node_v])
                )
            prefix = f"foreground_influence__{split}__{role}"
            result[f"{prefix}_count"] = len(values)
            result[f"{prefix}_median"] = float(np.median(values)) if values else None
            result[f"{prefix}_p95"] = float(np.quantile(values, 0.95)) if values else None
            result[f"{prefix}_absolute_median"] = (
                float(np.median(np.abs(values))) if values else None
            )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    k_values = tuple(int(value) for value in args.k_values.split(",") if value.strip())
    if not k_values or any(value <= 0 for value in k_values):
        raise ValueError("k-values 必须为正整数")
    cache = PlaceFeatureCache(args.cache_dir)
    audit = cache.audit()
    payload = cache.load_all()
    pairs_path = Path(args.calibration_pairs).expanduser().resolve()
    pairs = _read_pairs(pairs_path)
    view_types = sorted(set(payload["row__view_type"].astype(str)))
    metrics_rows = []
    descriptor_maps: dict[tuple[str, str], tuple[list[str], dict[str, np.ndarray]]] = {}
    for feature_name in cache.feature_names:
        for view_type in view_types:
            order, descriptors = _aggregate(payload, feature_name, view_type)
            if not descriptors:
                continue
            descriptor_maps[(feature_name, view_type)] = (order, descriptors)
            result = _ranking_metrics(order, descriptors, pairs, k_values)
            row: dict[str, Any] = {
                "feature_name": feature_name,
                "feature_dimension": audit["feature_dimensions"][feature_name],
                "view_type": view_type,
            }
            for split in ("calibration", "held_out_audit"):
                for name, value in result[split].items():
                    row[f"{split}__{name}"] = value
            metrics_rows.append(row)
    for row in metrics_rows:
        feature = row["feature_name"]
        if (feature, "original") in descriptor_maps and (feature, "masked_inpaint") in descriptor_maps:
            left = descriptor_maps[(feature, "original")]
            right = descriptor_maps[(feature, "masked_inpaint")]
            row["original_masked_neighbor_jaccard_at_20"] = _neighbor_jaccard(*left, *right, 20)
            row.update(_foreground_influence(left[1], right[1], pairs))
        else:
            row["original_masked_neighbor_jaccard_at_20"] = None
    maximum_k = max(k_values)
    heldout_key = f"held_out_audit__positive_recall_at_{maximum_k}"
    hard_negative_key = f"held_out_audit__hard_negative_top_at_{maximum_k}_rate"
    all_negative_key = f"held_out_audit__negative_top_at_{maximum_k}_rate"
    foreground_key = "foreground_influence__held_out_audit__positive_absolute_median"

    def selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
        recall = row.get(heldout_key)
        hard_negative_rate = row.get(hard_negative_key)
        if hard_negative_rate is None:
            hard_negative_rate = row.get(all_negative_key)
        foreground = row.get(foreground_key)
        return (
            -(float(recall) if recall is not None else -1.0),
            float(hard_negative_rate) if hard_negative_rate is not None else math.inf,
            float(foreground) if foreground is not None else math.inf,
            int(row["feature_dimension"]),
            row["view_type"] != "masked_inpaint",
            row["feature_name"],
        )

    metrics_rows.sort(
        key=selection_key
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "descriptor_bakeoff.csv"
    fields = sorted({key for row in metrics_rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics_rows)
    eligible = []
    for row in metrics_rows:
        directions = int(row.get("held_out_audit__positive_directions") or 0)
        recall = row.get(heldout_key)
        if (
            row["view_type"] in {"masked_inpaint", "background_tiles"}
            and directions >= args.minimum_heldout_positive_directions
            and recall is not None
            and float(recall) >= args.heldout_recall_target
        ):
            eligible.append(row)
    selection = {
        "status": "pass" if eligible else "insufficient_evidence_or_recall",
        "formal_retrieval_admission": bool(eligible),
        "selection_is_provisional_until_vlad": True,
        "selected_round_a": (
            {"feature_name": eligible[0]["feature_name"], "view_type": eligible[0]["view_type"]}
            if eligible
            else None
        ),
        "ranking_rule": [
            f"held_out positive recall@{maximum_k} descending",
            f"held_out hard-negative (or all-negative fallback) top@{maximum_k} ascending",
            "held_out positive absolute foreground influence ascending",
            "feature dimension ascending",
            "masked view before background view",
            "feature name deterministic tie break",
        ],
    }
    atomic_write_json(output_dir / "selected_descriptor.json", selection)
    summary = {
        "status": selection["status"],
        "protocol_version": PROTOCOL_VERSION,
        "cache_audit": audit,
        "pair_manifest_sha256": sha256_file(pairs_path),
        "metric_row_count": len(metrics_rows),
        "view_types": view_types,
        "selection": selection,
        "artifacts": {
            "descriptor_bakeoff.csv": sha256_file(csv_path),
            "selected_descriptor.json": sha256_file(output_dir / "selected_descriptor.json"),
        },
    }
    atomic_write_json(output_dir / "bakeoff_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
