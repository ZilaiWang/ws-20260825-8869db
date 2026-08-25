"""Deterministic multi-rotation retrieval utilities for MAR20 grouping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rsdet.grouping.cache import PlaceFeatureCache
from rsdet.grouping.contracts import parse_node_uid


@dataclass(frozen=True)
class RouteFeatures:
    """One descriptor route arranged as [node, rotation, dimension]."""

    name: str
    nodes: tuple[str, ...]
    rotations: tuple[int, ...]
    values: np.ndarray
    cache_fingerprint: str
    feature_name: str


@dataclass(frozen=True)
class Neighbor:
    node_uid: str
    rank: int
    score: float
    query_rotation: int
    neighbor_rotation: int


def load_rotation_route(route: dict[str, Any]) -> RouteFeatures:
    """Load and validate a four-rotation global descriptor route."""

    required = {"name", "cache_dir", "feature"}
    missing = required - set(route)
    if missing:
        raise ValueError(f"route missing fields: {sorted(missing)}")
    cache = PlaceFeatureCache(Path(route["cache_dir"]))
    feature_name = str(route["feature"])
    if feature_name not in cache.feature_names:
        raise ValueError(f"{route['name']}: feature {feature_name!r} missing from cache")
    payload = cache.load_all()
    keep = np.ones(len(payload["row__node_uid"]), dtype=bool)
    view_type = route.get("view_type")
    if view_type:
        keep &= payload["row__view_type"].astype(str) == str(view_type)
    nodes = payload["row__node_uid"].astype(str)[keep]
    rotations = payload["row__rotation"].astype(int)[keep]
    values = np.asarray(payload[f"feature__{feature_name}"][keep], dtype=np.float32)
    if not len(nodes) or not np.isfinite(values).all():
        raise ValueError(f"{route['name']}: empty or non-finite route")
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    by_node: dict[str, dict[int, np.ndarray]] = {}
    for node, rotation, value in zip(nodes, rotations, values, strict=True):
        mapping = by_node.setdefault(node, {})
        if int(rotation) in mapping:
            raise ValueError(f"{route['name']}: duplicate {node}/rotation={rotation}")
        mapping[int(rotation)] = value
    ordered_nodes = tuple(sorted(by_node, key=parse_node_uid))
    rotation_sets = {tuple(sorted(by_node[node])) for node in ordered_nodes}
    if len(rotation_sets) != 1:
        raise ValueError(f"{route['name']}: inconsistent rotation sets")
    ordered_rotations = next(iter(rotation_sets))
    if ordered_rotations != (0, 90, 180, 270):
        raise ValueError(
            f"{route['name']}: formal retrieval requires rotations (0,90,180,270), "
            f"actual={ordered_rotations}"
        )
    matrix = np.stack(
        [
            np.stack([by_node[node][rotation] for rotation in ordered_rotations])
            for node in ordered_nodes
        ]
    ).astype(np.float32, copy=False)
    return RouteFeatures(
        name=str(route["name"]),
        nodes=ordered_nodes,
        rotations=ordered_rotations,
        values=matrix,
        cache_fingerprint=str(cache.index["fingerprint"]),
        feature_name=feature_name,
    )


def directional_topk(
    route: RouteFeatures,
    *,
    k: int,
    device: str = "cuda",
    batch_size: int = 64,
) -> dict[str, list[Neighbor]]:
    """Retrieve node-level top-K using the best query/database rotation pair."""

    if not 0 < k < len(route.nodes):
        raise ValueError("k must be positive and smaller than node count")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    try:
        import torch
    except ImportError:
        torch = None
    use_torch = bool(
        torch is not None
        and (device == "cpu" or (device.startswith("cuda") and torch.cuda.is_available()))
    )
    result: dict[str, list[Neighbor]] = {}
    rotation_count = len(route.rotations)
    if use_torch:
        target = torch.device(device)
        database = torch.from_numpy(route.values).to(target)
        for offset in range(0, len(route.nodes), batch_size):
            query = database[offset : offset + batch_size]
            all_scores = torch.einsum("brd,nsd->brns", query, database)
            flattened = all_scores.permute(0, 2, 1, 3).reshape(
                query.shape[0], len(route.nodes), rotation_count * rotation_count
            )
            scores, rotation_pairs = flattened.max(dim=2)
            local = torch.arange(query.shape[0], device=target)
            scores[local, local + offset] = -torch.inf
            top_scores, indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
            selected_pairs = torch.gather(rotation_pairs, 1, indices)
            for local_index in range(query.shape[0]):
                values = []
                for rank, (index, score, pair_index) in enumerate(
                    zip(
                        indices[local_index].cpu().tolist(),
                        top_scores[local_index].float().cpu().tolist(),
                        selected_pairs[local_index].cpu().tolist(),
                        strict=True,
                    ),
                    1,
                ):
                    values.append(
                        Neighbor(
                            node_uid=route.nodes[int(index)],
                            rank=rank,
                            score=float(score),
                            query_rotation=route.rotations[int(pair_index) // rotation_count],
                            neighbor_rotation=route.rotations[int(pair_index) % rotation_count],
                        )
                    )
                result[route.nodes[offset + local_index]] = values
        return result

    values = route.values
    for query_index, node in enumerate(route.nodes):
        all_scores = np.einsum("rd,nsd->rns", values[query_index], values)
        flattened = all_scores.transpose(1, 0, 2).reshape(len(route.nodes), -1)
        pair_indices = flattened.argmax(axis=1)
        scores = flattened[np.arange(len(route.nodes)), pair_indices]
        scores[query_index] = -np.inf
        indices = np.argpartition(scores, -k)[-k:]
        indices = indices[np.lexsort((np.asarray(indices), -scores[indices]))]
        neighbors = []
        for rank, index in enumerate(indices, 1):
            pair_index = int(pair_indices[index])
            neighbors.append(
                Neighbor(
                    node_uid=route.nodes[int(index)],
                    rank=rank,
                    score=float(scores[index]),
                    query_rotation=route.rotations[pair_index // rotation_count],
                    neighbor_rotation=route.rotations[pair_index % rotation_count],
                )
            )
        result[node] = neighbors
    return result


def union_recall(
    route_neighbors: Sequence[dict[str, list[Neighbor]]],
    directions: Sequence[tuple[str, str]],
    *,
    k: int,
) -> tuple[int, int]:
    """Return hits/total for a per-route K union."""

    if k <= 0 or not route_neighbors:
        raise ValueError("union recall requires routes and positive K")
    hits = 0
    for query, target in directions:
        candidates = {
            item.node_uid for route in route_neighbors for item in route.get(query, [])[:k]
        }
        hits += target in candidates
    return hits, len(directions)
