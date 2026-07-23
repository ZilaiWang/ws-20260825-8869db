"""Deterministic airport-proxy clustering for the MAR20 source domain."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from rsdet.grouping.contracts import parse_node_uid
from rsdet.grouping.retrieval import RouteFeatures


def l2_normalize(values: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized finite float32 features."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
        raise ValueError("features must be a non-empty finite [N,D] matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("features contain a zero-norm row")
    return array / norms


def rotation_mean_embedding(route: RouteFeatures) -> tuple[tuple[str, ...], np.ndarray]:
    """Average the four rotation descriptors and normalize per MAR20 image."""

    if route.values.ndim != 3 or route.values.shape[1] != 4:
        raise ValueError(f"{route.name}: expected [N,4,D] descriptors")
    return route.nodes, l2_normalize(route.values.mean(axis=1))


def fuse_embeddings(routes: Sequence[RouteFeatures]) -> tuple[tuple[str, ...], np.ndarray]:
    """Concatenate equally weighted, rotation-averaged descriptor routes."""

    if not routes:
        raise ValueError("at least one descriptor route is required")
    nodes, first = rotation_mean_embedding(routes[0])
    parts = [first]
    for route in routes[1:]:
        current_nodes, values = rotation_mean_embedding(route)
        if current_nodes != nodes:
            raise ValueError("descriptor routes have different node ordering")
        parts.append(values)
    return nodes, l2_normalize(np.concatenate(parts, axis=1))


@dataclass(frozen=True)
class ComponentFeatures:
    component_ids: tuple[str, ...]
    members: tuple[tuple[str, ...], ...]
    values: np.ndarray


def collapse_components(
    nodes: Sequence[str], values: np.ndarray, component_by_node: dict[str, str]
) -> ComponentFeatures:
    """Collapse strict local-scene components before airport-level clustering."""

    if len(nodes) != len(values) or set(nodes) != set(component_by_node):
        raise ValueError("node/features/component membership mismatch")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(nodes):
        grouped[component_by_node[node]].append(index)
    ordered_ids = tuple(
        sorted(grouped, key=lambda item: min(parse_node_uid(nodes[i]) for i in grouped[item]))
    )
    members = tuple(
        tuple(sorted((nodes[index] for index in grouped[item]), key=parse_node_uid))
        for item in ordered_ids
    )
    centroids = np.stack([values[grouped[item]].mean(axis=0) for item in ordered_ids])
    return ComponentFeatures(ordered_ids, members, l2_normalize(centroids))


def cluster_components(components: ComponentFeatures, n_clusters: int) -> np.ndarray:
    """Average-link cosine clustering of strict components."""

    if not 1 < n_clusters < len(components.component_ids):
        raise ValueError("n_clusters must be between 2 and component_count-1")
    try:
        from sklearn.cluster import AgglomerativeClustering
    except ImportError as error:
        raise RuntimeError("airport-proxy clustering requires scikit-learn") from error
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="cosine",
        linkage="average",
    )
    return np.asarray(model.fit_predict(components.values), dtype=np.int64)


def expand_component_labels(
    components: ComponentFeatures,
    labels: np.ndarray,
) -> dict[str, int]:
    if labels.shape != (len(components.component_ids),):
        raise ValueError("component labels have an unexpected shape")
    return {
        node: int(label)
        for members, label in zip(components.members, labels, strict=True)
        for node in members
    }


def canonicalize_labels(labels_by_node: dict[str, int]) -> dict[str, str]:
    """Name clusters by their smallest MAR20 number for reproducible IDs."""

    grouped: dict[int, list[str]] = defaultdict(list)
    for node, label in labels_by_node.items():
        grouped[int(label)].append(node)
    ordered = sorted(grouped, key=lambda label: min(parse_node_uid(n) for n in grouped[label]))
    names = {label: f"mar20-airport-proxy-{index:03d}" for index, label in enumerate(ordered, 1)}
    return {node: names[label] for node, label in labels_by_node.items()}


def membership_scores(
    nodes: Sequence[str],
    values: np.ndarray,
    group_by_node: dict[str, str],
) -> dict[str, tuple[float, float]]:
    """Return own-centroid cosine and the margin over the next centroid."""

    if len(nodes) != len(values) or set(nodes) != set(group_by_node):
        raise ValueError("node/features/group membership mismatch")
    groups = sorted(set(group_by_node.values()))
    group_index = {group: index for index, group in enumerate(groups)}
    indices: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(nodes):
        indices[group_by_node[node]].append(index)
    centroids = l2_normalize(np.stack([values[indices[group]].mean(axis=0) for group in groups]))
    similarities = values @ centroids.T
    result = {}
    for index, node in enumerate(nodes):
        own_index = group_index[group_by_node[node]]
        own = float(similarities[index, own_index])
        other = np.delete(similarities[index], own_index)
        second = float(other.max()) if len(other) else -1.0
        result[node] = (own, own - second)
    return result
