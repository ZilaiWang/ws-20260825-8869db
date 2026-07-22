from __future__ import annotations

import numpy as np
import pytest

from rsdet.grouping.airport_proxy import (
    canonicalize_labels,
    cluster_components,
    collapse_components,
    expand_component_labels,
    fuse_embeddings,
    membership_scores,
    rotation_mean_embedding,
)
from rsdet.grouping.retrieval import RouteFeatures


def fixture_route(name: str, values: np.ndarray) -> RouteFeatures:
    repeated = np.repeat(values[:, None, :], 4, axis=1).astype(np.float32)
    return RouteFeatures(
        name=name,
        nodes=tuple(f"mar20:{index}" for index in range(1, len(values) + 1)),
        rotations=(0, 90, 180, 270),
        values=repeated,
        cache_fingerprint=name,
        feature_name=name,
    )


def test_rotation_mean_and_route_fusion_are_normalized() -> None:
    values = np.asarray([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]], dtype=np.float32)
    route_a = fixture_route("a", values)
    route_b = fixture_route("b", values[:, ::-1])
    nodes, averaged = rotation_mean_embedding(route_a)
    fused_nodes, fused = fuse_embeddings([route_a, route_b])
    assert nodes == fused_nodes
    assert np.allclose(np.linalg.norm(averaged, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(fused, axis=1), 1.0)


def test_strict_components_are_never_split_by_airport_clustering() -> None:
    pytest.importorskip("sklearn")
    values = np.asarray(
        [
            [1.0, 0.02],
            [0.99, 0.01],
            [0.95, 0.05],
            [0.02, 1.0],
            [0.01, 0.99],
            [0.05, 0.95],
            [-1.0, 0.02],
            [-0.99, 0.01],
            [-0.95, 0.05],
        ],
        dtype=np.float32,
    )
    route = fixture_route("a", values)
    nodes, embeddings = rotation_mean_embedding(route)
    components = {
        "mar20:1": "core-a",
        "mar20:2": "core-a",
        **{f"mar20:{index}": f"core-{index}" for index in range(3, 10)},
    }
    collapsed = collapse_components(nodes, embeddings, components)
    labels = cluster_components(collapsed, 3)
    expanded = expand_component_labels(collapsed, labels)
    assert expanded["mar20:1"] == expanded["mar20:2"]
    canonical = canonicalize_labels(expanded)
    assert len(set(canonical.values())) == 3
    assert sorted(set(canonical.values())) == [
        "mar20-airport-proxy-001",
        "mar20-airport-proxy-002",
        "mar20-airport-proxy-003",
    ]


def test_membership_scores_return_own_cosine_and_margin() -> None:
    nodes = ("mar20:1", "mar20:2", "mar20:3", "mar20:4")
    values = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    groups = {
        "mar20:1": "g1",
        "mar20:2": "g1",
        "mar20:3": "g2",
        "mar20:4": "g2",
    }
    scores = membership_scores(nodes, values, groups)
    assert all(0.99 < own <= 1.0 for own, _ in scores.values())
    assert all(margin > 0.8 for _, margin in scores.values())
