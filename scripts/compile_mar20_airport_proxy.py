#!/usr/bin/env python3
"""Compile the final K=60 MAR20 airport-proxy partition for CV grouping."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from rsdet.grouping.airport_proxy import (
    canonicalize_labels,
    cluster_components,
    collapse_components,
    expand_component_labels,
    fuse_embeddings,
    membership_scores,
    rotation_mean_embedding,
)
from rsdet.grouping.contracts import atomic_write_json, parse_node_uid, sha256_file
from rsdet.grouping.retrieval import load_rotation_route

SELECTED_ROUTES = (
    "masked_block10_vlad_k32_pca512",
    "masked_block11_vlad_k32_pca512",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--local-scene-groups", type=Path, required=True)
    parser.add_argument("--routes-json", type=Path, required=True)
    parser.add_argument("--round-b-decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-clusters", type=int, default=60)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def histogram(values: Counter[int]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(values.items())}


def main() -> None:
    args = parse_args()
    registry = read_csv(args.registry)
    local_rows = read_csv(args.local_scene_groups)
    if len(registry) != 3842 or len(local_rows) != 3842:
        raise ValueError("registry and local-scene assignments must both contain 3,842 rows")
    registry_by_node = {row["node_uid"]: row for row in registry}
    local_by_node = {row["node_uid"]: row for row in local_rows}
    if len(registry_by_node) != 3842 or set(registry_by_node) != set(local_by_node):
        raise ValueError("registry/local-scene node sets differ")

    route_config = json.loads(args.routes_json.read_text(encoding="utf-8"))
    route_by_name = {row["name"]: row for row in route_config["routes"]}
    round_b = json.loads(args.round_b_decision.read_text(encoding="utf-8"))
    if tuple(round_b["selected_routes"]) != SELECTED_ROUTES:
        raise ValueError("Round-B selected routes differ from the frozen airport-proxy contract")
    routes = [load_rotation_route(route_by_name[name]) for name in SELECTED_ROUTES]
    nodes, fused = fuse_embeddings(routes)
    if len(nodes) != 3842 or set(nodes) != set(registry_by_node):
        raise ValueError("descriptor cache does not cover the complete MAR20 registry")
    component_by_node = {node: local_by_node[node]["group_id"] for node in nodes}

    route_partitions: dict[str, dict[str, int]] = {}
    for route in routes:
        route_nodes, values = rotation_mean_embedding(route)
        components = collapse_components(route_nodes, values, component_by_node)
        route_partitions[route.name] = expand_component_labels(
            components, cluster_components(components, args.n_clusters)
        )
    fused_components = collapse_components(nodes, fused, component_by_node)
    final_numeric = expand_component_labels(
        fused_components, cluster_components(fused_components, args.n_clusters)
    )
    final_groups = canonicalize_labels(final_numeric)
    scores = membership_scores(nodes, fused, final_groups)

    strict_splits = 0
    for component_id in set(component_by_node.values()):
        members = [node for node in nodes if component_by_node[node] == component_id]
        strict_splits += len({final_groups[node] for node in members}) > 1
    if strict_splits:
        raise ValueError("airport proxy split a strict local-scene component")

    group_sizes_all = Counter(final_groups.values())
    target_nodes = [node for node in nodes if registry_by_node[node]["is_target"] == "1"]
    if len(target_nodes) != 3073:
        raise ValueError("expected exactly 3,073 competition MAR20 images")
    group_sizes_target = Counter(final_groups[node] for node in target_nodes)
    all_rows: list[dict[str, object]] = []
    for node in sorted(nodes, key=parse_node_uid):
        registry_row = registry_by_node[node]
        local_row = local_by_node[node]
        own_cosine, margin = scores[node]
        group = final_groups[node]
        all_rows.append(
            {
                "node_uid": node,
                "mar20_number": registry_row["mar20_number"],
                "competition_image_id": registry_row["competition_image_id"],
                "is_target": registry_row["is_target"],
                "is_bridge": registry_row["is_bridge"],
                "official_side": registry_row["official_side"],
                "airport_proxy_group_id": group,
                "airport_proxy_group_size_all": group_sizes_all[group],
                "airport_proxy_group_size_target": group_sizes_target.get(group, 0),
                "local_scene_core_id": local_row["group_id"],
                "local_scene_guard_id": local_row["cv_guard_group_id"],
                "membership_cosine": f"{own_cosine:.8f}",
                "centroid_margin": f"{margin:.8f}",
                "group_semantics": "airport_proxy_visual_cluster_k60_not_ground_truth",
            }
        )
    target_rows = [
        {
            "competition_image_id": row["competition_image_id"],
            "mar20_number": row["mar20_number"],
            "group_id": row["airport_proxy_group_id"],
            "group_size": row["airport_proxy_group_size_target"],
            "membership_cosine": row["membership_cosine"],
            "centroid_margin": row["centroid_margin"],
            "group_semantics": row["group_semantics"],
        }
        for row in all_rows
        if row["is_target"] == "1"
    ]

    labels = {
        name: np.asarray([partition[node] for node in nodes])
        for name, partition in route_partitions.items()
    }
    fused_labels = np.asarray([final_numeric[node] for node in nodes])
    route_agreement = {
        f"{name}__vs__fused_ari": float(adjusted_rand_score(value, fused_labels))
        for name, value in labels.items()
    }
    route_agreement["block10__vs__block11_ari"] = float(
        adjusted_rand_score(labels[SELECTED_ROUTES[0]], labels[SELECTED_ROUTES[1]])
    )
    cluster_target_counts = list(group_sizes_target.values())
    cluster_all_counts = list(group_sizes_all.values())
    summary = {
        "status": "airport_proxy_k60_ready_for_cv3",
        "formal_grouping_admission": True,
        "group_semantics": "airport_proxy_visual_cluster_not_airport_ground_truth",
        "known_source_count_prior": args.n_clusters,
        "algorithm": "strict-component collapse + rotation-mean route fusion + cosine average-link",
        "selected_routes": list(SELECTED_ROUTES),
        "registry_nodes": len(nodes),
        "target_nodes": len(target_nodes),
        "bridge_nodes": len(nodes) - len(target_nodes),
        "strict_component_count": len(fused_components.component_ids),
        "strict_component_split_count": strict_splits,
        "airport_proxy_groups_all": len(group_sizes_all),
        "airport_proxy_groups_with_target": len(group_sizes_target),
        "all_group_size_min": min(cluster_all_counts),
        "all_group_size_median": float(np.median(cluster_all_counts)),
        "all_group_size_max": max(cluster_all_counts),
        "target_group_size_min": min(cluster_target_counts),
        "target_group_size_median": float(np.median(cluster_target_counts)),
        "target_group_size_max": max(cluster_target_counts),
        "target_group_size_histogram": histogram(Counter(cluster_target_counts)),
        "route_partition_agreement": route_agreement,
        "confidence": {
            "membership_cosine_p05": float(np.quantile([scores[node][0] for node in nodes], 0.05)),
            "membership_cosine_median": float(np.median([scores[node][0] for node in nodes])),
            "centroid_margin_p05": float(np.quantile([scores[node][1] for node in nodes], 0.05)),
            "centroid_margin_median": float(np.median([scores[node][1] for node in nodes])),
        },
        "inputs": {
            "registry_sha256": sha256_file(args.registry),
            "local_scene_groups_sha256": sha256_file(args.local_scene_groups),
            "routes_json_sha256": sha256_file(args.routes_json),
            "round_b_decision_sha256": sha256_file(args.round_b_decision),
            "cache_fingerprints": {route.name: route.cache_fingerprint for route in routes},
        },
        "cv_recommendation": "use target CSV group_id; keep local-scene IDs for audit only",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "mar20_airport_proxy_assignments_all.csv", all_rows)
    write_csv(args.output_dir / "mar20_airport_proxy_assignments_target.csv", target_rows)
    atomic_write_json(args.output_dir / "airport_proxy_summary.json", summary)
    atomic_write_json(
        args.output_dir / "task_decision.json",
        {
            "status": summary["status"],
            "formal_grouping_admission": True,
            "airport_proxy_group_count": len(group_sizes_all),
            "strict_component_split_count": strict_splits,
            "target_assignment_sha256": sha256_file(
                args.output_dir / "mar20_airport_proxy_assignments_target.csv"
            ),
            "supersedes": "MAR20-FINAL-GROUPING-v1/mar20_final_group_assignments.csv",
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
