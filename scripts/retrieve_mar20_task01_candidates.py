#!/usr/bin/env python3
"""Run frozen K=50 MAR20 retrieval and emit a K=100 audit index."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageOps

from rsdet.grouping.contracts import (
    MASKED_PATCH_PROTOCOL_VERSION,
    atomic_write_json,
    canonical_pair_uid,
    parse_node_uid,
    sha256_file,
)
from rsdet.grouping.geometry import phash64
from rsdet.grouping.registry import load_registry
from rsdet.grouping.retrieval import Neighbor, directional_topk, load_rotation_route, union_recall


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAR20 TASK-01 formal retrieval")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--mar20-root", required=True)
    parser.add_argument("--routes-json", required=True)
    parser.add_argument("--round-b-decision", required=True)
    parser.add_argument("--task-00b2-decision", required=True)
    parser.add_argument("--calibration-pairs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--formal-k", type=int, default=50)
    parser.add_argument("--audit-k", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _wilson(success: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = success / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _pair_directions(
    pairs: list[dict[str, str]], split: str, role: str
) -> list[tuple[str, str]]:
    directions = []
    for row in pairs:
        if split != "all" and row["split"] != split:
            continue
        if row["binary_role"] != role:
            continue
        directions.extend(((row["node_u"], row["node_v"]), (row["node_v"], row["node_u"])))
    return directions


def _union_rate(
    routes: list[dict[str, list[Neighbor]]], directions: list[tuple[str, str]], k: int
) -> float | None:
    hits, total = union_recall(routes, directions, k=k)
    return hits / total if total else None


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(({key: row.get(key, "") for key in fields} for row in rows))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 < args.formal_k <= args.audit_k:
        raise ValueError("require 0 < formal-k <= audit-k")
    registry_path = Path(args.registry).expanduser().resolve()
    routes_path = Path(args.routes_json).expanduser().resolve()
    round_path = Path(args.round_b_decision).expanduser().resolve()
    task_path = Path(args.task_00b2_decision).expanduser().resolve()
    pairs_path = Path(args.calibration_pairs).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry_rows = load_registry(registry_path)
    registry = {row["node_uid"]: row for row in registry_rows}
    ordered_nodes = sorted(registry, key=parse_node_uid)
    if len(ordered_nodes) != 3842:
        raise ValueError(f"formal MAR20 registry requires 3842 nodes, actual={len(ordered_nodes)}")
    round_decision = json.loads(round_path.read_text(encoding="utf-8"))
    task_decision = json.loads(task_path.read_text(encoding="utf-8"))
    selected = list(round_decision.get("selected_routes", []))
    if (
        round_decision.get("formal_descriptor_selection_admission") is not True
        or round_decision.get("selection_uses_heldout") is not False
        or task_decision.get("task01_retrieval_admission") is not True
        or selected != task_decision.get("selected_routes")
        or len(selected) != 2
    ):
        raise ValueError("TASK-00B2 descriptor decisions are not admitted or inconsistent")
    route_config = json.loads(routes_path.read_text(encoding="utf-8"))
    by_name = {str(route["name"]): route for route in route_config.get("routes", [])}
    if any(name not in by_name for name in selected):
        raise ValueError("selected route missing from routes-json")
    routes = [load_rotation_route(by_name[name]) for name in selected]
    if any(tuple(route.nodes) != tuple(ordered_nodes) for route in routes):
        raise ValueError("route node order/coverage differs from registry")
    neighbors = [
        directional_topk(
            route,
            k=args.audit_k,
            device=args.device,
            batch_size=args.batch_size,
        )
        for route in routes
    ]

    edges: dict[str, dict[str, Any]] = {}
    for route_index, route_neighbors in enumerate(neighbors, 1):
        prefix = f"r{route_index}"
        for query, values in route_neighbors.items():
            for item in values:
                pair_uid = canonical_pair_uid(query, item.node_uid)
                left, right = pair_uid.split("--")
                row = edges.setdefault(pair_uid, {"pair_uid": pair_uid, "node_u": left, "node_v": right})
                direction = "u_to_v" if query == left else "v_to_u"
                row[f"{prefix}_rank_{direction}"] = item.rank
                row[f"{prefix}_score_{direction}"] = item.score
                row[f"{prefix}_query_rotation_{direction}"] = item.query_rotation
                row[f"{prefix}_neighbor_rotation_{direction}"] = item.neighbor_rotation

    by_pixel: dict[str, list[str]] = {}
    for uid, row in registry.items():
        by_pixel.setdefault(row["original_pixel_sha256"], []).append(uid)
    for group in by_pixel.values():
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                pair_uid = canonical_pair_uid(left, right)
                node_u, node_v = pair_uid.split("--")
                edges.setdefault(pair_uid, {"pair_uid": pair_uid, "node_u": node_u, "node_v": node_v})[
                    "exact_pixel"
                ] = 1

    root = Path(args.mar20_root).expanduser().resolve()
    hashes: dict[str, np.ndarray] = {}
    for uid in ordered_nodes:
        path = (root / registry[uid]["original_relative_path"]).resolve()
        path.relative_to(root)
        with Image.open(path) as image:
            image.load()
            hashes[uid] = phash64(ImageOps.exif_transpose(image).convert("RGB"))

    route_fields: list[str] = []
    for route_index in range(1, len(routes) + 1):
        for direction in ("u_to_v", "v_to_u"):
            route_fields.extend(
                [
                    f"r{route_index}_rank_{direction}",
                    f"r{route_index}_score_{direction}",
                    f"r{route_index}_query_rotation_{direction}",
                    f"r{route_index}_neighbor_rotation_{direction}",
                ]
            )
    rows = []
    for row in edges.values():
        u, v = row["node_u"], row["node_v"]
        is_target_u = registry[u]["is_target"] == "1"
        is_target_v = registry[v]["is_target"] == "1"
        formal_support = audit_support = mutual_support = 0
        formal_ranks = []
        similarities = []
        for route_index in range(1, len(routes) + 1):
            ranks = [
                row.get(f"r{route_index}_rank_u_to_v"),
                row.get(f"r{route_index}_rank_v_to_u"),
            ]
            scores = [
                row.get(f"r{route_index}_score_u_to_v"),
                row.get(f"r{route_index}_score_v_to_u"),
            ]
            if any(value is not None and int(value) <= args.audit_k for value in ranks):
                audit_support += 1
            if any(value is not None and int(value) <= args.formal_k for value in ranks):
                formal_support += 1
                formal_ranks.extend(int(value) for value in ranks if value is not None)
                similarities.extend(float(value) for value in scores if value is not None)
            if all(value is not None and int(value) <= args.formal_k for value in ranks):
                mutual_support += 1
        row.update(
            {
                "scope": "target_only" if is_target_u and is_target_v else "full_bridge_diagnostic",
                "target_target": int(is_target_u and is_target_v),
                "target_bridge": int(is_target_u != is_target_v),
                "bridge_bridge": int(not is_target_u and not is_target_v),
                "cross_official_side": int(
                    registry[u]["official_side"] != registry[v]["official_side"]
                ),
                "exact_pixel": int(bool(row.get("exact_pixel"))),
                "phash_distance": int(np.count_nonzero(hashes[u] != hashes[v])),
                "in_formal_k50": int(formal_support > 0 or bool(row.get("exact_pixel"))),
                "in_audit_k100": int(audit_support > 0 or bool(row.get("exact_pixel"))),
                "formal_route_support": formal_support,
                "audit_route_support": audit_support,
                "formal_mutual_route_support": mutual_support,
                "best_formal_rank": min(formal_ranks) if formal_ranks else "",
                "best_similarity": max(similarities) if similarities else "",
            }
        )
        rows.append(row)
    rows.sort(key=lambda row: (parse_node_uid(row["node_u"]), parse_node_uid(row["node_v"])))
    common_fields = [
        "pair_uid",
        "node_u",
        "node_v",
        "scope",
        "target_target",
        "target_bridge",
        "bridge_bridge",
        "cross_official_side",
        "exact_pixel",
        "phash_distance",
        "in_formal_k50",
        "in_audit_k100",
        "formal_route_support",
        "audit_route_support",
        "formal_mutual_route_support",
        "best_formal_rank",
        "best_similarity",
        *route_fields,
    ]
    audit_rows = [row for row in rows if row["in_audit_k100"]]
    formal_rows = [row for row in rows if row["in_formal_k50"]]
    target_rows = [row for row in formal_rows if row["target_target"]]
    paths = {
        "candidate_edges_full_bridge_k100.csv": audit_rows,
        "candidate_edges_full_bridge_k50.csv": formal_rows,
        "candidate_edges_target_k50.csv": target_rows,
    }
    for name, values in paths.items():
        _write_csv(output / name, values, common_fields)

    pairs = _read_csv(pairs_path)
    saturation: dict[str, Any] = {}
    for split in ("calibration", "held_out_audit", "all"):
        positives = _pair_directions(pairs, split, "positive")
        negatives = _pair_directions(pairs, split, "negative")
        for k in (20, args.formal_k, args.audit_k):
            hits, total = union_recall(neighbors, positives, k=k)
            low, high = _wilson(hits, total)
            saturation[f"{split}__positive_directions"] = total
            saturation[f"{split}__recall_at_{k}"] = hits / total if total else None
            saturation[f"{split}__recall_at_{k}_wilson_low"] = low
            saturation[f"{split}__recall_at_{k}_wilson_high"] = high
            saturation[f"{split}__known_negative_top_at_{k}"] = _union_rate(
                neighbors, negatives, k
            )
    heldout_recall = saturation[f"held_out_audit__recall_at_{args.formal_k}"]
    status = "pass" if heldout_recall is not None and heldout_recall >= 0.95 else "fail_recall"
    summary = {
        "status": status,
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "formal_k_per_route": args.formal_k,
        "audit_k_per_route": args.audit_k,
        "selected_routes": selected,
        "route_metadata": [
            {
                "name": route.name,
                "feature": route.feature_name,
                "cache_fingerprint": route.cache_fingerprint,
            }
            for route in routes
        ],
        "registry_sha256": sha256_file(registry_path),
        "routes_json_sha256": sha256_file(routes_path),
        "round_b_decision_sha256": sha256_file(round_path),
        "task_00b2_decision_sha256": sha256_file(task_path),
        "calibration_pairs_sha256": sha256_file(pairs_path),
        "node_count": len(ordered_nodes),
        "audit_edge_count": len(audit_rows),
        "formal_edge_count": len(formal_rows),
        "target_formal_edge_count": len(target_rows),
        "exact_pixel_edge_count": sum(row["exact_pixel"] for row in rows),
        "saturation": saturation,
        "formal_grouping_admission": False,
        "artifacts": {name: sha256_file(output / name) for name in paths},
    }
    atomic_write_json(output / "retrieval_summary.json", summary)
    atomic_write_json(
        output / "retrieval_decision.json",
        {
            "status": "ready_for_geometry_queue" if status == "pass" else status,
            "formal_retrieval_admission": status == "pass",
            "formal_grouping_admission": False,
            "formal_k_per_route": args.formal_k,
            "audit_k_per_route": args.audit_k,
            "selected_routes": selected,
            "retrieval_summary_sha256": sha256_file(output / "retrieval_summary.json"),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
