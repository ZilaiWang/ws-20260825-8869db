#!/usr/bin/env python3
"""Evaluate v1.2 masked GeM/VLAD routes and their pre-registered unions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rsdet.grouping.cache import PlaceFeatureCache
from rsdet.grouping.contracts import (
    MASKED_PATCH_PROTOCOL_VERSION,
    atomic_write_json,
    parse_node_uid,
    sha256_file,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MAR20 00B masked Round-B routes")
    parser.add_argument("--routes-json", required=True)
    parser.add_argument("--calibration-pairs", required=True)
    parser.add_argument("--calibration-summary", required=True)
    parser.add_argument("--patch-mask-decision", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k-values", default="20,50,100")
    parser.add_argument("--heldout-recall-target", type=float, default=0.95)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--node-batch-size", type=int, default=64)
    return parser.parse_args(argv)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _wilson(
    success: int, total: int, z: float = 1.959963984540054
) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    proportion = success / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _load_route(route: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    cache = PlaceFeatureCache(route["cache_dir"])
    feature = route["feature"]
    if feature not in cache.feature_names:
        raise ValueError(f"{route['name']}: missing feature {feature}")
    payload = cache.load_all()
    keep = np.ones(len(payload["row__node_uid"]), dtype=bool)
    if route.get("view_type"):
        keep &= payload["row__view_type"].astype(str) == route["view_type"]
    nodes = payload["row__node_uid"].astype(str)[keep]
    rotations = payload["row__rotation"].astype(int)[keep]
    values = np.asarray(payload[f"feature__{feature}"][keep], dtype=np.float32)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    by_node: dict[str, list[tuple[int, np.ndarray]]] = {}
    for node, rotation, value in zip(nodes, rotations, values, strict=True):
        by_node.setdefault(node, []).append((int(rotation), value))
    ordered_nodes = sorted(by_node, key=parse_node_uid)
    rotation_sets = {
        tuple(sorted(rotation for rotation, _ in by_node[node])) for node in ordered_nodes
    }
    if len(rotation_sets) != 1:
        raise ValueError(f"{route['name']}: inconsistent rotation sets")
    expected_rotations = next(iter(rotation_sets))
    if len(expected_rotations) != len(set(expected_rotations)):
        raise ValueError(f"{route['name']}: duplicate rotations")
    matrix = np.stack(
        [np.stack([value for _, value in sorted(by_node[node])]) for node in ordered_nodes]
    )
    return ordered_nodes, matrix


def _neighbors(
    nodes: list[str], values: np.ndarray, *, maximum_k: int, device: str, batch_size: int
) -> dict[str, list[str]]:
    if maximum_k >= len(nodes):
        raise ValueError("maximum K must be smaller than node count")
    try:
        import torch
    except ImportError:
        torch = None
    result: dict[str, list[str]] = {}
    if torch is not None and (
        device == "cpu" or (device.startswith("cuda") and torch.cuda.is_available())
    ):
        target = torch.device(device)
        database = torch.from_numpy(values).to(target)
        for offset in range(0, len(nodes), batch_size):
            query = database[offset : offset + batch_size]
            scores = torch.einsum("brd,nsd->brns", query, database).amax(dim=(1, 3))
            local = torch.arange(scores.shape[0], device=target)
            scores[local, local + offset] = -torch.inf
            indices = (
                torch.topk(scores, k=maximum_k, dim=1, largest=True, sorted=True)
                .indices.cpu()
                .numpy()
            )
            for local_index, row in enumerate(indices):
                result[nodes[offset + local_index]] = [nodes[int(index)] for index in row]
        return result
    flat = values
    for query_index, node in enumerate(nodes):
        scores = np.einsum("rd,nsd->rns", flat[query_index], flat).max(axis=(0, 2))
        scores[query_index] = -np.inf
        indices = np.argpartition(scores, -maximum_k)[-maximum_k:]
        indices = indices[np.argsort(scores[indices])[::-1]]
        result[node] = [nodes[int(index)] for index in indices]
    return result


def _metrics(
    neighbors: dict[str, list[str]],
    pairs: list[dict[str, str]],
    k_values: tuple[int, ...],
    *,
    union_route_count: int = 1,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in ("calibration", "held_out_audit", "all"):
        positive_directions: list[tuple[str, str]] = []
        negative_directions: list[tuple[str, str]] = []
        exact_directions: list[tuple[str, str]] = []
        for row in pairs:
            if split != "all" and row["split"] != split:
                continue
            directions = [(row["node_u"], row["node_v"]), (row["node_v"], row["node_u"])]
            if row["binary_role"] == "positive":
                positive_directions.extend(directions)
                if row["label"] == "same_frame":
                    exact_directions.extend(directions)
            elif row["binary_role"] == "negative":
                negative_directions.extend(directions)
        prefix = f"{split}__"
        output[prefix + "positive_directions"] = len(positive_directions)
        output[prefix + "negative_directions"] = len(negative_directions)
        output[prefix + "exact_directions"] = len(exact_directions)
        for k in k_values:
            candidate_limit = k * union_route_count
            positive_hits = sum(
                target in neighbors.get(query, [])[:candidate_limit]
                for query, target in positive_directions
            )
            negative_hits = sum(
                target in neighbors.get(query, [])[:candidate_limit]
                for query, target in negative_directions
            )
            exact_hits = sum(
                target in neighbors.get(query, [])[:candidate_limit]
                for query, target in exact_directions
            )
            recall = positive_hits / len(positive_directions) if positive_directions else None
            low, high = _wilson(positive_hits, len(positive_directions))
            output[prefix + f"positive_recall_at_{k}"] = recall
            output[prefix + f"positive_recall_at_{k}_wilson_low"] = low
            output[prefix + f"positive_recall_at_{k}_wilson_high"] = high
            output[prefix + f"negative_top_at_{k}_rate"] = (
                negative_hits / len(negative_directions) if negative_directions else None
            )
            output[prefix + f"exact_recall_at_{k}"] = (
                exact_hits / len(exact_directions) if exact_directions else None
            )
    output["mean_candidates_at_max_k"] = float(
        np.mean(
            [len(set(values[: max(k_values) * union_route_count])) for values in neighbors.values()]
        )
    )
    output["union_k_is_per_route"] = union_route_count > 1
    return output


def _union_neighbors(
    route_neighbors: dict[str, dict[str, list[str]]], route_names: list[str], maximum_k: int
) -> dict[str, list[str]]:
    nodes = sorted(
        set.intersection(*(set(route_neighbors[name]) for name in route_names)), key=parse_node_uid
    )
    result = {}
    for node in nodes:
        ordered: list[str] = []
        seen = set()
        for rank in range(maximum_k):
            for name in route_names:
                values = route_neighbors[name][node]
                if rank < len(values) and values[rank] not in seen:
                    seen.add(values[rank])
                    ordered.append(values[rank])
        result[node] = ordered
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    k_values = tuple(sorted({int(value) for value in args.k_values.split(",")}))
    maximum_k = max(k_values)
    routes_path = Path(args.routes_json).expanduser().resolve()
    pairs_path = Path(args.calibration_pairs).expanduser().resolve()
    calibration_summary_path = Path(args.calibration_summary).expanduser().resolve()
    patch_decision_path = Path(args.patch_mask_decision).expanduser().resolve()
    route_config = json.loads(routes_path.read_text(encoding="utf-8"))
    routes = route_config.get("routes", [])
    if not routes or len({route["name"] for route in routes}) != len(routes):
        raise ValueError("routes must be non-empty and uniquely named")
    pairs = _read(pairs_path)
    calibration_summary = json.loads(calibration_summary_path.read_text(encoding="utf-8"))
    patch_decision = json.loads(patch_decision_path.read_text(encoding="utf-8"))
    route_neighbors: dict[str, dict[str, list[str]]] = {}
    rows = []
    for route in routes:
        nodes, values = _load_route(route)
        neighbors = _neighbors(
            nodes, values, maximum_k=maximum_k, device=args.device, batch_size=args.node_batch_size
        )
        route_neighbors[route["name"]] = neighbors
        rows.append(
            {
                "candidate_type": "individual",
                "candidate_name": route["name"],
                "route_count": 1,
                "routes": route["name"],
                **_metrics(neighbors, pairs, k_values),
            }
        )
    calibration_key = f"calibration__positive_recall_at_{maximum_k}"
    calibration_negative_key = f"calibration__negative_top_at_{maximum_k}_rate"
    individual = sorted(
        rows,
        key=lambda row: (
            -(row[calibration_key] if row[calibration_key] is not None else -1),
            row[calibration_negative_key]
            if row[calibration_negative_key] is not None
            else math.inf,
            row["candidate_name"],
        ),
    )
    top2 = [row["candidate_name"] for row in individual[:2]]
    gem = [route["name"] for route in routes if "gem" in route["name"].lower()]
    vlad = [route["name"] for route in routes if "vlad" in route["name"].lower()]
    unions: dict[str, list[str]] = {"top2_calibration": top2}
    if gem:
        unions["all_masked_gem"] = gem
    if vlad:
        unions["all_masked_vlad"] = vlad
        best_vlad = next(
            (row["candidate_name"] for row in individual if row["candidate_name"] in vlad), None
        )
        if best_vlad:
            unions["top2_plus_best_vlad"] = list(dict.fromkeys(top2 + [best_vlad]))
    for name, route_names in unions.items():
        neighbors = _union_neighbors(route_neighbors, route_names, maximum_k)
        rows.append(
            {
                "candidate_type": "union",
                "candidate_name": name,
                "route_count": len(route_names),
                "routes": "+".join(route_names),
                **_metrics(neighbors, pairs, k_values, union_route_count=len(route_names)),
            }
        )
    union_rows = [row for row in rows if row["candidate_type"] == "union"]
    selected = sorted(
        union_rows,
        key=lambda row: (
            -(row[calibration_key] if row[calibration_key] is not None else -1),
            row[calibration_negative_key]
            if row[calibration_negative_key] is not None
            else math.inf,
            row["route_count"],
            row["candidate_name"],
        ),
    )[0]
    heldout_key = f"held_out_audit__positive_recall_at_{maximum_k}"
    exact_key = f"all__exact_recall_at_{maximum_k}"
    technical_inputs_pass = bool(
        patch_decision.get("formal_patch_mask_admission")
        and calibration_summary.get("formal_threshold_admission")
        and calibration_summary.get("heldout_positive_direction_count", 0) >= 10
    )
    retrieval_pass = bool(
        selected.get(heldout_key) is not None
        and float(selected[heldout_key]) >= args.heldout_recall_target
        and (selected.get(exact_key) is None or float(selected[exact_key]) == 1.0)
    )
    admitted = technical_inputs_pass and retrieval_pass
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "round_b_descriptor_bakeoff.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    decision = {
        "status": "pass" if admitted else "insufficient_evidence_or_recall",
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "formal_descriptor_selection_admission": admitted,
        "formal_grouping_admission": False,
        "selected_candidate": selected["candidate_name"] if admitted else None,
        "selected_routes": selected["routes"].split("+") if admitted else [],
        "exploratory_best_candidate": selected["candidate_name"],
        "exploratory_best_routes": selected["routes"].split("+"),
        "selection_uses_heldout": False,
        "technical_inputs_pass": technical_inputs_pass,
        "retrieval_pass": retrieval_pass,
        "heldout_recall_target": args.heldout_recall_target,
        "selected_metrics": selected,
    }
    atomic_write_json(output / "round_b_decision.json", decision)
    summary = {
        "status": decision["status"],
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "route_count": len(routes),
        "candidate_count": len(rows),
        "routes_json_sha256": sha256_file(routes_path),
        "calibration_pairs_sha256": sha256_file(pairs_path),
        "calibration_summary_sha256": sha256_file(calibration_summary_path),
        "patch_mask_decision_sha256": sha256_file(patch_decision_path),
        "artifacts": {
            "round_b_descriptor_bakeoff.csv": sha256_file(csv_path),
            "round_b_decision.json": sha256_file(output / "round_b_decision.json"),
        },
    }
    atomic_write_json(output / "round_b_summary.json", summary)
    print(
        json.dumps(
            {"summary": summary, "decision": decision}, ensure_ascii=False, indent=2, sort_keys=True
        )
    )
    return 0 if admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
