#!/usr/bin/env python3
"""Compile conservative MAR20 source-proxy groups from frozen blind review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

STRICT_LABELS = {"same_frame", "geometric_overlap", "same_local_site"}
NEGATIVE_LABELS = {"not_same_local_site", "different_airport"}
WEAK_LABELS = {"likely_same_airport", "uncertain"}


class UnionFind:
    def __init__(self, nodes: list[str]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: str) -> str:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        self.parent[max(a, b)] = min(a, b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument("--guard-min-confidence", type=float, default=0.60)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def role(label: str) -> str:
    if label in STRICT_LABELS:
        return "positive"
    if label in NEGATIVE_LABELS:
        return "negative"
    return "uncertain"


def main() -> None:
    args = parse_args()
    registry = read_csv(args.registry)
    mapping = read_csv(args.mapping)
    decisions = {row["card_id"]: row for row in read_csv(args.decisions)}
    nodes = [row["node_uid"] for row in registry]
    if len(nodes) != 3842 or len(set(nodes)) != 3842:
        raise ValueError("registry must contain exactly 3,842 unique MAR20 nodes")
    if set(decisions) != {row["card_id"] for row in mapping}:
        raise ValueError("decision and private-mapping card IDs differ")

    duplicate_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in mapping:
        duplicate_groups[row["duplicate_group"]].append(row)
    repeated = [rows for rows in duplicate_groups.values() if len(rows) > 1]
    for rows in repeated:
        roles = {role(decisions[row["card_id"]]["label"]) for row in rows}
        if len(roles) != 1:
            raise ValueError(f"unresolved duplicate disagreement: {rows[0]['duplicate_group']}")

    pair_rows: dict[str, tuple[dict[str, str], dict[str, str]]] = {}
    for row in mapping:
        pair_rows.setdefault(row["pair_uid"], (row, decisions[row["card_id"]]))

    uf = UnionFind(nodes)
    guard_uf = UnionFind(nodes)
    strict_edges: list[dict[str, object]] = []
    weak_nodes: set[str] = set()
    negative_pairs: list[tuple[str, str, str]] = []
    for pair_uid, (mapped, decision) in sorted(pair_rows.items()):
        label = decision["label"]
        confidence = float(decision["confidence"])
        if label in STRICT_LABELS and confidence >= args.min_confidence:
            uf.union(mapped["node_u"], mapped["node_v"])
            guard_uf.union(mapped["node_u"], mapped["node_v"])
            strict_edges.append(
                {
                    "pair_uid": pair_uid,
                    "node_u": mapped["node_u"],
                    "node_v": mapped["node_v"],
                    "label": label,
                    "confidence": confidence,
                    "queue_grade": mapped["queue_grade"],
                    "target_relation": mapped["target_relation"],
                    "cross_official_side": mapped["cross_official_side"],
                }
            )
        elif label in WEAK_LABELS:
            known_negative_control = (
                mapped["is_control"] == "1" and mapped["expected_role"] == "negative"
            )
            if not known_negative_control:
                weak_nodes.update((mapped["node_u"], mapped["node_v"]))
            if (
                not known_negative_control
                and label == "likely_same_airport"
                and confidence >= args.guard_min_confidence
            ):
                guard_uf.union(mapped["node_u"], mapped["node_v"])
        elif label in NEGATIVE_LABELS:
            negative_pairs.append((pair_uid, mapped["node_u"], mapped["node_v"]))

    conflicts = [pair for pair in negative_pairs if uf.find(pair[1]) == uf.find(pair[2])]
    if conflicts:
        raise ValueError(f"strict graph violates {len(conflicts)} reviewed negative pairs")
    guard_conflicts = [
        pair for pair in negative_pairs if guard_uf.find(pair[1]) == guard_uf.find(pair[2])
    ]
    if guard_conflicts:
        raise ValueError(f"CV guard graph violates {len(guard_conflicts)} reviewed negative pairs")

    components: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        components[uf.find(node)].append(node)
    component_key: dict[str, str] = {}
    for root, members in components.items():
        minimum = min(int(node.split(":", 1)[1]) for node in members)
        component_key[root] = f"mar20-core-{minimum:04d}"
    guard_components: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        guard_components[guard_uf.find(node)].append(node)
    guard_component_key: dict[str, str] = {}
    for root, members in guard_components.items():
        minimum = min(int(node.split(":", 1)[1]) for node in members)
        guard_component_key[root] = f"mar20-guard-{minimum:04d}"

    registry_by_node = {row["node_uid"]: row for row in registry}
    target_counts = {
        root: sum(registry_by_node[node]["is_target"] == "1" for node in members)
        for root, members in components.items()
    }
    guard_target_counts = {
        root: sum(registry_by_node[node]["is_target"] == "1" for node in members)
        for root, members in guard_components.items()
    }
    output_rows: list[dict[str, object]] = []
    for row in registry:
        node = row["node_uid"]
        root = uf.find(node)
        guard_root = guard_uf.find(node)
        all_size = len(components[root])
        output_rows.append(
            {
                "node_uid": node,
                "mar20_number": row["mar20_number"],
                "competition_image_id": row["competition_image_id"],
                "is_target": row["is_target"],
                "is_bridge": row["is_bridge"],
                "official_side": row["official_side"],
                "group_id": component_key[root],
                "group_size_all": all_size,
                "group_size_target": target_counts[root],
                "cv_guard_group_id": guard_component_key[guard_root],
                "cv_guard_size_all": len(guard_components[guard_root]),
                "cv_guard_size_target": guard_target_counts[guard_root],
                "fold_group_id": guard_component_key[guard_root],
                "strict_core_member": int(all_size > 1),
                "unresolved_risk": int(node in weak_nodes),
                "core_only_embargo_recommended": int(
                    row["is_target"] == "1" and node in weak_nodes
                ),
                "group_semantics": "source_proxy_local_scene",
            }
        )

    controls = [row for row in mapping if row["is_control"] == "1"]
    control_counts = Counter(
        (row["expected_role"], role(decisions[row["card_id"]]["label"])) for row in controls
    )
    duplicate_role_agreement = sum(
        len({role(decisions[row["card_id"]]["label"]) for row in rows}) == 1 for rows in repeated
    )
    target_rows = [row for row in output_rows if row["is_target"] == "1"]
    target_components = Counter(row["group_id"] for row in target_rows)
    target_guard_components = Counter(row["cv_guard_group_id"] for row in target_rows)
    final_rows = []
    for row in target_rows:
        if int(row["group_size_target"]) > 1:
            evidence_level = "strict_core"
        elif int(row["cv_guard_size_target"]) > 1:
            evidence_level = "conservative_guard"
        else:
            evidence_level = "singleton"
        final_rows.append(
            {
                "competition_image_id": row["competition_image_id"],
                "mar20_number": row["mar20_number"],
                "group_id": row["fold_group_id"],
                "group_size": row["cv_guard_size_target"],
                "evidence_level": evidence_level,
                "group_semantics": "source_proxy_not_airport_ground_truth",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "mar20_group_assignments_all.csv",
        output_rows,
        list(output_rows[0]),
    )
    write_csv(
        args.output_dir / "mar20_target_core_group_assignments.csv",
        target_rows,
        list(target_rows[0]),
    )
    write_csv(
        args.output_dir / "strict_core_edges.csv",
        strict_edges,
        list(strict_edges[0]),
    )
    write_csv(
        args.output_dir / "mar20_final_group_assignments.csv",
        final_rows,
        list(final_rows[0]),
    )
    summary = {
        "status": "final_source_proxy_grouping_ready_for_cv3",
        "formal_grouping_admission": True,
        "group_semantics": "source_proxy_local_scene_not_airport_ground_truth",
        "min_confidence": args.min_confidence,
        "guard_min_confidence": args.guard_min_confidence,
        "registry_nodes": len(nodes),
        "target_nodes": len(target_rows),
        "bridge_nodes": len(nodes) - len(target_rows),
        "unique_reviewed_pairs": len(pair_rows),
        "strict_edges": len(strict_edges),
        "strict_graph_negative_conflicts": len(conflicts),
        "cv_guard_negative_conflicts": len(guard_conflicts),
        "duplicate_groups": len(repeated),
        "duplicate_role_agreement": duplicate_role_agreement / len(repeated),
        "control_confusion": {
            f"{a}_as_{b}": value for (a, b), value in sorted(control_counts.items())
        },
        "target_groups": len(target_components),
        "target_non_singleton_groups": sum(size > 1 for size in target_components.values()),
        "target_nodes_in_non_singletons": sum(
            size for size in target_components.values() if size > 1
        ),
        "target_core_only_embargo_recommended": sum(
            int(row["core_only_embargo_recommended"]) for row in target_rows
        ),
        "target_guard_groups": len(target_guard_components),
        "target_guard_non_singleton_groups": sum(
            size > 1 for size in target_guard_components.values()
        ),
        "target_guard_nodes_in_non_singletons": sum(
            size for size in target_guard_components.values() if size > 1
        ),
        "group_size_target_histogram": dict(sorted(Counter(target_components.values()).items())),
        "guard_size_target_histogram": dict(
            sorted(Counter(target_guard_components.values()).items())
        ),
        "cv_recommendation": "use fold_group_id; use group_id only for strict-core analysis",
        "inputs": {
            "registry_sha256": sha256(args.registry),
            "mapping_sha256": sha256(args.mapping),
            "decisions_sha256": sha256(args.decisions),
        },
    }
    summary_path = args.output_dir / "target_core_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
