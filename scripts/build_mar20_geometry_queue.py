#!/usr/bin/env python3
"""Build a bounded, deterministic TASK-01 geometry queue."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageOps

from rsdet.grouping.contracts import atomic_write_json, canonical_pair_uid, sha256_file
from rsdet.grouping.geometry import phash64
from rsdet.grouping.registry import load_registry


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MAR20 geometry verification queue")
    parser.add_argument("--formal-candidates", required=True)
    parser.add_argument("--retrieval-decision", required=True)
    parser.add_argument("--calibration-pairs", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--mar20-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--maximum-pairs", type=int, default=12000)
    return parser.parse_args(argv)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _integer(row: dict[str, str], key: str, default: int) -> int:
    try:
        return int(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def _number(row: dict[str, str], key: str, default: float) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.maximum_pairs < 1000:
        raise ValueError("maximum-pairs below 1000 is not admitted")
    candidate_path = Path(args.formal_candidates).expanduser().resolve()
    decision_path = Path(args.retrieval_decision).expanduser().resolve()
    calibration_path = Path(args.calibration_pairs).expanduser().resolve()
    registry_path = Path(args.registry).expanduser().resolve()
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("formal_retrieval_admission") is not True:
        raise ValueError("retrieval decision is not admitted")
    candidates = _read(candidate_path)
    calibration = _read(calibration_path)
    registry = {row["node_uid"]: row for row in load_registry(registry_path)}
    fields = list(candidates[0])
    by_pair = {row["pair_uid"]: row for row in candidates}
    calibration_by_pair = {row["pair_uid"]: row for row in calibration}
    missing = [row for row in calibration if row["pair_uid"] not in by_pair]
    root = Path(args.mar20_root).expanduser().resolve()
    needed_nodes = {row[side] for row in missing for side in ("node_u", "node_v")}
    hashes = {}
    for uid in sorted(needed_nodes):
        path = (root / registry[uid]["original_relative_path"]).resolve()
        path.relative_to(root)
        with Image.open(path) as image:
            image.load()
            hashes[uid] = phash64(ImageOps.exif_transpose(image).convert("RGB"))
    for source in missing:
        pair_uid = canonical_pair_uid(source["node_u"], source["node_v"])
        u, v = pair_uid.split("--")
        target_u = registry[u]["is_target"] == "1"
        target_v = registry[v]["is_target"] == "1"
        row = {field: "" for field in fields}
        row.update(
            {
                "pair_uid": pair_uid,
                "node_u": u,
                "node_v": v,
                "scope": "target_only" if target_u and target_v else "full_bridge_diagnostic",
                "target_target": int(target_u and target_v),
                "target_bridge": int(target_u != target_v),
                "bridge_bridge": int(not target_u and not target_v),
                "cross_official_side": int(
                    registry[u]["official_side"] != registry[v]["official_side"]
                ),
                "exact_pixel": int(
                    registry[u]["original_pixel_sha256"] == registry[v]["original_pixel_sha256"]
                ),
                "phash_distance": int(np.count_nonzero(hashes[u] != hashes[v])),
                "in_formal_k50": 0,
                "in_audit_k100": 0,
                "formal_route_support": 0,
                "audit_route_support": 0,
                "formal_mutual_route_support": 0,
            }
        )
        by_pair[pair_uid] = row

    def priority(row: dict[str, str]) -> tuple[Any, ...]:
        return (
            0 if _integer(row, "exact_pixel", 0) else 1,
            -_integer(row, "formal_mutual_route_support", 0),
            -_integer(row, "formal_route_support", 0),
            _integer(row, "best_formal_rank", 999999),
            _integer(row, "phash_distance", 999),
            -_number(row, "best_similarity", -1.0),
            -_integer(row, "cross_official_side", 0),
            row["pair_uid"],
        )

    controls = []
    for pair_uid, source in calibration_by_pair.items():
        row = dict(by_pair[pair_uid])
        row.update(
            {
                "queue_source": "calibration_control",
                "known_split": source["split"],
                "known_binary_role": source["binary_role"],
                "known_label": source["label"],
            }
        )
        controls.append(row)
    selected_ids = {row["pair_uid"] for row in controls}
    pool = [row for pair_uid, row in by_pair.items() if pair_uid not in selected_ids]
    pool.sort(key=priority)
    remaining = max(args.maximum_pairs - len(controls), 0)
    selected = controls + [
        {
            **row,
            "queue_source": "formal_k50_candidate",
            "known_split": "",
            "known_binary_role": "",
            "known_label": "",
        }
        for row in pool[:remaining]
    ]
    if len(selected) < len(controls):
        raise RuntimeError("calibration controls were truncated")
    selected.sort(
        key=lambda row: (
            0 if row["queue_source"] == "calibration_control" else 1,
            priority(row),
        )
    )
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    queue_path = output / "geometry_queue.csv"
    output_fields = [
        *fields,
        "queue_source",
        "known_split",
        "known_binary_role",
        "known_label",
    ]
    with queue_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(({key: row.get(key, "") for key in output_fields} for row in selected))
    summary = {
        "status": "pass",
        "maximum_pairs": args.maximum_pairs,
        "queue_pair_count": len(selected),
        "calibration_control_count": len(controls),
        "calibration_missing_from_formal_k50_count": len(missing),
        "new_candidate_count": len(selected) - len(controls),
        "relation_counts": {
            "target_target": sum(_integer(row, "target_target", 0) for row in selected),
            "target_bridge": sum(_integer(row, "target_bridge", 0) for row in selected),
            "bridge_bridge": sum(_integer(row, "bridge_bridge", 0) for row in selected),
        },
        "exact_pixel_count": sum(_integer(row, "exact_pixel", 0) for row in selected),
        "two_route_count": sum(_integer(row, "formal_route_support", 0) >= 2 for row in selected),
        "retrieval_decision_sha256": sha256_file(decision_path),
        "formal_candidates_sha256": sha256_file(candidate_path),
        "calibration_pairs_sha256": sha256_file(calibration_path),
        "registry_sha256": sha256_file(registry_path),
        "artifact_sha256": sha256_file(queue_path),
        "formal_grouping_admission": False,
    }
    atomic_write_json(output / "geometry_queue_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
