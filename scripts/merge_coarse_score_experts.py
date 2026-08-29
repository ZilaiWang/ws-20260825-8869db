#!/usr/bin/env python3
"""Merge aligned proposal scores from one expert per official coarse class."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def proposal_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(item["image_id"]),
        int(item["category_id"]),
        *(round(float(value), 5) for value in item["bbox"]),
        int(item.get("source_fold", -1)),
    )


def align_scores(
    reference: Sequence[Mapping[str, Any]], expert: Sequence[Mapping[str, Any]]
) -> list[float]:
    scores: dict[tuple[Any, ...], float] = {}
    for item in expert:
        key = proposal_key(item)
        if key in scores:
            raise ValueError(f"duplicate expert proposal: {key}")
        scores[key] = float(item["score"])
    aligned: list[float] = []
    for item in reference:
        key = proposal_key(item)
        if key not in scores:
            raise ValueError(f"expert proposal missing: {key}")
        aligned.append(scores.pop(key))
    if scores:
        raise ValueError(f"expert has {len(scores)} unmatched proposals")
    return aligned


def merge_scores(
    reference: Sequence[Mapping[str, Any]],
    experts: Mapping[str, Sequence[float]],
    *,
    routes: Mapping[str, str],
    category_mapping: Mapping[int, str],
) -> list[dict[str, Any]]:
    if set(routes) != {"ship", "aircraft", "vehicle"}:
        raise ValueError("routes must define ship, aircraft and vehicle")
    if any(name not in experts for name in routes.values()):
        raise ValueError("route refers to an unknown expert")
    if any(len(values) != len(reference) for values in experts.values()):
        raise ValueError("expert scores do not align with reference length")
    output: list[dict[str, Any]] = []
    for index, item in enumerate(reference):
        coarse = category_mapping[int(item["category_id"])]
        expert = routes[coarse]
        row = dict(item)
        row["score"] = float(experts[expert][index])
        row["score_expert"] = expert
        output.append(row)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--expert", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--ship", required=True)
    parser.add_argument("--aircraft", required=True)
    parser.add_argument("--vehicle", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    expert_paths = {name: Path(path) for name, path in args.expert}
    if len(expert_paths) != len(args.expert):
        raise ValueError("expert names must be unique")
    experts = {
        name: align_scores(reference, json.loads(path.read_text(encoding="utf-8")))
        for name, path in expert_paths.items()
    }
    routes = {"ship": args.ship, "aircraft": args.aircraft, "vehicle": args.vehicle}
    output = merge_scores(
        reference,
        experts,
        routes=routes,
        category_mapping=protocol.category_mapping,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": "complete",
        "protocol": "aligned_coarse_score_expert_merge_v1",
        "warning": "Expert routing is exploratory until confirmed on an independent split.",
        "routes": routes,
        "proposal_count": len(output),
        "inputs": {
            "reference": {"path": str(args.reference.resolve()), "sha256": _sha256(args.reference)},
            "experts": {
                name: {"path": str(path.resolve()), "sha256": _sha256(path)}
                for name, path in expert_paths.items()
            },
        },
        "output_sha256": _sha256(args.output),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
