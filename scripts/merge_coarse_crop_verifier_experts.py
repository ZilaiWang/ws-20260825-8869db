#!/usr/bin/env python3
"""Route aligned crop-verifier evidence by official coarse class."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

CROP_FIELDS = (
    "score",
    "foreground_probability",
    "crop_class_probability",
    "crop_conditional_class_probability",
    "crop_top1_class",
    "crop_top1",
    "crop_top1_absolute",
    "crop_margin",
    "crop_entropy",
    "detector_crop_agree",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coarse_name(category_id: int) -> str:
    if 0 <= category_id < 4:
        return "ship"
    if 4 <= category_id < 24:
        return "aircraft"
    if category_id == 24:
        return "vehicle"
    raise ValueError(f"invalid category_id={category_id}")


def proposal_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        int(item["image_id"]),
        int(item["category_id"]),
        *(round(float(value), 5) for value in item["bbox"]),
        int(item.get("source_fold", -1)),
    )


def route_evidence(
    reference: Sequence[Mapping[str, Any]],
    experts: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    routes: Mapping[str, str],
) -> list[dict[str, Any]]:
    aligned: dict[str, dict[tuple[Any, ...], Mapping[str, Any]]] = {}
    reference_keys = {proposal_key(item) for item in reference}
    if len(reference_keys) != len(reference):
        raise ValueError("duplicate proposal in reference")
    for name, rows in experts.items():
        lookup = {proposal_key(item): item for item in rows}
        if len(lookup) != len(rows):
            raise ValueError(f"duplicate proposal in expert {name}")
        if set(lookup) != reference_keys:
            raise ValueError(f"expert {name} proposal set does not match reference")
        aligned[name] = lookup
    output: list[dict[str, Any]] = []
    for item in reference:
        key = proposal_key(item)
        coarse = coarse_name(int(item["category_id"]))
        expert_name = routes[coarse]
        source = aligned[expert_name].get(key)
        if source is None:
            raise ValueError(f"expert {expert_name} lacks proposal {key}")
        row = dict(item)
        for field in CROP_FIELDS:
            if field not in source:
                raise ValueError(f"expert {expert_name} lacks field {field}")
            row[field] = source[field]
        row["crop_evidence_expert"] = expert_name
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
    args = parser.parse_args()
    paths = {name: Path(path) for name, path in args.expert}
    if len(paths) != len(args.expert):
        raise ValueError("expert names must be unique")
    routes = {"ship": args.ship, "aircraft": args.aircraft, "vehicle": args.vehicle}
    if any(name not in paths for name in routes.values()):
        raise ValueError("route refers to an unknown expert")
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    experts = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    output = route_evidence(reference, experts, routes=routes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": "complete",
        "protocol": "aligned_coarse_crop_evidence_expert_route_v1",
        "routes": routes,
        "proposal_count": len(output),
        "reference_sha256": _sha256(args.reference),
        "expert_sha256": {name: _sha256(path) for name, path in paths.items()},
        "output_sha256": _sha256(args.output),
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
