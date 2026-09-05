#!/usr/bin/env python3
"""Audit full containment and context across the production and shifted grids."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsdet.tiling.boundary_geometry import (  # noqa: E402
    best_geometry,
    build_virtual_tiles,
    guaranteed_full_containment,
)


def _coarse(label: int) -> str:
    if 0 <= label <= 3:
        return "ship"
    if 4 <= label <= 23:
        return "aircraft"
    if label == 24:
        return "vehicle"
    raise ValueError(f"unsupported category_id={label}")


def _size_bin(width: float, height: float) -> str:
    side = max(width, height)
    for upper, name in ((48, "lt48"), (80, "48_80"), (128, "80_128"), (256, "128_256")):
        if side < upper:
            return name
    return "ge256"


def _parse_phases(raw: str) -> tuple[tuple[int, int], ...]:
    phases = tuple(
        tuple(int(value) for value in token.strip().split(":", 1))
        for token in raw.split(",")
        if token.strip()
    )
    if not phases or any(len(phase) != 2 for phase in phases):
        raise ValueError("phases must be comma-separated x:y pairs")
    return phases  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--phases", default="0:0,384:0,0:384,384:384")
    args = parser.parse_args()

    document = json.loads(args.coco.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in document["images"]}
    phases = _parse_phases(args.phases)
    grids: dict[tuple[int, int, int], object] = {}
    summary: dict[str, Counter[str]] = defaultdict(Counter)
    rows = []
    for offset, annotation in enumerate(document["annotations"]):
        image_id = int(annotation["image_id"])
        image = images[image_id]
        image_width = int(image["width"])
        image_height = int(image["height"])
        x, y, width, height = (float(value) for value in annotation["bbox"])
        box = (x, y, x + width, y + height)
        current_key = (image_id, 0, 0)
        if current_key not in grids:
            grids[current_key] = build_virtual_tiles(
                image_width, image_height, args.tile_size, args.overlap
            )
        current = best_geometry(box, grids[current_key])
        phase_results = []
        for phase_x, phase_y in phases:
            key = (image_id, phase_x, phase_y)
            if key not in grids:
                grids[key] = build_virtual_tiles(
                    image_width,
                    image_height,
                    args.tile_size,
                    args.overlap,
                    phase_x=phase_x,
                    phase_y=phase_y,
                    padded_phase=True,
                )
            phase_results.append(best_geometry(box, grids[key]))
        phase_full = any(bool(result["has_fully_contained_view"]) for result in phase_results)
        if bool(current["has_fully_contained_view"]):
            risk = "full_current"
        elif phase_full:
            risk = "phase_recovers_full"
        else:
            risk = "never_full"
        label = int(annotation["category_id"])
        guaranteed = guaranteed_full_containment(
            width,
            height,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )
        size_bin = _size_bin(width, height)
        for group in ("all", f"coarse:{_coarse(label)}", f"class:{label}", f"size:{size_bin}"):
            summary[group]["objects"] += 1
            summary[group][risk] += 1
            summary[group]["guaranteed_by_overlap"] += int(guaranteed)
        rows.append(
            {
                "annotation_id": int(annotation.get("id", offset)),
                "image_id": image_id,
                "category_id": label,
                "coarse": _coarse(label),
                "width": width,
                "height": height,
                "size_bin": size_bin,
                "guaranteed_by_overlap": guaranteed,
                "current": current,
                "phase_oracle_has_full_view": phase_full,
                "risk": risk,
            }
        )

    output = {
        "schema_version": "hera_guard_batis_tiling_geometry_v1",
        "source": str(args.coco),
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "stride": args.tile_size - args.overlap,
        "phases": [list(phase) for phase in phases],
        "image_count": len(images),
        "object_count": len(rows),
        "summary": {key: dict(value) for key, value in sorted(summary.items())},
        "objects": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "objects"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
