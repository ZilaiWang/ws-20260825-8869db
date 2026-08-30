#!/usr/bin/env python3
"""Build a finite, preregistered class-routed TTA strategy set from cached views.

This is a causal readout, not a fusion-weight search.  It tests view replacement
and same-fine-supported augmentation while aircraft is protected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.official_metric import compute_iou
from rsdet.postprocess.nms import class_aware_nms_predictions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(path: Path, source_view: str) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    output = []
    for order, raw in enumerate(rows):
        if "bbox_xyxy" in raw:
            box = [float(value) for value in raw["bbox_xyxy"]]
        else:
            x, y, width, height = (float(value) for value in raw["bbox"])
            box = [x, y, x + width, y + height]
        output.append(
            {
                "image_id": int(raw["image_id"]),
                "category_id": int(raw["category_id"]),
                "bbox_xyxy": box,
                "score": float(raw["score"]),
                "source_view": source_view,
                "source_order": order,
            }
        )
    return output


def _coarse(category_id: int) -> str:
    if category_id <= 3:
        return "ship"
    if category_id <= 23:
        return "aircraft"
    return "vehicle"


def _supported_rotated(identity: list[dict], rotated: list[dict], threshold: float) -> list[dict]:
    identity_by_key: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    for row in identity:
        identity_by_key[(row["image_id"], row["category_id"])].append(row["bbox_xyxy"])
    output = []
    for row in rotated:
        if _coarse(row["category_id"]) == "aircraft":
            continue
        best = max(
            (
                compute_iou(row["bbox_xyxy"], box)
                for box in identity_by_key.get((row["image_id"], row["category_id"]), ())
            ),
            default=0.0,
        )
        if best >= threshold:
            output.append({**row, "same_fine_identity_iou": best})
    return output


def _export(rows: list[dict], output: Path, nms_iou: float) -> int:
    by_image: dict[int, list[dict]] = defaultdict(list)
    for source_index, raw in enumerate(rows):
        by_image[int(raw["image_id"])].append(
            {**raw, "source_prediction_index": source_index}
        )
    kept = class_aware_nms_predictions(dict(by_image), nms_iou)
    records = []
    for image_id in sorted(kept):
        for row in kept[image_id]:
            x0, y0, x1, y1 = (float(value) for value in row["bbox_xyxy"])
            records.append(
                {
                    "image_id": image_id,
                    "category_id": int(row["category_id"]),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "score": float(row["score"]),
                    "source_prediction_index": int(row["source_prediction_index"]),
                }
            )
    output.write_text(json.dumps(records, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--rot90", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-iou", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    args = parser.parse_args()
    if not 0.0 <= args.support_iou <= 1.0:
        raise ValueError("support-iou must be within [0, 1]")
    if not 0.0 < args.nms_iou <= 1.0:
        raise ValueError("nms-iou must be within (0, 1]")
    identity = _normalize(args.identity, "identity")
    rotated = _normalize(args.rot90, "rot90")
    supported = _supported_rotated(identity, rotated, args.support_iou)

    def select(rows: list[dict], coarse_names: set[str]) -> list[dict]:
        return [row for row in rows if _coarse(row["category_id"]) in coarse_names]

    strategies = {
        "identity": identity,
        "rot90": rotated,
        "identity_aircraft_rot_ship_vehicle": (
            select(identity, {"aircraft"}) + select(rotated, {"ship", "vehicle"})
        ),
        "identity_aircraft_vehicle_rot_ship": (
            select(identity, {"aircraft", "vehicle"}) + select(rotated, {"ship"})
        ),
        "identity_aircraft_ship_rot_vehicle": (
            select(identity, {"aircraft", "ship"}) + select(rotated, {"vehicle"})
        ),
        "identity_plus_supported_rot_ship_vehicle": identity + supported,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, rows in strategies.items():
        counts[name] = {
            "before_nms": len(rows),
            "after_nms": _export(rows, args.output_dir / f"{name}.json", args.nms_iou),
        }
    audit = {
        "status": "complete",
        "protocol": "finite_class_routed_cached_tta_v1",
        "support_iou": args.support_iou,
        "nms_iou": args.nms_iou,
        "identity_rows": len(identity),
        "rot90_rows": len(rotated),
        "supported_rotated_rows": len(supported),
        "strategies": counts,
        "input_sha256": {
            "identity": _sha256(args.identity),
            "rot90": _sha256(args.rot90),
        },
    }
    (args.output_dir / "build_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
