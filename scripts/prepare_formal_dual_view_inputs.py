#!/usr/bin/env python3
"""Freeze formal-CV3 Y5 candidates into the generic dual-view COCO contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.experiments.cv3_oof import sha256_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_registry(manifest: Path) -> dict[int, dict[str, Any]]:
    registry: dict[int, dict[str, Any]] = {}
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "formal_image_id",
            "fold",
            "group_id",
            "source_relative_path",
            "source_width",
            "source_height",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"formal manifest lacks image fields: {sorted(missing)}")
        for row in reader:
            image_id = int(row["formal_image_id"])
            item = {
                "id": image_id,
                "fold": int(row["fold"]),
                "group_id": str(row["group_id"]),
                "file_name": str(row["source_relative_path"]),
                "width": int(row["source_width"]),
                "height": int(row["source_height"]),
            }
            previous = registry.setdefault(image_id, item)
            if previous != item:
                raise ValueError(f"inconsistent image metadata for formal_image_id={image_id}")
    return registry


def _normalize_candidates(
    raw: list[dict[str, Any]],
    *,
    registry: dict[int, dict[str, Any]],
    label: str,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        image_id = int(item["image_id"])
        if image_id not in registry:
            raise ValueError(f"{label} references unknown image_id={image_id}")
        box = [float(value) for value in item["bbox_xyxy"]]
        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"{label} candidate {index} has invalid xyxy box")
        result.append(
            {
                "proposal_uid": str(item.get("proposal_uid", f"formal-p{index:06d}")),
                "proposal_index": index,
                "image_id": image_id,
                "category_id": int(item["category_id"]),
                "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                "score": float(item["score"]),
                "source_fold": int(registry[image_id]["fold"]),
                "source_model": str(item.get("source_model", label)),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-crop-manifest", type=Path, required=True)
    parser.add_argument("--evidence-predictions", type=Path, required=True)
    parser.add_argument("--anchor-predictions", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    registry = _image_registry(args.formal_crop_manifest)
    if set(registry) != set(formal.image_ids):
        raise ValueError("formal GT and image registry disagree")
    missing_images = [
        item["file_name"]
        for item in registry.values()
        if not (args.image_root / item["file_name"]).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            f"formal image root is incomplete: {len(missing_images)} missing; first={missing_images[0]}"
        )

    evidence_raw = json.loads(args.evidence_predictions.read_text(encoding="utf-8"))
    anchor_raw = json.loads(args.anchor_predictions.read_text(encoding="utf-8"))
    evidence = _normalize_candidates(evidence_raw, registry=registry, label="evidence")
    anchor = _normalize_candidates(anchor_raw, registry=registry, label="anchor")
    evidence_keys = [
        (row["proposal_uid"], row["image_id"], row["category_id"], row["bbox"])
        for row in evidence
    ]
    anchor_keys = [
        (row["proposal_uid"], row["image_id"], row["category_id"], row["bbox"])
        for row in anchor
    ]
    if evidence_keys != anchor_keys:
        raise ValueError("evidence and anchor candidate ledgers are not exactly aligned")

    categories = sorted({obj.category_id for obj in formal.objects.values()})
    annotations = []
    for annotation_id, obj in enumerate(formal.objects.values(), 1):
        x0, y0, x1, y1 = obj.bbox_xyxy
        annotations.append(
            {
                "id": annotation_id,
                "image_id": int(obj.image_id),
                "category_id": int(obj.category_id),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": (x1 - x0) * (y1 - y0),
                "iscrowd": 0,
            }
        )
    gt = {
        "images": [registry[key] for key in sorted(registry)],
        "annotations": annotations,
        "categories": [{"id": value, "name": str(value)} for value in categories],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt_path = args.output_dir / "ground_truth.json"
    evidence_path = args.output_dir / "evidence_predictions.json"
    anchor_path = args.output_dir / "anchor_predictions.json"
    gt_path.write_text(json.dumps(gt, ensure_ascii=False) + "\n", encoding="utf-8")
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False) + "\n", encoding="utf-8")
    anchor_path.write_text(json.dumps(anchor, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "status": "formal_dual_view_inputs_ready",
        "protocol": "formal_cv3_exact_candidate_alignment_v1",
        "counts": {
            "images": len(registry),
            "annotations": len(annotations),
            "candidates": len(evidence),
            "fold_images": {
                str(fold): sum(item["fold"] == fold for item in registry.values())
                for fold in (0, 1, 2)
            },
        },
        "input_sha256": {
            "formal_crop_manifest": sha256_file(args.formal_crop_manifest),
            "evidence_predictions": _sha256(args.evidence_predictions),
            "anchor_predictions": _sha256(args.anchor_predictions),
        },
        "output_sha256": {
            "ground_truth": _sha256(gt_path),
            "evidence_predictions": _sha256(evidence_path),
            "anchor_predictions": _sha256(anchor_path),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
