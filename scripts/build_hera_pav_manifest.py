#!/usr/bin/env python3
"""Build the official-label two-view manifest for HERA Proposal Verifier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oer_labels import build_official_proposal_labels
from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.analysis.workpoint_labels import build_workpoint_labels
from rsdet.evaluation.official_frontier import official_fixed_risk_frontier
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.hera_guard.labels import build_proposal_object_labels
from rsdet.hera_guard.manifest import (
    PAV_MANIFEST_VERSION,
    PAV_METADATA_COLUMNS,
    metadata_from_node,
    square_crop_box,
)
from rsdet.utils.config import load_config

COARSE_INDEX = {"aircraft": 0, "ship": 1, "vehicle": 2}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _source_ledger(path: Path) -> dict[int, dict[str, str]]:
    ledger: dict[int, dict[str, str]] = {}
    for row in _read_csv(path):
        image_id = int(row["formal_image_id"])
        item = {
            "source_relative_path": row["source_relative_path"],
            "fold": row["fold"],
            "group_id": row["group_id"],
        }
        previous = ledger.setdefault(image_id, item)
        if previous != item:
            raise ValueError(f"inconsistent source ledger for image {image_id}")
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--label-predictions",
        type=Path,
        default=None,
        help="Frozen proposal scores used only for official foreground matching; defaults to --predictions",
    )
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--formal-crop-manifest", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tight-scale", type=float, default=1.10)
    parser.add_argument("--context-scale", type=float, default=1.60)
    parser.add_argument("--target-fdr", type=float, default=0.12)
    args = parser.parse_args()
    if args.context_scale <= args.tight_scale:
        raise ValueError("context scale must exceed tight scale")

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    source_ledger = _source_ledger(args.formal_crop_manifest)
    nodes = _read_csv(args.nodes)
    node_by_idx = {int(row["idx"]): row for row in nodes}
    if len(node_by_idx) != len(nodes):
        raise ValueError("nodes contain duplicate idx values")
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    label_path = args.label_predictions or args.predictions
    label_predictions = json.loads(label_path.read_text(encoding="utf-8"))
    if len(predictions) != len(nodes):
        raise ValueError("predictions/nodes row counts differ")
    if len(label_predictions) != len(predictions):
        raise ValueError("label/ranking prediction row counts differ")

    normalized = []
    normalized_labels = []
    for candidate_id, raw in enumerate(predictions):
        image_id = int(raw["image_id"])
        record = {
            "image_id": image_id,
            "category_id": int(raw["category_id"]),
            "score": float(raw["score"]),
            "bbox_xyxy": [float(value) for value in raw["bbox_xyxy"]],
            "source_prediction_index": candidate_id,
        }
        if image_id not in source_ledger or candidate_id not in node_by_idx:
            raise ValueError(f"candidate {candidate_id} lacks source/node provenance")
        normalized.append(record)
        label_raw = label_predictions[candidate_id]
        label_record = {
            "image_id": int(label_raw["image_id"]),
            "category_id": int(label_raw["category_id"]),
            "score": float(label_raw["score"]),
            "bbox_xyxy": [float(value) for value in label_raw["bbox_xyxy"]],
            "source_prediction_index": candidate_id,
        }
        if (
            label_record["image_id"] != record["image_id"]
            or label_record["category_id"] != record["category_id"]
            or label_record["bbox_xyxy"] != record["bbox_xyxy"]
        ):
            raise ValueError(f"label/ranking candidate identity differs at {candidate_id}")
        normalized_labels.append(label_record)

    proposal_labels = build_official_proposal_labels(
        gt_boxes=formal.boxes,
        predictions=normalized_labels,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    object_labels = build_proposal_object_labels(
        gt_boxes=formal.boxes,
        predictions=normalized_labels,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    frontier = official_fixed_risk_frontier(
        gt_boxes=formal.boxes,
        predictions=normalized,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        fdr_levels=(args.target_fdr,),
        nms_iou=0.50,
    )
    workpoint = build_workpoint_labels(frontier, target_fdr=args.target_fdr)

    rows: list[dict[str, object]] = []
    role_counts: Counter[str] = Counter()
    fine_counts: Counter[int] = Counter()
    for candidate_id, pred in enumerate(normalized):
        node = node_by_idx[candidate_id]
        image_id = pred["image_id"]
        detector_category = pred["category_id"]
        detector_coarse = protocol.category_mapping[detector_category]
        proposal_label = proposal_labels.labels[candidate_id]
        object_label = object_labels[candidate_id]
        role = workpoint.get(candidate_id)
        workpoint_role = role.role if role is not None else "inactive_tail"
        role_counts[workpoint_role] += 1
        target_fine = object_label.matched_category_id if object_label.is_object else -1
        target_coarse_name = object_label.matched_coarse or detector_coarse
        if target_fine >= 0:
            fine_counts[target_fine] += 1
        tight = square_crop_box(pred["bbox_xyxy"], scale=args.tight_scale)
        context = square_crop_box(pred["bbox_xyxy"], scale=args.context_scale)
        source = source_ledger[image_id]
        row: dict[str, object] = {
            "manifest_version": PAV_MANIFEST_VERSION,
            "candidate_id": candidate_id,
            "proposal_uid": node["proposal_uid"],
            "image_id": image_id,
            "fold": int(source["fold"]),
            "group_id": source["group_id"],
            "source_relative_path": source["source_relative_path"],
            "detector_category_id": detector_category,
            "detector_coarse": detector_coarse,
            "proposal_x0": pred["bbox_xyxy"][0],
            "proposal_y0": pred["bbox_xyxy"][1],
            "proposal_x1": pred["bbox_xyxy"][2],
            "proposal_y1": pred["bbox_xyxy"][3],
            "tight_x0": tight[0],
            "tight_y0": tight[1],
            "tight_x1": tight[2],
            "tight_y1": tight[3],
            "context_x0": context[0],
            "context_y0": context[1],
            "context_x1": context[2],
            "context_y1": context[3],
            "target_foreground": int(object_label.is_object),
            "target_coarse": COARSE_INDEX[target_coarse_name],
            "target_fine": target_fine,
            "target_quality": object_label.matched_iou,
            "target_protect": int(workpoint_role == "protected_tp"),
            "target_active_fp": int(workpoint_role == "active_fp"),
            "target_official_tp": int(proposal_label.is_valid),
            "workpoint_role": workpoint_role,
            "official_match_iou": proposal_label.matched_iou,
        }
        row.update(metadata_from_node(node, coarse_name=detector_coarse))
        rows.append(row)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "hera_pav_manifest.csv"
    fieldnames = list(rows[0])
    if fieldnames[-len(PAV_METADATA_COLUMNS) :] != list(PAV_METADATA_COLUMNS):
        raise RuntimeError("metadata columns are not in frozen order")
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "ready_for_pav_fast_screen",
        "manifest_version": PAV_MANIFEST_VERSION,
        "official_label_contract": proposal_labels.contract_version,
        "matching_policy": proposal_labels.metrics.details["matching_policy"],
        "n_rows": len(rows),
        "n_images": len({row["image_id"] for row in rows}),
        "target_fdr": args.target_fdr,
        "tight_scale": args.tight_scale,
        "context_scale": args.context_scale,
        "role_counts": dict(sorted(role_counts.items())),
        "foreground_count": sum(int(row["target_foreground"]) for row in rows),
        "official_tp_count": sum(int(row["target_official_tp"]) for row in rows),
        "active_fp_count": sum(int(row["target_active_fp"]) for row in rows),
        "fine_counts": {str(key): value for key, value in sorted(fine_counts.items())},
        "manifest_sha256": _sha256(manifest_path),
        "input_sha256": {
            "predictions": _sha256(args.predictions),
            "label_predictions": _sha256(label_path),
            "nodes": _sha256(args.nodes),
            "formal_crop_manifest": _sha256(args.formal_crop_manifest),
        },
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
