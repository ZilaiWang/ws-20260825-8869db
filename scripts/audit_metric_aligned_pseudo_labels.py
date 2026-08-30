#!/usr/bin/env python3
"""E0: audit official one-winner proposal roles on pseudo-10K candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.coco import load_coco_ground_truth
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.hera_guard.metric_aligned import build_metric_aligned_roles
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--background-iou", type=float, default=0.05)
    args = parser.parse_args()

    predictions = json.loads(args.pred.read_text(encoding="utf-8"))
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    result = build_metric_aligned_roles(
        gt_boxes=load_coco_ground_truth(args.gt),
        predictions=predictions,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        background_iou=args.background_iou,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "metric_aligned_roles.csv"
    fields = list(result.roles[0].__dataclass_fields__)
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: getattr(row, field) for field in fields} for row in result.roles)
    by_predicted_coarse = Counter(
        f"{row.predicted_coarse}/{row.role}" for row in result.roles
    )
    by_support_coarse = Counter(
        f"{row.support_coarse or 'none'}/{row.role}" for row in result.roles
    )
    payload = {
        "status": "metric_aligned_label_audit_complete",
        "protocol": "official_prediction_first_one_winner_pseudo_v1",
        "input_candidates": len(predictions),
        "official_tp": result.official_tp,
        "official_fp": result.official_fp,
        "role_counts": result.role_counts,
        "by_predicted_coarse": dict(sorted(by_predicted_coarse.items())),
        "by_support_coarse": dict(sorted(by_support_coarse.items())),
        "object_group_count": result.object_group_count,
        "max_group_size": result.max_group_size,
        "cross_coarse_foreground_pollution": result.cross_coarse_foreground_pollution,
        "background_iou": args.background_iou,
        "input_sha256": {"gt": _sha256(args.gt), "pred": _sha256(args.pred)},
        "roles_sha256": _sha256(rows_path),
    }
    (output / "audit_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
