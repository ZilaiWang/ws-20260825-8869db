#!/usr/bin/env python3
"""Build the exact P40 + hierarchical Vehicle + Aircraft-D4 runtime config.

The input is the already-audited P40 + Aircraft-D4 config.  This command adds
one class-disjoint detector branch which may own Vehicle (fine 24) only.  It
hashes the deployment checkpoint at build time and validates the resulting
competition config before writing it, so an unfinished or stale checkpoint
cannot silently enter a submission candidate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from rsdet.submission.competition import load_submission_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_config(
    base: dict[str, Any],
    *,
    expert_weight: Path,
    expert_threshold: float,
    workpoint_id: str,
    primary_rescue_threshold: float | None = None,
    primary_rescue_dedup_iou: float | None = None,
) -> dict[str, Any]:
    if not expert_weight.is_absolute() or not expert_weight.is_file():
        raise FileNotFoundError(f"expert deployment weight missing: {expert_weight}")
    if not 0.0 <= expert_threshold <= 1.0:
        raise ValueError("expert_threshold must be in [0, 1]")
    if "aircraft_classifier_model" not in base:
        raise ValueError("base config must contain the admitted Aircraft-D4 module")
    if "resolution_expert_model" in base or "resolution_route" in base:
        raise ValueError("base config must not already contain a resolution expert")
    result = copy.deepcopy(base)
    primary_threshold = float(result.pop("post_fusion_score_threshold"))
    result["deployment_role"] = "pre_submission_runtime_candidate_p40_hier_aircraft_d4"
    result["workpoint_id"] = workpoint_id
    expert_model = copy.deepcopy(result["model"])
    expert_model["weight_path"] = str(expert_weight)
    expert_model["expected_sha256"] = sha256(expert_weight)
    result["resolution_expert_model"] = expert_model
    result["resolution_expert_pipeline"] = copy.deepcopy(result["pipeline"])
    result["resolution_route"] = {
        "primary_labels": list(range(24)),
        "expert_labels": [24],
        "primary_threshold": primary_threshold,
        "expert_threshold": float(expert_threshold),
    }
    if (primary_rescue_threshold is None) != (primary_rescue_dedup_iou is None):
        raise ValueError("primary rescue threshold and dedup_iou must be supplied together")
    if primary_rescue_threshold is not None and primary_rescue_dedup_iou is not None:
        if not 0.0 <= float(primary_rescue_threshold) <= 1.0:
            raise ValueError("primary_rescue_threshold must be in [0, 1]")
        if not 0.0 < float(primary_rescue_dedup_iou) <= 1.0:
            raise ValueError("primary_rescue_dedup_iou must be in (0, 1]")
        result["resolution_primary_rescue"] = {
            "labels": [24],
            "threshold": float(primary_rescue_threshold),
            "dedup_iou": float(primary_rescue_dedup_iou),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--expert-weight", type=Path, required=True)
    parser.add_argument("--expert-threshold", type=float, required=True)
    parser.add_argument("--workpoint-id", required=True)
    parser.add_argument("--primary-rescue-threshold", type=float)
    parser.add_argument("--primary-rescue-dedup-iou", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    result = build_config(
        base,
        expert_weight=args.expert_weight,
        expert_threshold=args.expert_threshold,
        workpoint_id=args.workpoint_id,
        primary_rescue_threshold=args.primary_rescue_threshold,
        primary_rescue_dedup_iou=args.primary_rescue_dedup_iou,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    try:
        load_submission_config(args.output)
    except Exception:
        args.output.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "expert_sha256": result["resolution_expert_model"]["expected_sha256"],
                "primary_threshold": result["resolution_route"]["primary_threshold"],
                "expert_threshold": result["resolution_route"]["expert_threshold"],
                "label_owners": {"p40": "0-23", "hierarchy": [24]},
                "aircraft_d4_enabled": True,
                "primary_vehicle_rescue": result.get("resolution_primary_rescue"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
