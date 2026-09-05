#!/usr/bin/env python3
"""Evaluate one frozen OTO/OTM ownership route without selecting on labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sprint20.evaluation import cache_predictions, evaluate, gt_from_coco, route_dicts


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _count_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        label: {
            key: int(candidate["per_fine"][label][key]) - int(row[key])
            for key in ("tp", "fp", "fn")
        }
        for label, row in baseline["per_fine"].items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--oto-cache", type=Path, required=True)
    parser.add_argument("--otm-cache", type=Path, required=True)
    parser.add_argument("--primary-threshold", type=float, required=True)
    parser.add_argument("--alternative-threshold", type=float, required=True)
    parser.add_argument("--alternative-labels", type=int, nargs="+", required=True)
    parser.add_argument("--latency", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if len(args.alternative_labels) != len(set(args.alternative_labels)):
        raise ValueError("alternative labels must be unique")

    coco, oto_cache, otm_cache = _read(args.coco), _read(args.oto_cache), _read(args.otm_cache)
    if oto_cache.get("head") != "oto" or otm_cache.get("head") != "otm":
        raise ValueError("Expected aligned OTO and OTM caches")
    for key in (
        "role",
        "base_commit",
        "config_sha256",
        "coco_sha256",
        "weight_sha256",
        "pipeline",
        "inference_model",
        "aircraft_d4_applied",
        "threshold_stage",
        "cache_source",
    ):
        if oto_cache.get(key) != otm_cache.get(key):
            raise ValueError(f"Aligned caches changed {key}")
    if oto_cache["coco_sha256"] != _sha256(args.coco):
        raise ValueError("COCO provenance mismatch")
    oto_pixels = {int(row["image_id"]): row["pixel_sha256"] for row in oto_cache["images"]}
    otm_pixels = {int(row["image_id"]): row["pixel_sha256"] for row in otm_cache["images"]}
    if not oto_pixels or oto_pixels != otm_pixels:
        raise ValueError("Aligned caches changed image pixels")

    gt = gt_from_coco(coco)
    oto = cache_predictions(oto_cache, args.primary_threshold)
    otm = cache_predictions(otm_cache, args.alternative_threshold)
    if set(gt) != set(oto) or set(gt) != set(otm):
        raise ValueError("Caches do not cover the complete evaluation image set")
    candidate_pred = route_dicts(oto, otm, args.alternative_labels)
    baseline = evaluate(gt, oto, args.latency)
    candidate = evaluate(gt, candidate_pred, args.latency)
    payload = {
        "status": "complete_diagnostic_only",
        "role": oto_cache["role"],
        "cache_source": oto_cache.get("cache_source", "native_independent_forward"),
        "formal_admission": False,
        "policy": {
            "primary_threshold": args.primary_threshold,
            "alternative_threshold": args.alternative_threshold,
            "alternative_labels": args.alternative_labels,
            "latency_seconds": args.latency,
        },
        "input_sha256": {
            "coco": _sha256(args.coco),
            "oto_cache": _sha256(args.oto_cache),
            "otm_cache": _sha256(args.otm_cache),
        },
        "baseline": baseline,
        "candidate": candidate,
        "delta_score": candidate["score"]["total_score"] - baseline["score"]["total_score"],
        "count_delta_by_fine": _count_delta(baseline, candidate),
        "warning": (
            "This tool evaluates a pre-frozen policy. Its evidence strength remains the role "
            "declared by the input cache; full_seen labels cannot certify generalization."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "delta_score": payload["delta_score"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
