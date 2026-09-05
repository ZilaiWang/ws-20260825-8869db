#!/usr/bin/env python3
"""Aggregate fold-aligned Sprint20 OTO/OTM caches into one short-OOF pair.

The default ``native`` source preserves the original two-forward diagnostic.
``shared`` extracts both heads from the exact shared-forward cache used by the
deployment path.  Keeping the two sources explicit prevents a failed native vs
shared parity audit from being silently reinterpreted as deployment evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_fold_pair(input_dir: Path, fold: int, source: str) -> dict[str, dict[str, Any]]:
    if source == "native":
        return {
            head: _read(input_dir / f"fold_{fold}/native_{head}.json")
            for head in ("oto", "otm")
        }

    cache = _read(input_dir / f"fold_{fold}/shared.json")
    if cache.get("head") != "shared" or cache.get("role") != "outer_oof_short":
        raise ValueError(f"Invalid shared fold {fold} identity")
    split: dict[str, dict[str, Any]] = {}
    for head in ("oto", "otm"):
        rows = []
        for row in cache["images"]:
            if "otm_prediction" not in row:
                raise ValueError(f"Shared fold {fold} row lacks otm_prediction")
            output = {key: value for key, value in row.items() if key != "otm_prediction"}
            if head == "otm":
                output["prediction"] = row["otm_prediction"]
            rows.append(output)
        split[head] = {**cache, "head": head, "images": rows}
    return split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--aggregate-coco", type=Path, required=True)
    parser.add_argument("--output-oto", type=Path, required=True)
    parser.add_argument("--output-otm", type=Path, required=True)
    parser.add_argument(
        "--source",
        choices=("native", "shared"),
        default="native",
        help="Read native_oto/native_otm pairs or split a deployment shared-forward cache.",
    )
    args = parser.parse_args()

    coco = _read(args.aggregate_coco)
    expected_ids = {int(row["id"]) for row in coco["images"]}
    aggregate: dict[str, dict[str, Any]] = {}
    fold_pairs = [_load_fold_pair(args.input_dir, fold, args.source) for fold in (0, 1, 2)]
    for head in ("oto", "otm"):
        fold_caches = [pair[head] for pair in fold_pairs]
        for fold, cache in enumerate(fold_caches):
            if cache.get("head") != head or cache.get("role") != "outer_oof_short":
                raise ValueError(f"Invalid {head} fold {fold} identity")
            if any(int(row.get("fold", -1)) != fold for row in cache["images"]):
                raise ValueError(f"Fold metadata mismatch in {head} fold {fold}")
        for key in ("base_commit", "pipeline", "aircraft_d4_applied", "threshold_stage"):
            if len({json.dumps(cache.get(key), sort_keys=True) for cache in fold_caches}) != 1:
                raise ValueError(f"Fold caches changed {key}")
        rows = sorted(
            [row for cache in fold_caches for row in cache["images"]],
            key=lambda row: int(row["image_id"]),
        )
        ids = [int(row["image_id"]) for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != expected_ids:
            raise ValueError(f"{head} aggregate does not cover each COCO image exactly once")
        model = dict(fold_caches[0]["inference_model"])
        model["weight_path"] = "PER_FOLD_OUTER_OOF"
        model["expected_sha256"] = {
            str(fold): cache["weight_sha256"] for fold, cache in enumerate(fold_caches)
        }
        aggregate[head] = {
            **{
                key: fold_caches[0][key]
                for key in (
                    "schema",
                    "base_commit",
                    "pipeline",
                    "aircraft_d4_applied",
                    "threshold_stage",
                )
            },
            "head": head,
            "role": "outer_oof_short",
            "config_sha256": {
                str(fold): cache["config_sha256"] for fold, cache in enumerate(fold_caches)
            },
            "coco_sha256": _sha256(args.aggregate_coco),
            "weight_sha256": {
                str(fold): cache["weight_sha256"] for fold, cache in enumerate(fold_caches)
            },
            "inference_model": model,
            "formal_admission": False,
            "cache_source": args.source,
            "lineage_warning": (
                "Source-disjoint OOF, but historical P40 folds use S1024/40e -> P40/40e, "
                "not the full model's S1024/160e -> P40/40e maturity."
            ),
            "images": rows,
        }
    oto_pixels = {int(row["image_id"]): row["pixel_sha256"] for row in aggregate["oto"]["images"]}
    otm_pixels = {int(row["image_id"]): row["pixel_sha256"] for row in aggregate["otm"]["images"]}
    if oto_pixels != otm_pixels:
        raise ValueError("Aggregated OTO/OTM pixels differ")
    _write(args.output_oto, aggregate["oto"])
    _write(args.output_otm, aggregate["otm"])
    print(
        json.dumps(
            {
                "status": "complete",
                "role": "outer_oof_short",
                "source": args.source,
                "images": len(expected_ids),
                "oto_sha256": _sha256(args.output_oto),
                "otm_sha256": _sha256(args.output_otm),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
