#!/usr/bin/env python3
"""Fit 0-degree global PCA-whitening and project all VLAD cache rows to 512D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from rsdet.grouping.cache import PlaceFeatureCache, PlaceFeatureCacheWriter
from rsdet.grouping.contracts import MASKED_PATCH_PROTOCOL_VERSION, atomic_write_json, sha256_file
from rsdet.grouping.vlad import apply_global_pca, fit_global_pca


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project MAR20 VLAD cache")
    parser.add_argument("--input-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-dimension", type=int, default=512)
    parser.add_argument("--seed", type=int, default=202625)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = PlaceFeatureCache(args.input_cache)
    payload = source.load_all()
    rotation = payload["row__rotation"].astype(int)
    fit_mask = rotation == 0
    if not fit_mask.any():
        raise ValueError("VLAD cache has no 0-degree rows")
    projected: dict[str, np.ndarray] = {}
    pca_dir = Path(args.output_dir).expanduser().resolve() / "pca"
    pca_dir.mkdir(parents=True, exist_ok=True)
    pca_entries = []
    for feature in source.feature_names:
        values = np.asarray(payload[f"feature__{feature}"], dtype=np.float32)
        dimension = min(args.output_dimension, values[fit_mask].shape[0], values.shape[1])
        if dimension != args.output_dimension:
            raise ValueError(f"{feature}: requested PCA dimension not feasible")
        mean, components = fit_global_pca(
            values[fit_mask], output_dimension=dimension, seed=args.seed
        )
        name = f"{feature}_pca{dimension}"
        projected[name] = apply_global_pca(values, mean, components)
        path = pca_dir / f"{name}.npz"
        with path.open("wb") as file:
            np.savez(file, mean=mean, components=components)
        pca_entries.append(
            {
                "feature": feature,
                "projected_feature": name,
                "path": path.name,
                "sha256": sha256_file(path),
            }
        )
    output = Path(args.output_dir).expanduser().resolve()
    metadata = {
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "source_cache_fingerprint": source.index["fingerprint"],
        "source_index_sha256": sha256_file(Path(args.input_cache) / "index.json"),
        "fit_rows": int(fit_mask.sum()),
        "fit_rotation": 0,
        "output_dimension": args.output_dimension,
        "seed": args.seed,
        "pca_entries": pca_entries,
    }
    writer = PlaceFeatureCacheWriter(
        output / "cache", metadata=metadata, feature_names=tuple(projected), storage_dtype="float16"
    )
    row_keys = {
        key.removeprefix("row__"): value
        for key, value in payload.items()
        if key.startswith("row__")
    }
    writer.write_shard(0, rows=row_keys, features=projected)
    index = writer.finalize(expected_shards=1, expected_rows=len(rotation))
    audit = PlaceFeatureCache(output / "cache").audit()
    summary = {
        "status": "pass",
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "cache": audit,
        "index_sha256": sha256_file(output / "cache" / "index.json"),
        "pca_entries": pca_entries,
        "row_count": index["row_count"],
    }
    atomic_write_json(output / "projection_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
