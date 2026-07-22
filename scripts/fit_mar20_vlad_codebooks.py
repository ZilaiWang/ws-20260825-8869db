#!/usr/bin/env python3
"""Fit image-balanced local PCA and VLAD codebooks from 00B patch samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from rsdet.grouping.contracts import MASKED_PATCH_PROTOCOL_VERSION, atomic_write_json, sha256_file
from rsdet.grouping.vlad import fit_local_pca_vlad


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit MAR20 masked-patch VLAD codebooks")
    parser.add_argument("--patch-sample-dir", required=True)
    parser.add_argument("--extraction-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="9,10,11")
    parser.add_argument("--cluster-counts", default="16,32")
    parser.add_argument("--local-dimension", type=int, default=128)
    parser.add_argument("--seed", type=int, default=202625)
    return parser.parse_args(argv)


def _integers(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError("integer list must be non-empty and unique")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    layers = _integers(args.layers)
    clusters = _integers(args.cluster_counts)
    sample_dir = Path(args.patch_sample_dir).expanduser().resolve()
    extraction_summary_path = Path(args.extraction_summary).expanduser().resolve()
    extraction = json.loads(extraction_summary_path.read_text(encoding="utf-8"))
    if extraction.get("status") != "pass":
        raise ValueError("masked-patch extraction did not pass")
    if extraction.get("protocol_version") != MASKED_PATCH_PROTOCOL_VERSION:
        raise ValueError("masked-patch protocol mismatch")
    files = sorted(sample_dir.glob("sample-shard-*.npz"))
    if len(files) != int(extraction["sample_shard_count"]):
        raise ValueError("patch sample shard count mismatch")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for layer in layers:
        key = f"block{layer}_patch_tokens"
        parts = []
        node_count = 0
        for path in files:
            with np.load(path, allow_pickle=False) as payload:
                values = np.asarray(payload[key], dtype=np.float32)
                if values.ndim != 3:
                    raise ValueError(f"{path}:{key} must be [N,S,D]")
                node_count += values.shape[0]
                parts.append(values.reshape(-1, values.shape[-1]))
        tokens = np.concatenate(parts, axis=0)
        if node_count != int(extraction["sampled_node_count"]):
            raise ValueError("sampled node count mismatch")
        for cluster_count in clusters:
            codebook = fit_local_pca_vlad(
                tokens,
                layer=layer,
                local_dimension=args.local_dimension,
                cluster_count=cluster_count,
                seed=args.seed + layer * 100 + cluster_count,
            )
            path = output / f"block{layer}-vlad-k{cluster_count}-localpca{args.local_dimension}.npz"
            with path.open("wb") as file:
                np.savez(file, **codebook.to_payload())
            entries.append(
                {
                    "layer": layer,
                    "cluster_count": cluster_count,
                    "local_dimension": args.local_dimension,
                    "input_token_count": int(tokens.shape[0]),
                    "input_dimension": int(tokens.shape[1]),
                    "path": path.name,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {
        "status": "pass",
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "extraction_summary_sha256": sha256_file(extraction_summary_path),
        "sample_fingerprint": extraction["sample_fingerprint"],
        "image_balanced_samples": True,
        "seed": args.seed,
        "entries": entries,
    }
    atomic_write_json(output / "codebook_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
