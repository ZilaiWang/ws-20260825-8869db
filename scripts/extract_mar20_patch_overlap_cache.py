#!/usr/bin/env python3
"""Extract one-rotation, 19x19 DINO patch maps for the TASK-01 geometry queue."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from rsdet.grouping.cache import PlaceFeatureCache, PlaceFeatureCacheWriter
from rsdet.grouping.contracts import MASKED_PATCH_PROTOCOL_VERSION, atomic_write_json, sha256_file
from rsdet.grouping.descriptors import DinoV2MaskedPlaceEncoder
from rsdet.grouping.geometry import pool_masked_patch_tokens
from rsdet.grouping.masks import render_masked_patch_inputs
from rsdet.grouping.registry import load_annotations, load_registry
from rsdet.grouping.vlad import LocalPcaVladCodebook


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MAR20 TASK-01 patch-overlap cache")
    parser.add_argument("--geometry-queue", required=True)
    parser.add_argument("--geometry-queue-summary", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--mar20-root", required=True)
    parser.add_argument("--asset-lock", required=True)
    parser.add_argument("--codebook-manifest", required=True)
    parser.add_argument("--codebook-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-dtype", default="float16")
    return parser.parse_args(argv)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _asset_values(path: Path) -> dict[str, str]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    values = {
        "weights": lock.get("files", {}).get("dinov2_vitb14", {}).get("path"),
        "weight_sha256": lock.get("files", {}).get("dinov2_vitb14", {}).get("sha256"),
        "repo": lock.get("repositories", {}).get("dinov2", {}).get("path"),
    }
    if any(not value for value in values.values()):
        raise ValueError("DINOv2-B assets are incomplete")
    return {key: str(value) for key, value in values.items()}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0 or args.shard_size <= 0 or args.batch_size > args.shard_size:
        raise ValueError("invalid batch/shard size")
    queue_path = Path(args.geometry_queue).expanduser().resolve()
    queue_summary_path = Path(args.geometry_queue_summary).expanduser().resolve()
    queue_summary = json.loads(queue_summary_path.read_text(encoding="utf-8"))
    if queue_summary.get("status") != "pass" or queue_summary.get(
        "artifact_sha256"
    ) != sha256_file(queue_path):
        raise ValueError("geometry queue artifacts are not admitted")
    registry_path = Path(args.registry).expanduser().resolve()
    annotations_path = Path(args.annotations).expanduser().resolve()
    registry = {row["node_uid"]: row for row in load_registry(registry_path)}
    annotations = load_annotations(annotations_path)
    queue = _read(queue_path)
    node_uids = sorted(
        {row[side] for row in queue for side in ("node_u", "node_v")},
        key=lambda uid: int(uid.split(":")[1]),
    )
    if not node_uids or any(uid not in registry for uid in node_uids):
        raise ValueError("geometry queue contains no nodes or unknown nodes")
    assets = _asset_values(Path(args.asset_lock).expanduser().resolve())
    codebook_manifest_path = Path(args.codebook_manifest).expanduser().resolve()
    codebook_manifest = json.loads(codebook_manifest_path.read_text(encoding="utf-8"))
    entries = [
        entry
        for entry in codebook_manifest.get("entries", [])
        if int(entry.get("layer", -1)) == 11 and int(entry.get("cluster_count", -1)) == 32
    ]
    if codebook_manifest.get("status") != "pass" or len(entries) != 1:
        raise ValueError("exactly one admitted block11/K32 local-PCA codebook is required")
    codebook_path = (Path(args.codebook_dir).expanduser().resolve() / entries[0]["path"]).resolve()
    if sha256_file(codebook_path) != entries[0]["sha256"]:
        raise ValueError("block11/K32 codebook SHA mismatch")
    with np.load(codebook_path, allow_pickle=False) as payload:
        codebook = LocalPcaVladCodebook.from_payload(payload)
    if codebook.local_dimension != 128 or codebook.local_input_dimension != 768:
        raise ValueError("patch-overlap codebook must project 768D tokens to frozen 128D")
    encoder = DinoV2MaskedPlaceEncoder(
        repo=assets["repo"],
        weights=assets["weights"],
        expected_weight_sha256=assets["weight_sha256"],
        layers=(11,),
        gem_powers=(3.0,),
        device=args.device,
        compute_dtype=args.compute_dtype,
    )
    metadata = {
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "task": "mar20-task01-coarse-patch-overlap",
        "geometry_queue_sha256": sha256_file(queue_path),
        "registry_sha256": sha256_file(registry_path),
        "annotations_sha256": sha256_file(annotations_path),
        "node_count": len(node_uids),
        "node_uids": node_uids,
        "rotation": 0,
        "input_grid": 37,
        "output_grid": 19,
        "pooling": "mask_aware_adaptive_average",
        "minimum_coarse_valid_fraction": 0.5,
        "local_projection": "block11_vlad_k32_localpca128",
        "codebook_manifest_sha256": sha256_file(codebook_manifest_path),
        "codebook_sha256": sha256_file(codebook_path),
        "encoder": encoder.metadata(),
    }
    output = Path(args.output_dir).expanduser().resolve()
    cache_dir = output / "cache"
    writer = PlaceFeatureCacheWriter(
        cache_dir,
        metadata=metadata,
        feature_names=("block11_coarse_tokens_flat", "coarse_valid"),
        storage_dtype="float16",
    )
    root = Path(args.mar20_root).expanduser().resolve()
    shard_count = math.ceil(len(node_uids) / args.shard_size)
    computed = skipped = 0
    for shard_index in range(shard_count):
        subset = node_uids[shard_index * args.shard_size : (shard_index + 1) * args.shard_size]
        if writer.valid_existing_shard(shard_index, len(subset)):
            skipped += len(subset)
            continue
        pooled_parts: list[np.ndarray] = []
        valid_parts: list[np.ndarray] = []
        input_sha: list[str] = []
        for offset in range(0, len(subset), args.batch_size):
            batch_uids = subset[offset : offset + args.batch_size]
            rendered = []
            for uid in batch_uids:
                path = (root / registry[uid]["original_relative_path"]).resolve()
                path.relative_to(root)
                with Image.open(path) as image:
                    image.load()
                    item = render_masked_patch_inputs(
                        node_uid=uid,
                        image=image,
                        boxes=[box["xyxy"] for box in annotations[uid]["boxes"]],
                        rotations=(0,),
                        input_size=518,
                        patch_size=14,
                        dilation_ratio=0.15,
                        maximum_patch_foreground_fraction=0.20,
                    )[0]
                rendered.append(item)
            token_payload = encoder.extract_patch_tokens([item.image for item in rendered])
            tokens = np.asarray(token_payload["block11_patch_tokens"], dtype=np.float32)
            for item, value in zip(rendered, tokens, strict=True):
                value = codebook.project_local(value)
                pooled, valid = pool_masked_patch_tokens(
                    value,
                    item.valid_patch_mask,
                    input_grid=37,
                    output_grid=19,
                    minimum_valid_fraction=0.5,
                )
                if not valid.any() or not np.isfinite(pooled).all():
                    raise ValueError(f"{item.node_uid}: invalid pooled patch tokens")
                pooled_parts.append(pooled.reshape(-1))
                valid_parts.append(valid.astype(np.float32))
                input_sha.append(item.input_sha256)
        writer.write_shard(
            shard_index,
            rows={
                "node_uid": subset,
                "view_type": ["masked_patch_coarse19"] * len(subset),
                "rotation": [0] * len(subset),
                "item_index": [0] * len(subset),
                "input_sha256": input_sha,
            },
            features={
                "block11_coarse_tokens_flat": np.stack(pooled_parts),
                "coarse_valid": np.stack(valid_parts),
            },
        )
        computed += len(subset)
    index = writer.finalize(expected_shards=shard_count, expected_rows=len(node_uids))
    audit = PlaceFeatureCache(cache_dir).audit()
    summary = {
        "status": "pass",
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "node_count": len(node_uids),
        "computed_nodes": computed,
        "skipped_nodes": skipped,
        "cache": audit,
        "cache_index_sha256": sha256_file(cache_dir / "index.json"),
        "cache_fingerprint": index["fingerprint"],
        "geometry_queue_sha256": sha256_file(queue_path),
        "asset_lock_sha256": sha256_file(args.asset_lock),
        "codebook_manifest_sha256": sha256_file(codebook_manifest_path),
        "codebook_sha256": sha256_file(codebook_path),
        "formal_grouping_admission": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "patch_overlap_extraction_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
