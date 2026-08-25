#!/usr/bin/env python3
"""Second-pass extraction of mask-aware MAR20 VLAD descriptors."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from rsdet.grouping.cache import PlaceFeatureCache, PlaceFeatureCacheWriter
from rsdet.grouping.contracts import (
    MASKED_PATCH_PROTOCOL_VERSION,
    atomic_write_json,
    sha256_file,
    stable_json_sha256,
)
from rsdet.grouping.descriptors import DinoV2MaskedPlaceEncoder, MockMaskedPlaceEncoder
from rsdet.grouping.masks import MaskedPatchInput, render_masked_patch_inputs
from rsdet.grouping.registry import load_annotations, load_registry
from rsdet.grouping.vlad import LocalPcaVladCodebook


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MAR20 mask-aware VLAD")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--mar20-root", required=True)
    parser.add_argument("--codebook-manifest", required=True)
    parser.add_argument("--codebook-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-lock")
    parser.add_argument("--encoder", choices=("dinov2_vitb14", "mock"), default="dinov2_vitb14")
    parser.add_argument("--weights")
    parser.add_argument("--weight-sha256")
    parser.add_argument("--source-repo")
    parser.add_argument("--scope", choices=("target_only", "full_bridge"), default="full_bridge")
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--rotations", default="0,90,180,270")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--dilation-ratio", type=float, default=0.15)
    parser.add_argument("--maximum-patch-foreground-fraction", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compute-dtype", default="float16", choices=("float16", "float32", "bfloat16")
    )
    parser.add_argument("--storage-dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--seed", type=int, default=202625)
    parser.add_argument("--allow-mock", action="store_true")
    return parser.parse_args(argv)


def _assets(args: argparse.Namespace) -> dict[str, str]:
    values = {
        "weights": args.weights,
        "weight_sha256": args.weight_sha256,
        "repo": args.source_repo,
    }
    if args.asset_lock:
        lock = json.loads(Path(args.asset_lock).expanduser().resolve().read_text(encoding="utf-8"))
        values["weights"] = values["weights"] or lock.get("files", {}).get("dinov2_vitb14", {}).get(
            "path"
        )
        values["weight_sha256"] = values["weight_sha256"] or lock.get("files", {}).get(
            "dinov2_vitb14", {}
        ).get("sha256")
        values["repo"] = values["repo"] or lock.get("repositories", {}).get("dinov2", {}).get(
            "path"
        )
    if args.encoder != "mock" and any(not value for value in values.values()):
        raise ValueError("DINO assets incomplete")
    return {key: str(value) for key, value in values.items() if value}


def _load_codebooks(
    manifest_path: Path, root: Path
) -> tuple[dict[int, list[tuple[str, LocalPcaVladCodebook]]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "pass"
        or manifest.get("protocol_version") != MASKED_PATCH_PROTOCOL_VERSION
    ):
        raise ValueError("VLAD codebook manifest not admitted")
    result: dict[int, list[tuple[str, LocalPcaVladCodebook]]] = {}
    for entry in manifest["entries"]:
        path = (root / entry["path"]).resolve()
        path.relative_to(root)
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"codebook SHA mismatch: {path}")
        with np.load(path, allow_pickle=False) as payload:
            codebook = LocalPcaVladCodebook.from_payload(payload)
        name = f"block{codebook.layer}_masked_vlad_k{codebook.cluster_count}_localpca{codebook.local_dimension}"
        result.setdefault(codebook.layer, []).append((name, codebook))
    if not result:
        raise ValueError("no codebooks")
    return result, manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.encoder == "mock" and not args.allow_mock:
        raise ValueError("mock requires --allow-mock")
    rotations = tuple(int(value) for value in args.rotations.split(","))
    manifest_path = Path(args.codebook_manifest).expanduser().resolve()
    codebook_root = Path(args.codebook_dir).expanduser().resolve()
    codebooks, manifest = _load_codebooks(manifest_path, codebook_root)
    layers = tuple(sorted(codebooks))
    if args.encoder == "mock":
        encoder: Any = MockMaskedPlaceEncoder()
    else:
        assets = _assets(args)
        encoder = DinoV2MaskedPlaceEncoder(
            repo=assets["repo"],
            weights=assets["weights"],
            expected_weight_sha256=assets["weight_sha256"],
            layers=layers,
            gem_powers=(3.0,),
            device=args.device,
            compute_dtype=args.compute_dtype,
        )
    feature_names = tuple(name for layer in layers for name, _ in codebooks[layer])
    rows = load_registry(args.registry)
    if args.scope == "target_only":
        rows = [row for row in rows if row["is_target"] == "1"]
    rows.sort(key=lambda row: int(row["mar20_number"]))
    if args.max_nodes is not None:
        rows = rows[: args.max_nodes]
    annotations = load_annotations(args.annotations)
    mar20_root = Path(args.mar20_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    config = {
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "registry_sha256": sha256_file(args.registry),
        "annotations_sha256": sha256_file(args.annotations),
        "scope": args.scope,
        "node_uids_sha256": stable_json_sha256([row["node_uid"] for row in rows]),
        "rotations": list(rotations),
        "input_size": args.input_size,
        "patch_size": args.patch_size,
        "dilation_ratio": args.dilation_ratio,
        "maximum_patch_foreground_fraction": args.maximum_patch_foreground_fraction,
        "codebook_manifest_sha256": sha256_file(manifest_path),
        "codebook_sample_fingerprint": manifest["sample_fingerprint"],
        "encoder": encoder.metadata(),
        "runtime": {
            "batch_size": args.batch_size,
            "shard_size": args.shard_size,
            "storage_dtype": args.storage_dtype,
        },
    }
    writer = PlaceFeatureCacheWriter(
        output / "cache",
        metadata=config,
        feature_names=feature_names,
        storage_dtype=args.storage_dtype,
    )
    pending: list[MaskedPatchInput] = []
    shard_index = total = computed = skipped = 0
    started = time.perf_counter()

    def flush() -> None:
        nonlocal shard_index, total, computed, skipped
        if not pending:
            return
        count = len(pending)
        if writer.valid_existing_shard(shard_index, count):
            skipped += count
        else:
            parts: dict[str, list[np.ndarray]] = {name: [] for name in feature_names}
            for offset in range(0, count, args.batch_size):
                batch = pending[offset : offset + args.batch_size]
                images = [item.image for item in batch]
                tokens = encoder.extract_patch_tokens(images)
                for local, item in enumerate(batch):
                    valid = item.valid_patch_mask
                    for layer in layers:
                        token_name = (
                            "mock_patch_tokens"
                            if args.encoder == "mock"
                            else f"block{layer}_patch_tokens"
                        )
                        selected = np.asarray(tokens[token_name][local, valid], dtype=np.float32)
                        for name, codebook in codebooks[layer]:
                            parts[name].append(codebook.encode(selected))
            writer.write_shard(
                shard_index,
                rows={
                    "node_uid": [item.node_uid for item in pending],
                    "view_type": ["masked_patch_vlad"] * count,
                    "rotation": [item.rotation for item in pending],
                    "item_index": [0] * count,
                    "input_sha256": [item.input_sha256 for item in pending],
                    "patch_mask_sha256": [item.patch_mask_sha256 for item in pending],
                    "valid_patch_fraction": [item.valid_patch_fraction for item in pending],
                },
                features={
                    name: np.stack(values).astype(np.float32) for name, values in parts.items()
                },
            )
            computed += count
        total += count
        shard_index += 1
        pending.clear()

    for row in rows:
        uid = row["node_uid"]
        path = (mar20_root / row["original_relative_path"]).resolve()
        path.relative_to(mar20_root)
        with Image.open(path) as source:
            source.load()
            items = render_masked_patch_inputs(
                node_uid=uid,
                image=source,
                boxes=[box["xyxy"] for box in annotations[uid]["boxes"]],
                rotations=rotations,
                input_size=args.input_size,
                patch_size=args.patch_size,
                dilation_ratio=args.dilation_ratio,
                maximum_patch_foreground_fraction=args.maximum_patch_foreground_fraction,
            )
        pending.extend(items)
        if len(pending) >= args.shard_size:
            while len(pending) >= args.shard_size:
                tail = pending[args.shard_size :]
                del pending[args.shard_size :]
                flush()
                pending.extend(tail)
    flush()
    index = writer.finalize(expected_shards=shard_index, expected_rows=total)
    audit = PlaceFeatureCache(output / "cache").audit()
    elapsed = time.perf_counter() - started
    summary = {
        "status": "pass",
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "config": config,
        "cache": audit,
        "index_sha256": sha256_file(output / "cache" / "index.json"),
        "computed_rows": computed,
        "skipped_rows": skipped,
        "elapsed_seconds": elapsed,
        "rows_per_second": total / elapsed if elapsed else None,
        "expected_shards": int(math.ceil(total / args.shard_size)),
        "actual_shards": index["shard_count"],
    }
    atomic_write_json(output / "extraction_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
