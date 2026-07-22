#!/usr/bin/env python3
"""Extract v1.2 mask-aware DINO descriptors and image-balanced patch samples."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MAR20 masked-patch DINO features")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--mar20-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-lock")
    parser.add_argument("--encoder", choices=("dinov2_vitb14", "mock"), default="dinov2_vitb14")
    parser.add_argument("--weights")
    parser.add_argument("--weight-sha256")
    parser.add_argument("--source-repo")
    parser.add_argument("--scope", choices=("target_only", "full_bridge"), default="full_bridge")
    parser.add_argument("--node-list")
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--rotations", default="0,90,180,270")
    parser.add_argument("--layers", default="9,10,11")
    parser.add_argument("--gem-powers", default="2,3,4")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--dilation-ratio", type=float, default=0.15)
    parser.add_argument("--maximum-patch-foreground-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-valid-patch-fraction", type=float, default=0.25)
    parser.add_argument("--patch-samples-per-node", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compute-dtype", choices=("float16", "float32", "bfloat16"), default="float16"
    )
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--seed", type=int, default=202625)
    parser.add_argument("--allow-mock", action="store_true")
    return parser.parse_args(argv)


def _csv(text: str, caster: Any) -> tuple[Any, ...]:
    values = tuple(caster(value.strip()) for value in text.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError("CSV option must be non-empty and unique")
    return values


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
        raise ValueError("DINOv2 assets incomplete")
    return {key: str(value) for key, value in values.items() if value}


def _rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = load_registry(args.registry)
    if args.scope == "target_only":
        rows = [row for row in rows if row["is_target"] == "1"]
    if args.node_list:
        allowed = {
            value.strip()
            for value in Path(args.node_list)
            .expanduser()
            .resolve()
            .read_text(encoding="utf-8")
            .splitlines()
            if value.strip()
        }
        known = {row["node_uid"] for row in rows}
        if not allowed or not allowed <= known:
            raise ValueError("node-list is empty or contains unknown nodes")
        rows = [row for row in rows if row["node_uid"] in allowed]
    rows.sort(key=lambda row: int(row["mar20_number"]))
    if args.max_nodes is not None:
        rows = rows[: args.max_nodes]
    if not rows:
        raise ValueError("selected node set is empty")
    return rows


def _sample_indices(valid: np.ndarray, count: int, key: str) -> np.ndarray:
    indices = np.flatnonzero(np.asarray(valid, dtype=bool))
    if indices.size < count:
        raise ValueError(f"{key}: only {indices.size} valid patches, need {count}")
    seed = int(stable_json_sha256({"key": key, "count": count})[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=count, replace=False))


def _sample_paths(root: Path, index: int) -> tuple[Path, Path]:
    return root / f"sample-shard-{index:05d}.npz", root / f"sample-shard-{index:05d}.json"


def _sample_valid(root: Path, index: int, fingerprint: str) -> bool:
    data, sidecar = _sample_paths(root, index)
    if not data.is_file() or not sidecar.is_file():
        return False
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("fingerprint") == fingerprint and value.get("sha256") == sha256_file(data)


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.splitlines()
        return {"commit": commit, "dirty": bool(dirty), "dirty_count": len(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "dirty_count": None}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.encoder == "mock" and not args.allow_mock:
        raise ValueError("mock requires --allow-mock")
    if args.batch_size <= 0 or args.shard_size <= 0 or args.batch_size > args.shard_size:
        raise ValueError("invalid batch/shard size")
    if args.patch_samples_per_node <= 0:
        raise ValueError("patch-samples-per-node must be positive")
    layers = _csv(args.layers, int)
    powers = _csv(args.gem_powers, float)
    rotations = _csv(args.rotations, int)
    rows = _rows(args)
    annotations = load_annotations(args.annotations)
    mar20_root = Path(args.mar20_root).expanduser().resolve()
    np.random.seed(args.seed)
    if args.encoder == "mock":
        encoder: Any = MockMaskedPlaceEncoder()
    else:
        assets = _assets(args)
        encoder = DinoV2MaskedPlaceEncoder(
            repo=assets["repo"],
            weights=assets["weights"],
            expected_weight_sha256=assets["weight_sha256"],
            layers=layers,
            gem_powers=powers,
            device=args.device,
            compute_dtype=args.compute_dtype,
        )
    output = Path(args.output_dir).expanduser().resolve()
    sample_root = output / "patch_samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "registry_sha256": sha256_file(args.registry),
        "annotations_sha256": sha256_file(args.annotations),
        "scope": args.scope,
        "node_uids_sha256": stable_json_sha256([row["node_uid"] for row in rows]),
        "node_count": len(rows),
        "rotations": list(rotations),
        "input_size": args.input_size,
        "patch_size": args.patch_size,
        "dilation_ratio": args.dilation_ratio,
        "maximum_patch_foreground_fraction": args.maximum_patch_foreground_fraction,
        "minimum_valid_patch_fraction": args.minimum_valid_patch_fraction,
        "patch_samples_per_node": args.patch_samples_per_node,
        "encoder": encoder.metadata(),
        "runtime": {
            "batch_size": args.batch_size,
            "shard_size": args.shard_size,
            "storage_dtype": args.storage_dtype,
        },
        "seed": args.seed,
    }
    sample_fingerprint = stable_json_sha256(
        {"config": config, "sample_schema": "mar20-balanced-patch-samples-v1"}
    )
    writer = PlaceFeatureCacheWriter(
        output / "cache",
        metadata=config,
        feature_names=encoder.feature_names,
        storage_dtype=args.storage_dtype,
    )
    pending: list[MaskedPatchInput] = []
    shard_index = 0
    total_rows = computed_rows = skipped_rows = 0
    low_valid_nodes: set[str] = set()
    started = time.perf_counter()

    def flush() -> None:
        nonlocal shard_index, total_rows, computed_rows, skipped_rows
        if not pending:
            return
        count = len(pending)
        cache_ok = writer.valid_existing_shard(shard_index, count)
        samples_ok = _sample_valid(sample_root, shard_index, sample_fingerprint)
        if cache_ok and samples_ok:
            skipped_rows += count
        else:
            feature_parts: dict[str, list[np.ndarray]] = {
                name: [] for name in encoder.feature_names
            }
            sample_parts: dict[str, list[np.ndarray]] = {
                name: [] for name in encoder.patch_token_names
            }
            sample_nodes: list[str] = []
            for offset in range(0, count, args.batch_size):
                batch = pending[offset : offset + args.batch_size]
                images = [item.image for item in batch]
                masks = np.stack([item.valid_patch_mask for item in batch])
                features, tokens = encoder.extract_masked_with_tokens(images, masks)
                for name in encoder.feature_names:
                    feature_parts[name].append(np.asarray(features[name], dtype=np.float32))
                for local, item in enumerate(batch):
                    if item.rotation != 0:
                        continue
                    sample_nodes.append(item.node_uid)
                    for name in encoder.patch_token_names:
                        index = _sample_indices(
                            item.valid_patch_mask,
                            args.patch_samples_per_node,
                            f"{item.node_uid}|{name}|{args.seed}",
                        )
                        sample_parts[name].append(
                            np.asarray(tokens[name][local, index], dtype=np.float32)
                        )
            if not cache_ok:
                row_payload = {
                    "node_uid": [item.node_uid for item in pending],
                    "view_type": ["masked_patch_original_input"] * count,
                    "rotation": [item.rotation for item in pending],
                    "item_index": [0] * count,
                    "input_sha256": [item.input_sha256 for item in pending],
                    "patch_mask_sha256": [item.patch_mask_sha256 for item in pending],
                    "valid_patch_fraction": [item.valid_patch_fraction for item in pending],
                    "valid_patch_count": [item.valid_patch_count for item in pending],
                    "patch_count": [item.patch_count for item in pending],
                    "foreground_fraction": [item.foreground_fraction for item in pending],
                }
                writer.write_shard(
                    shard_index,
                    rows=row_payload,
                    features={name: np.concatenate(parts) for name, parts in feature_parts.items()},
                )
            if not samples_ok:
                payload: dict[str, np.ndarray] = {"node_uid": np.asarray(sample_nodes)}
                for name, parts in sample_parts.items():
                    dimension = 3 if args.encoder == "mock" else 768
                    payload[name] = (
                        np.stack(parts).astype(np.float16)
                        if parts
                        else np.empty((0, args.patch_samples_per_node, dimension), dtype=np.float16)
                    )
                data_path, sidecar_path = _sample_paths(sample_root, shard_index)
                temporary = data_path.with_suffix(".npz.tmp")
                with temporary.open("wb") as file:
                    np.savez_compressed(file, **payload)
                os.replace(temporary, data_path)
                atomic_write_json(
                    sidecar_path,
                    {
                        "fingerprint": sample_fingerprint,
                        "shard_index": shard_index,
                        "node_count": len(sample_nodes),
                        "sha256": sha256_file(data_path),
                    },
                )
            computed_rows += count
        total_rows += count
        shard_index += 1
        pending.clear()

    for row in rows:
        uid = row["node_uid"]
        path = (mar20_root / row["original_relative_path"]).resolve()
        path.relative_to(mar20_root)
        with Image.open(path) as source:
            source.load()
            rendered = render_masked_patch_inputs(
                node_uid=uid,
                image=source,
                boxes=[box["xyxy"] for box in annotations[uid]["boxes"]],
                rotations=rotations,
                input_size=args.input_size,
                patch_size=args.patch_size,
                dilation_ratio=args.dilation_ratio,
                maximum_patch_foreground_fraction=args.maximum_patch_foreground_fraction,
            )
        if min(item.valid_patch_fraction for item in rendered) < args.minimum_valid_patch_fraction:
            low_valid_nodes.add(uid)
        pending.extend(rendered)
        if len(pending) >= args.shard_size:
            while len(pending) >= args.shard_size:
                tail = pending[args.shard_size :]
                del pending[args.shard_size :]
                flush()
                pending.extend(tail)
    flush()
    index = writer.finalize(expected_shards=shard_index, expected_rows=total_rows)
    audit = PlaceFeatureCache(output / "cache").audit()
    sample_sidecars = []
    sampled_nodes = 0
    for index_value in range(shard_index):
        _, sidecar = _sample_paths(sample_root, index_value)
        value = json.loads(sidecar.read_text(encoding="utf-8"))
        if value["fingerprint"] != sample_fingerprint:
            raise ValueError("patch sample fingerprint mismatch")
        sampled_nodes += int(value["node_count"])
        sample_sidecars.append(value)
    if sampled_nodes != len(rows):
        raise ValueError(f"patch sample node count expected={len(rows)}, actual={sampled_nodes}")
    elapsed = time.perf_counter() - started
    summary = {
        "status": "pass" if not low_valid_nodes else "fail_low_valid_patch_fraction",
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "config": config,
        "cache": audit,
        "index_sha256": sha256_file(output / "cache" / "index.json"),
        "sample_fingerprint": sample_fingerprint,
        "sampled_node_count": sampled_nodes,
        "sample_shard_count": len(sample_sidecars),
        "low_valid_patch_nodes": sorted(low_valid_nodes),
        "computed_rows": computed_rows,
        "skipped_rows": skipped_rows,
        "elapsed_seconds": elapsed,
        "rows_per_second": total_rows / elapsed if elapsed else None,
        "expected_shards": int(math.ceil(total_rows / args.shard_size)),
        "actual_shards": index["shard_count"],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "git": _git_state(),
        },
    }
    atomic_write_json(output / "extraction_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not low_valid_nodes else 2


if __name__ == "__main__":
    raise SystemExit(main())
