#!/usr/bin/env python3
"""Compute resumable SIFT and coarse DINO patch evidence for TASK-01 pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps

from rsdet.grouping.cache import PlaceFeatureCache
from rsdet.grouping.contracts import (
    MASKED_PATCH_PROTOCOL_VERSION,
    atomic_write_json,
    sha256_file,
    stable_json_sha256,
)
from rsdet.grouping.geometry import (
    SiftFeatures,
    extract_sift_features,
    patch_pair_evidence,
    sift_pair_evidence,
)
from rsdet.grouping.masks import build_protocol_foreground_mask
from rsdet.grouping.registry import load_annotations, load_registry


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify MAR20 TASK-01 local geometry")
    parser.add_argument("--geometry-queue", required=True)
    parser.add_argument("--geometry-queue-summary", required=True)
    parser.add_argument("--patch-cache", required=True)
    parser.add_argument("--patch-summary", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--mar20-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sift-max-dimension", type=int, default=1024)
    parser.add_argument("--sift-features", type=int, default=2500)
    parser.add_argument("--sift-ratio", type=float, default=0.75)
    parser.add_argument("--ransac-repeat-count", type=int, default=20)
    parser.add_argument("--pair-shard-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=202625)
    return parser.parse_args(argv)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _atomic_npz(path: Path, **payload: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **payload)
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.pair_shard_size <= 0 or args.sift_max_dimension <= 0 or args.sift_features <= 0:
        raise ValueError("invalid geometry runtime arguments")
    queue_path = Path(args.geometry_queue).expanduser().resolve()
    queue_summary_path = Path(args.geometry_queue_summary).expanduser().resolve()
    patch_summary_path = Path(args.patch_summary).expanduser().resolve()
    queue_summary = json.loads(queue_summary_path.read_text(encoding="utf-8"))
    patch_summary = json.loads(patch_summary_path.read_text(encoding="utf-8"))
    if queue_summary.get("artifact_sha256") != sha256_file(queue_path):
        raise ValueError("geometry queue SHA mismatch")
    patch_cache = PlaceFeatureCache(args.patch_cache)
    if (
        patch_summary.get("status") != "pass"
        or patch_summary.get("cache_fingerprint") != patch_cache.index["fingerprint"]
    ):
        raise ValueError("patch overlap cache is not admitted")
    queue = _read(queue_path)
    if len(queue) != int(queue_summary["queue_pair_count"]):
        raise ValueError("geometry queue row count mismatch")
    registry_path = Path(args.registry).expanduser().resolve()
    annotations_path = Path(args.annotations).expanduser().resolve()
    registry = {row["node_uid"]: row for row in load_registry(registry_path)}
    annotations = load_annotations(annotations_path)
    root = Path(args.mar20_root).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    shard_dir = output / "pair_shards"
    sift_dir = output / "sift_cache"
    shard_dir.mkdir(parents=True, exist_ok=True)
    sift_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "task": "mar20-task01-geometry-v1",
        "geometry_queue_sha256": sha256_file(queue_path),
        "patch_cache_fingerprint": patch_cache.index["fingerprint"],
        "registry_sha256": sha256_file(registry_path),
        "annotations_sha256": sha256_file(annotations_path),
        "sift_max_dimension": args.sift_max_dimension,
        "sift_features": args.sift_features,
        "sift_ratio": args.sift_ratio,
        "ransac_fraction_diagonal": 0.005,
        "ransac_repeat_count": args.ransac_repeat_count,
        "seed": args.seed,
    }
    fingerprint = stable_json_sha256(config)
    meta_path = sift_dir / "cache_meta.json"
    if meta_path.exists():
        if json.loads(meta_path.read_text(encoding="utf-8")).get("fingerprint") != fingerprint:
            raise ValueError("existing SIFT cache contract differs")
    else:
        atomic_write_json(meta_path, {"fingerprint": fingerprint, "config": config})

    def image_and_mask(uid: str) -> tuple[np.ndarray, np.ndarray]:
        path = (root / registry[uid]["original_relative_path"]).resolve()
        path.relative_to(root)
        with Image.open(path) as image:
            image.load()
            rgb = ImageOps.exif_transpose(image).convert("RGB")
        scale = min(1.0, args.sift_max_dimension / max(rgb.size))
        size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
        gray = np.asarray(rgb.convert("L").resize(size, Image.Resampling.LANCZOS), dtype=np.uint8)
        foreground = build_protocol_foreground_mask(
            rgb.size,
            [box["xyxy"] for box in annotations[uid]["boxes"]],
            dilation_ratio=0.15,
        ).resize(size, Image.Resampling.NEAREST)
        background = 255 - np.asarray(foreground, dtype=np.uint8)
        return gray, background

    sift_memory: dict[str, SiftFeatures] = {}

    def sift(uid: str) -> SiftFeatures:
        if uid in sift_memory:
            return sift_memory[uid]
        path = sift_dir / f"{uid.replace(':', '-')}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as payload:
                if str(payload["fingerprint"].item()) == fingerprint:
                    value = SiftFeatures(
                        points=np.asarray(payload["points"], dtype=np.float32),
                        descriptors=np.asarray(payload["descriptors"], dtype=np.float32),
                        image_size=tuple(int(item) for item in payload["image_size"]),
                    )
                    sift_memory[uid] = value
                    return value
        gray, background = image_and_mask(uid)
        value = extract_sift_features(gray, background, nfeatures=args.sift_features)
        _atomic_npz(
            path,
            fingerprint=np.asarray(fingerprint),
            points=value.points,
            descriptors=value.descriptors,
            image_size=np.asarray(value.image_size, dtype=np.int32),
        )
        sift_memory[uid] = value
        return value

    patch_payload = patch_cache.load_all()
    patch_nodes = patch_payload["row__node_uid"].astype(str)
    if len(patch_nodes) != len(set(patch_nodes)):
        raise ValueError("patch cache has duplicate node rows")
    patch_index = {uid: index for index, uid in enumerate(patch_nodes)}
    token_flat = patch_payload["feature__block11_coarse_tokens_flat"]
    patch_valid = patch_payload["feature__coarse_valid"] > 0.5
    expected_dimension = 19 * 19 * 128
    if token_flat.shape[1] != expected_dimension or patch_valid.shape[1] != 19 * 19:
        raise ValueError("patch cache dimensions differ from frozen 19x19/128 contract")
    if any(row[side] not in patch_index for row in queue for side in ("node_u", "node_v")):
        raise ValueError("patch cache does not cover geometry queue")
    input_fields = list(queue[0])
    computed_pairs = skipped_pairs = 0
    shard_count = math.ceil(len(queue) / args.pair_shard_size)
    shard_records = []
    for shard_index in range(shard_count):
        subset = queue[
            shard_index * args.pair_shard_size : (shard_index + 1) * args.pair_shard_size
        ]
        csv_path = shard_dir / f"shard-{shard_index:05d}.csv"
        sidecar_path = shard_dir / f"shard-{shard_index:05d}.json"
        if csv_path.exists() and sidecar_path.exists():
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if (
                sidecar.get("fingerprint") == fingerprint
                and sidecar.get("row_count") == len(subset)
                and sidecar.get("sha256") == sha256_file(csv_path)
            ):
                skipped_pairs += len(subset)
                shard_records.append(sidecar)
                continue
        results = []
        for row in subset:
            uid_u, uid_v = row["node_u"], row["node_v"]
            pair_seed = args.seed + int(stable_json_sha256({"pair": row["pair_uid"]})[:8], 16)
            sift_metrics = sift_pair_evidence(
                sift(uid_u),
                sift(uid_v),
                ratio=args.sift_ratio,
                ransac_fraction_diagonal=0.005,
                repeat_count=args.ransac_repeat_count,
                seed=pair_seed,
            )
            index_u, index_v = patch_index[uid_u], patch_index[uid_v]
            patch_metrics = patch_pair_evidence(
                np.asarray(token_flat[index_u], dtype=np.float32).reshape(19 * 19, 128),
                patch_valid[index_u],
                np.asarray(token_flat[index_v], dtype=np.float32).reshape(19 * 19, 128),
                patch_valid[index_v],
                grid=19,
            )
            results.append({**row, **sift_metrics, **patch_metrics})
        output_fields = list(results[0])
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=output_fields)
            writer.writeheader()
            writer.writerows(results)
        sidecar = {
            "fingerprint": fingerprint,
            "shard_index": shard_index,
            "row_count": len(results),
            "sha256": sha256_file(csv_path),
        }
        atomic_write_json(sidecar_path, sidecar)
        shard_records.append(sidecar)
        computed_pairs += len(results)

    evidence_path = output / "pair_evidence.csv"
    output_fields: list[str] | None = None
    total = nonfinite = 0
    with evidence_path.open("w", encoding="utf-8", newline="") as destination:
        writer = None
        for shard_index, sidecar in enumerate(shard_records):
            csv_path = shard_dir / f"shard-{shard_index:05d}.csv"
            if sidecar["sha256"] != sha256_file(csv_path):
                raise ValueError(f"geometry shard {shard_index} SHA mismatch")
            rows = _read(csv_path)
            if output_fields is None:
                output_fields = list(rows[0])
                writer = csv.DictWriter(destination, fieldnames=output_fields)
                writer.writeheader()
            if list(rows[0]) != output_fields:
                raise ValueError("geometry shard field mismatch")
            writer.writerows(rows)
            total += len(rows)
            for row in rows:
                for key, value in row.items():
                    if not value or key.endswith("_matrix") or key in input_fields:
                        continue
                    try:
                        number = float(value)
                    except ValueError:
                        continue
                    nonfinite += not math.isfinite(number)
    if total != len(queue) or nonfinite:
        raise ValueError(f"geometry merge failed: rows={total}/{len(queue)}, nonfinite={nonfinite}")
    summary = {
        "status": "pass",
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "fingerprint": fingerprint,
        "pair_count": total,
        "computed_pairs": computed_pairs,
        "skipped_pairs": skipped_pairs,
        "pair_shard_count": shard_count,
        "sift_cached_node_count": len(list(sift_dir.glob("mar20-*.npz"))),
        "nonfinite_count": nonfinite,
        "geometry_queue_sha256": sha256_file(queue_path),
        "patch_cache_fingerprint": patch_cache.index["fingerprint"],
        "pair_evidence_sha256": sha256_file(evidence_path),
        "formal_grouping_admission": False,
    }
    atomic_write_json(output / "geometry_verification_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
