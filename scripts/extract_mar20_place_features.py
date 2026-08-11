#!/usr/bin/env python3
"""MG01：按冻结背景视图提取 DINOv2 多层地点描述子。"""

from __future__ import annotations

import argparse
import json
import math
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
    PROTOCOL_VERSION,
    atomic_write_json,
    sha256_file,
    stable_json_sha256,
)
from rsdet.grouping.descriptors import DinoV2PlaceEncoder, MockPlaceEncoder
from rsdet.grouping.masks import render_place_inputs
from rsdet.grouping.registry import load_annotations, load_registry


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取 MAR20 DINOv2 地点描述子")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--mar20-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asset-lock")
    parser.add_argument("--encoder", choices=("dinov2_vitb14", "mock"), default="dinov2_vitb14")
    parser.add_argument("--weights")
    parser.add_argument("--weight-sha256")
    parser.add_argument("--source-repo")
    parser.add_argument("--scope", choices=("target_only", "full_bridge"), default="target_only")
    parser.add_argument("--node-list", help="每行一个 node_uid 的冻结子集")
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--view-types", default="masked_inpaint")
    parser.add_argument("--rotations", default="0,90,180,270")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--dilation-ratio", type=float, default=0.15)
    parser.add_argument("--fill-method", choices=("telea", "blur", "local_mean"), default="telea")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--tile-stride", type=int, default=112)
    parser.add_argument("--tile-valid-fraction", type=float, default=0.95)
    parser.add_argument("--max-tiles", type=int, default=8)
    parser.add_argument("--layers", default="9,10,11")
    parser.add_argument("--gem-powers", default="2,3,4")
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


def _csv_values(text: str, caster: Any) -> tuple[Any, ...]:
    values = tuple(caster(value.strip()) for value in text.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"逗号列表必须非空且唯一: {text!r}")
    return values


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


def _asset_options(args: argparse.Namespace) -> dict[str, str]:
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
    missing = [key for key, value in values.items() if not value]
    if args.encoder != "mock" and missing:
        raise ValueError(f"DINOv2 资产缺失: {missing}")
    return {key: str(value) for key, value in values.items() if value}


def _select_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = load_registry(args.registry)
    if args.scope == "target_only":
        rows = [row for row in rows if row["is_target"] == "1"]
    if args.node_list:
        allow = {
            value.strip()
            for value in Path(args.node_list)
            .expanduser()
            .resolve()
            .read_text(encoding="utf-8")
            .splitlines()
            if value.strip() and not value.lstrip().startswith("#")
        }
        if not allow:
            raise ValueError("node-list 为空")
        known = {row["node_uid"] for row in rows}
        missing = allow - known
        if missing:
            raise ValueError(f"node-list 有未知或不在 scope 的 UID: {sorted(missing)[:10]}")
        rows = [row for row in rows if row["node_uid"] in allow]
    rows.sort(key=lambda row: int(row["mar20_number"]))
    if args.max_nodes is not None:
        if args.max_nodes <= 0:
            raise ValueError("max-nodes 必须大于 0")
        rows = rows[: args.max_nodes]
    if not rows:
        raise ValueError("节点筛选后为空")
    return rows


def _configure_seed(seed: int, device: str, encoder: str) -> None:
    np.random.seed(seed)
    if encoder == "mock":
        return
    import torch

    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.encoder == "mock" and not args.allow_mock:
        raise ValueError("mock encoder 只允许显式 --allow-mock 的 smoke")
    if args.batch_size <= 0 or args.shard_size <= 0:
        raise ValueError("batch-size/shard-size 必须大于 0")
    if args.batch_size > args.shard_size:
        raise ValueError("batch-size 不得大于 shard-size")
    layers = _csv_values(args.layers, int)
    powers = _csv_values(args.gem_powers, float)
    rotations = _csv_values(args.rotations, int)
    view_types = _csv_values(args.view_types, str)
    rows = _select_rows(args)
    annotations = load_annotations(args.annotations)
    mar20_root = Path(args.mar20_root).expanduser().resolve()
    if not mar20_root.is_dir():
        raise FileNotFoundError(mar20_root)
    _configure_seed(args.seed, args.device, args.encoder)
    assets = _asset_options(args)
    if args.encoder == "mock":
        encoder: Any = MockPlaceEncoder()
    else:
        encoder = DinoV2PlaceEncoder(
            repo=assets["repo"],
            weights=assets["weights"],
            expected_weight_sha256=assets["weight_sha256"],
            layers=layers,
            gem_powers=powers,
            device=args.device,
            compute_dtype=args.compute_dtype,
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "registry_sha256": sha256_file(args.registry),
        "annotations_sha256": sha256_file(args.annotations),
        "scope": args.scope,
        "node_uids_sha256": stable_json_sha256([row["node_uid"] for row in rows]),
        "node_count": len(rows),
        "view_types": list(view_types),
        "rotations": list(rotations),
        "input_size": args.input_size,
        "dilation_ratio": args.dilation_ratio,
        "fill_method": args.fill_method,
        "tile_size": args.tile_size,
        "tile_stride": args.tile_stride,
        "tile_valid_fraction": args.tile_valid_fraction,
        "max_tiles": args.max_tiles,
        "encoder": encoder.metadata(),
        "runtime_contract": {
            "batch_size": args.batch_size,
            "shard_size": args.shard_size,
            "device": args.device,
            "compute_dtype": args.compute_dtype,
            "storage_dtype": args.storage_dtype,
        },
        "seed": args.seed,
    }
    writer = PlaceFeatureCacheWriter(
        output_dir / "cache",
        metadata=config,
        feature_names=encoder.feature_names,
        storage_dtype=args.storage_dtype,
    )
    pending_images: list[Image.Image] = []
    pending_rows: dict[str, list[Any]] = {
        "node_uid": [],
        "view_type": [],
        "rotation": [],
        "item_index": [],
        "input_sha256": [],
        "source_box_json": [],
        "valid_background_fraction": [],
    }
    shard_index = 0
    total_rows = 0
    skipped_rows = 0
    computed_rows = 0
    nodes_without_background_tiles: list[str] = []
    started = time.perf_counter()

    def flush() -> None:
        nonlocal shard_index, total_rows, skipped_rows, computed_rows
        if not pending_images:
            return
        row_count = len(pending_images)
        if writer.valid_existing_shard(shard_index, row_count):
            skipped_rows += row_count
        else:
            chunks: dict[str, list[np.ndarray]] = {name: [] for name in encoder.feature_names}
            for offset in range(0, row_count, args.batch_size):
                batch = pending_images[offset : offset + args.batch_size]
                extracted = encoder.extract(batch)
                for name in encoder.feature_names:
                    chunks[name].append(np.asarray(extracted[name], dtype=np.float32))
            features = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
            writer.write_shard(shard_index, rows=pending_rows, features=features)
            computed_rows += row_count
        total_rows += row_count
        shard_index += 1
        pending_images.clear()
        for values in pending_rows.values():
            values.clear()

    for row in rows:
        node_uid = row["node_uid"]
        annotation = annotations[node_uid]
        image_path = (mar20_root / row["original_relative_path"]).resolve()
        try:
            image_path.relative_to(mar20_root)
        except ValueError as error:
            raise ValueError(f"原图路径逃逸 MAR20_ROOT: {image_path}") from error
        with Image.open(image_path) as source:
            source.load()
            rendered = render_place_inputs(
                node_uid=node_uid,
                image=source,
                boxes=[item["xyxy"] for item in annotation["boxes"]],
                view_types=view_types,
                rotations=rotations,
                input_size=args.input_size,
                dilation_ratio=args.dilation_ratio,
                fill_method=args.fill_method,
                tile_size=args.tile_size,
                tile_stride=args.tile_stride,
                tile_valid_fraction=args.tile_valid_fraction,
                max_tiles=args.max_tiles,
            )
        if "background_tiles" in view_types and not any(
            item.view_type == "background_tiles" for item in rendered
        ):
            nodes_without_background_tiles.append(node_uid)
        for item in rendered:
            pending_images.append(item.image)
            pending_rows["node_uid"].append(item.node_uid)
            pending_rows["view_type"].append(item.view_type)
            pending_rows["rotation"].append(item.rotation)
            pending_rows["item_index"].append(item.item_index)
            pending_rows["input_sha256"].append(item.input_sha256)
            pending_rows["source_box_json"].append(
                json.dumps(item.source_box, separators=(",", ":")) if item.source_box else ""
            )
            pending_rows["valid_background_fraction"].append(item.valid_background_fraction)
            if len(pending_images) == args.shard_size:
                flush()
    flush()
    index = writer.finalize(expected_shards=shard_index, expected_rows=total_rows)
    audit = PlaceFeatureCache(output_dir / "cache").audit()
    elapsed = time.perf_counter() - started
    summary = {
        "status": "pass",
        "protocol_version": PROTOCOL_VERSION,
        "config": config,
        "cache": audit,
        "index_sha256": sha256_file(output_dir / "cache" / "index.json"),
        "computed_rows": computed_rows,
        "skipped_rows": skipped_rows,
        "nodes_without_background_tiles": nodes_without_background_tiles,
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
    atomic_write_json(output_dir / "extraction_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
