#!/usr/bin/env python3
"""P04 教师特征分片抽取。"""

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

from rsdet.data.crop_classification import CropClassificationDataset
from rsdet.features.p04_cache import FeatureCacheWriter, sha256_file, stable_json_sha256
from rsdet.features.p04_inputs import (
    D4_VIEW_IDS,
    apply_d4_view,
    canonical_image_sha256,
    filter_records,
    load_all_crop_records,
    load_annotation_allowlist,
    preprocessing_fingerprint,
)
from rsdet.features.p04_teachers import build_teacher


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P04 无标签教师特征缓存")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--asset-lock",
        help="P04-TASK-00 生成的 ASSET_LOCK.json；正式教师建议使用",
    )
    parser.add_argument(
        "--teacher",
        required=True,
        choices=(
            "mock",
            "convnext_tiny",
            "dinov2_vits14",
            "dinov2_vitb14",
            "cleandift_sd15",
            "raw_dift_sd15",
        ),
    )
    parser.add_argument("--weights")
    parser.add_argument("--weight-sha256")
    parser.add_argument("--source-repo")
    parser.add_argument("--base-model")
    parser.add_argument("--policy", default="tight")
    parser.add_argument("--resolution", type=int, default=224, choices=(224,))
    parser.add_argument("--views", default="d4", choices=("identity", "d4"))
    parser.add_argument("--annotation-list")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compute-dtype",
        default="float16",
        choices=("float32", "float16", "bfloat16"),
    )
    parser.add_argument("--storage-dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--include-patch-mean", action="store_true")
    parser.add_argument("--latent-policy", default="mode", choices=("mode", "sample"))
    parser.add_argument("--raw-ensemble-sizes", default="1,4,8")
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--image-cache-size", type=int, default=8)
    return parser.parse_args(argv)


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"commit": commit, "dirty": bool(dirty), "dirty_count": len(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "dirty_count": None}


def _runtime_meta(device: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "git": _git_state(),
    }
    try:
        import torch
        import torchvision

        result.update(
            {
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "torch_cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            }
        )
        if device.startswith("cuda") and torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(torch.device(device))
    except ImportError:
        result["torch"] = None
    return result


def _teacher_options(args: argparse.Namespace) -> dict[str, Any]:
    lock: dict[str, Any] = {}
    if args.asset_lock:
        lock_path = Path(args.asset_lock).expanduser().resolve()
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        files = lock.get("files", {})
        repositories = lock.get("repositories", {})
        defaults: dict[str, dict[str, Any]] = {
            "convnext_tiny": {
                "weights": files.get("convnext_tiny", {}).get("path"),
                "weight_sha256": files.get("convnext_tiny", {}).get("sha256"),
            },
            "dinov2_vits14": {
                "weights": files.get("dinov2_vits14", {}).get("path"),
                "weight_sha256": files.get("dinov2_vits14", {}).get("sha256"),
                "repo": repositories.get("dinov2", {}).get("path"),
            },
            "dinov2_vitb14": {
                "weights": files.get("dinov2_vitb14", {}).get("path"),
                "weight_sha256": files.get("dinov2_vitb14", {}).get("sha256"),
                "repo": repositories.get("dinov2", {}).get("path"),
            },
            "cleandift_sd15": {
                "weights": files.get("cleandift_sd15", {}).get("path"),
                "weight_sha256": files.get("cleandift_sd15", {}).get("sha256"),
                "repo": repositories.get("cleandift", {}).get("path"),
                "base_model": lock.get("stable_diffusion_v15", {}).get("path"),
            },
            "raw_dift_sd15": {
                "repo": repositories.get("cleandift", {}).get("path"),
                "base_model": lock.get("stable_diffusion_v15", {}).get("path"),
            },
        }
    else:
        defaults = {}
    selected_defaults = defaults.get(args.teacher, {})
    values: dict[str, Any] = {
        "weights": args.weights or selected_defaults.get("weights"),
        "weight_sha256": args.weight_sha256
        or selected_defaults.get("weight_sha256"),
        "repo": args.source_repo or selected_defaults.get("repo"),
        "base_model": args.base_model or selected_defaults.get("base_model"),
        "include_patch_mean": args.include_patch_mean,
        "latent_policy": args.latent_policy,
        "raw_ensemble_sizes": tuple(
            int(value) for value in args.raw_ensemble_sizes.split(",") if value.strip()
        ),
        "global_seed": args.global_seed,
    }
    required: dict[str, tuple[str, ...]] = {
        "mock": (),
        "convnext_tiny": ("weights",),
        "dinov2_vits14": ("weights", "weight_sha256", "repo"),
        "dinov2_vitb14": ("weights", "weight_sha256", "repo"),
        "cleandift_sd15": ("weights", "repo", "base_model"),
        "raw_dift_sd15": ("repo", "base_model"),
    }
    missing = [name for name in required[args.teacher] if not values.get(name)]
    if missing:
        raise ValueError(f"teacher {args.teacher} 缺少参数: {missing}")
    return values


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _implementation_fingerprint() -> dict[str, Any]:
    repo = Path(__file__).resolve().parent.parent
    relatives = (
        "scripts/extract_p04_features.py",
        "src/rsdet/data/crop_classification.py",
        "src/rsdet/features/p04_cache.py",
        "src/rsdet/features/p04_inputs.py",
        "src/rsdet/features/p04_teachers.py",
    )
    files = {relative: sha256_file(repo / relative) for relative in relatives}
    return {"files": files, "fingerprint": stable_json_sha256(files)}


def _configure_deterministic_runtime(teacher: str, device: str, seed: int) -> None:
    if teacher == "mock":
        return
    import torch

    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0 or args.shard_size <= 0:
        raise ValueError("batch-size/shard-size 必须大于 0")
    if args.teacher == "raw_dift_sd15" and args.batch_size != 1:
        raise ValueError("raw_dift_sd15 必须使用 --batch-size 1")
    manifest = Path(args.manifest).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    if not manifest.is_file() or not data_root.is_dir():
        raise FileNotFoundError("manifest 或 data-root 不存在")
    records = load_all_crop_records(manifest, crop_policy=args.policy)
    records = filter_records(
        records,
        allowlist=load_annotation_allowlist(args.annotation_list),
        max_samples=args.max_samples,
    )
    if not records:
        raise ValueError("筛选后无对象")
    view_ids = ("r0",) if args.views == "identity" else D4_VIEW_IDS
    _configure_deterministic_runtime(args.teacher, args.device, args.global_seed)
    teacher = build_teacher(
        args.teacher,
        device=args.device,
        compute_dtype=args.compute_dtype,
        options=_teacher_options(args),
    )
    torch_for_memory = None
    if args.device.startswith("cuda") and args.teacher != "mock":
        import torch as torch_for_memory

        torch_for_memory.cuda.reset_peak_memory_stats(torch_for_memory.device(args.device))

    preprocess_contract = {
        "canonical_resolution": args.resolution,
        "render": "P03 PIL EXTENT bicubic RGB pad-black direct-square-resize",
        "views": list(view_ids),
        "diffusion_resize": (
            "canonical224 bicubic align_corners_false antialias_true clamp_0_1 to 512"
        ),
        "policy": args.policy,
    }
    metadata = {
        "teacher": teacher.metadata(),
        "asset_lock_sha256": sha256_file(args.asset_lock) if args.asset_lock else None,
        "manifest_sha256": sha256_file(manifest),
        "policy": args.policy,
        "canonical_resolution": args.resolution,
        "preprocessing": preprocess_contract,
        "preprocessing_fingerprint": preprocessing_fingerprint(preprocess_contract),
        "view_ids": list(view_ids),
        "object_count": len(records),
        "row_count": len(records) * len(view_ids),
        "annotation_list_sha256": sha256_file(args.annotation_list)
        if args.annotation_list
        else None,
        "extractor_implementation": _implementation_fingerprint(),
    }
    output = Path(args.output_dir).expanduser().resolve()
    writer = FeatureCacheWriter(
        output,
        metadata=metadata,
        feature_names=teacher.feature_names,
        storage_dtype=args.storage_dtype,
    )
    dataset = CropClassificationDataset(
        records,
        data_root,
        args.resolution,
        transform=None,
        image_cache_size=args.image_cache_size,
    )
    entries = [(record_index, view_id) for record_index in range(len(records)) for view_id in view_ids]
    shard_count = math.ceil(len(entries) / args.shard_size)
    start = time.perf_counter()
    processed_rows = 0
    computed_rows = 0
    skipped_rows = 0
    skipped_shards = 0
    for shard_index in range(shard_count):
        shard_entries = entries[
            shard_index * args.shard_size : (shard_index + 1) * args.shard_size
        ]
        if writer.valid_existing_shard(shard_index, len(shard_entries)):
            processed_rows += len(shard_entries)
            skipped_rows += len(shard_entries)
            skipped_shards += 1
            print(f"SKIP shard={shard_index:05d} rows={len(shard_entries)}", flush=True)
            continue
        canonical: dict[int, Any] = {}
        canonical_sha: dict[int, str] = {}
        for record_index, _ in shard_entries:
            if record_index not in canonical:
                image, _, _ = dataset[record_index]
                canonical[record_index] = image
                canonical_sha[record_index] = canonical_image_sha256(image)

        row_data: dict[str, list[Any]] = {
            "annotation_uid": [],
            "crop_id": [],
            "canonical_input_sha256": [],
            "view_id": [],
            "source_image_id": [],
            "class_id_at_extraction": [],
        }
        feature_chunks: dict[str, list[np.ndarray]] = {
            name: [] for name in teacher.feature_names
        }
        for batch_start in range(0, len(shard_entries), args.batch_size):
            batch_entries = shard_entries[batch_start : batch_start + args.batch_size]
            images = [apply_d4_view(canonical[index], view_id) for index, view_id in batch_entries]
            keys = [f"{records[index].annotation_uid}|{view_id}" for index, view_id in batch_entries]
            outputs = teacher.extract(images, sample_keys=keys)
            if set(outputs) != set(teacher.feature_names):
                raise ValueError("teacher 返回的 feature keys 与合同不一致")
            for name, values in outputs.items():
                array = np.asarray(values, dtype=np.float32)
                if array.ndim != 2 or array.shape[0] != len(batch_entries):
                    raise ValueError(f"teacher feature {name} shape 非法: {array.shape}")
                feature_chunks[name].append(array)
            for record_index, view_id in batch_entries:
                record = records[record_index]
                row_data["annotation_uid"].append(record.annotation_uid)
                row_data["crop_id"].append(record.crop_id)
                row_data["canonical_input_sha256"].append(canonical_sha[record_index])
                row_data["view_id"].append(view_id)
                row_data["source_image_id"].append(record.source_image_id)
                row_data["class_id_at_extraction"].append(record.class_id)
        features = {name: np.concatenate(values, axis=0) for name, values in feature_chunks.items()}
        writer.write_shard(shard_index, row_data, features)
        processed_rows += len(shard_entries)
        computed_rows += len(shard_entries)
        elapsed = time.perf_counter() - start
        rate = processed_rows / elapsed if elapsed else 0.0
        print(
            f"DONE shard={shard_index:05d}/{shard_count:05d} "
            f"rows={processed_rows}/{len(entries)} rate={rate:.3f}/s",
            flush=True,
        )
    index = writer.finalize(shard_count, len(entries))
    elapsed = time.perf_counter() - start
    summary = {
        "status": "complete",
        "teacher": args.teacher,
        "feature_names": list(teacher.feature_names),
        "object_count": len(records),
        "row_count": len(entries),
        "shard_count": shard_count,
        "computed_rows_this_invocation": computed_rows,
        "skipped_rows_this_invocation": skipped_rows,
        "skipped_shards_this_invocation": skipped_shards,
        "elapsed_seconds": elapsed,
        "rows_per_second_including_resume": len(entries) / elapsed if elapsed else None,
        "peak_cuda_memory_bytes": int(torch_for_memory.cuda.max_memory_allocated())
        if torch_for_memory is not None
        else None,
        "cache_fingerprint": index["config_fingerprint"],
        "runtime": _runtime_meta(args.device),
    }
    extraction_summary = output / "extraction_summary.json"
    if computed_rows == 0 and extraction_summary.is_file():
        summary["previous_extraction_summary_sha256"] = sha256_file(extraction_summary)
        _write_json(output / "resume_check_summary.json", summary)
    else:
        _write_json(extraction_summary, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
