#!/usr/bin/env python3
"""P04 环境、资产锁和数据入口门禁。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from rsdet.data.crop_classification import load_crop_records, validate_fold_isolation
from rsdet.features.p04_cache import sha256_file, stable_json_sha256

EXPECTED_PACKAGES = {
    "numpy": "1.26.4",
    "Pillow": "10.4.0",
    "PyYAML": "6.0.2",
    "safetensors": "0.4.5",
    "huggingface-hub": "0.26.2",
    "transformers": "4.46.3",
    "diffusers": "0.32.2",
    "accelerate": "1.1.1",
    "einops": "0.8.0",
    "jaxtyping": "0.2.36",
    "scipy": "1.14.1",
    "scikit-learn": "1.5.2",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P04 environment/data/assets gate")
    parser.add_argument("--asset-lock", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify-sd-inventory", action="store_true")
    parser.add_argument("--expected-gpu")
    parser.add_argument(
        "--verify-source-count",
        type=int,
        default=32,
        help="按稳定路径顺序核对源图尺寸与 SHA；0 表示核对全部",
    )
    return parser.parse_args(argv)


def _git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    failures: list[str] = []
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {},
    }
    if sys.version_info[:3] != (3, 10, 12):
        failures.append(f"Python 必须是 3.10.12，当前 {sys.version_info[:3]}")
    if args.verify_source_count < 0:
        raise ValueError("--verify-source-count 不得为负")
    for name, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        report["packages"][name] = {"expected": expected, "actual": actual}
        if actual != expected:
            failures.append(f"{name} expected={expected}, actual={actual}")
    try:
        import torch
        import torchvision

        report["torch"] = {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else None,
        }
        if torch.__version__ != "2.5.1+cu121":
            failures.append(f"torch 必须是 2.5.1+cu121，当前 {torch.__version__}")
        if torchvision.__version__ != "0.20.1+cu121":
            failures.append(
                f"torchvision 必须是 0.20.1+cu121，当前 {torchvision.__version__}"
            )
        if torch.version.cuda != "12.1":
            failures.append(f"PyTorch CUDA runtime 必须是 12.1，当前 {torch.version.cuda}")
        if not torch.cuda.is_available():
            failures.append("CUDA 不可用")
        elif args.expected_gpu and torch.cuda.get_device_name(0) != args.expected_gpu:
            failures.append(
                f"GPU expected={args.expected_gpu}, actual={torch.cuda.get_device_name(0)}"
            )
    except ImportError as error:
        failures.append(f"PyTorch import 失败: {error}")

    manifest = Path(args.manifest).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    actual_manifest_sha = sha256_file(manifest) if manifest.is_file() else None
    report["data"] = {
        "manifest": str(manifest),
        "expected_manifest_sha256": args.expected_manifest_sha256,
        "actual_manifest_sha256": actual_manifest_sha,
        "data_root": str(data_root),
        "data_root_exists": data_root.is_dir(),
    }
    if actual_manifest_sha != args.expected_manifest_sha256:
        failures.append("manifest SHA-256 不匹配")
    if not data_root.is_dir():
        failures.append("data-root 不存在")
    source_checks: dict[str, Any] = {
        "fold_counts": {},
        "source_files_checked": 0,
        "source_file_sha256_verified": False,
    }
    if actual_manifest_sha == args.expected_manifest_sha256 and data_root.is_dir():
        try:
            all_records = []
            for fold in range(3):
                train = load_crop_records(
                    manifest, crop_policy="tight", held_out_fold=fold, split="train"
                )
                val = load_crop_records(
                    manifest, crop_policy="tight", held_out_fold=fold, split="val"
                )
                validate_fold_isolation(train, val)
                source_checks["fold_counts"][f"fold{fold}_train"] = len(train)
                source_checks["fold_counts"][f"fold{fold}_val"] = len(val)
                if fold == 0:
                    all_records = train + val
            by_source: dict[str, Any] = {}
            for record in all_records:
                previous = by_source.get(record.source_relative_path)
                if previous is not None and (
                    previous.source_checksum_sha256 != record.source_checksum_sha256
                    or previous.source_width != record.source_width
                    or previous.source_height != record.source_height
                ):
                    raise ValueError(f"{record.source_relative_path} 源信息冲突")
                by_source[record.source_relative_path] = record
            source_records = [by_source[key] for key in sorted(by_source)]
            if args.verify_source_count:
                source_records = source_records[: args.verify_source_count]
            for record in source_records:
                path = (data_root / record.source_relative_path).resolve()
                try:
                    path.relative_to(data_root)
                except ValueError as error:
                    raise ValueError(f"源图路径逃逸 data-root: {path}") from error
                if not path.is_file():
                    raise FileNotFoundError(path)
                with Image.open(path) as image:
                    if image.size != (record.source_width, record.source_height):
                        raise ValueError(f"{path} 尺寸与 manifest 不一致")
                if sha256_file(path) != record.source_checksum_sha256:
                    raise ValueError(f"{path} SHA-256 与 manifest 不一致")
            source_checks["source_files_checked"] = len(source_records)
            source_checks["source_file_sha256_verified"] = True
        except (OSError, ValueError) as error:
            failures.append(f"数据/fold/source 门禁失败: {error}")
    report["data"].update(source_checks)

    lock_path = Path(args.asset_lock).expanduser().resolve()
    if not lock_path.is_file():
        failures.append("asset lock 不存在")
        lock = {}
    else:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    report["asset_lock"] = {
        "path": str(lock_path),
        "sha256": sha256_file(lock_path) if lock_path.is_file() else None,
        "fingerprint": lock.get("asset_lock_fingerprint"),
    }
    if lock:
        fingerprint_payload = dict(lock)
        recorded_fingerprint = fingerprint_payload.pop("asset_lock_fingerprint", None)
        actual_fingerprint = stable_json_sha256(fingerprint_payload)
        report["asset_lock"]["recomputed_fingerprint"] = actual_fingerprint
        if recorded_fingerprint != actual_fingerprint:
            failures.append("asset lock fingerprint 不匹配")
    for name, item in lock.get("files", {}).items():
        path = Path(item["path"])
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            failures.append(f"asset {name} 缺失或大小改变")
        elif sha256_file(path) != item["sha256"]:
            failures.append(f"asset {name} SHA-256 改变")
    for name, item in lock.get("repositories", {}).items():
        path = Path(item["path"])
        try:
            actual = _git_commit(path)
        except (OSError, subprocess.CalledProcessError):
            actual = None
        if actual != item["commit"]:
            failures.append(f"repo {name} commit 改变")
    if args.verify_sd_inventory and lock:
        sd = lock["stable_diffusion_v15"]
        root = Path(sd["path"])
        observed = {}
        for item in sd["inventory"]:
            path = root / item["relative_path"]
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                failures.append(f"SD1.5 file 改变: {item['relative_path']}")
            else:
                observed[item["relative_path"]] = item["sha256"]
        if stable_json_sha256(observed) != sd["inventory_fingerprint"]:
            failures.append("SD1.5 inventory fingerprint 不匹配")
    report["status"] = "pass" if not failures else "fail"
    report["failures"] = failures
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
