#!/usr/bin/env python3
"""P0-3 服务器环境和输入资产门禁检查。"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from rsdet.data.crop_classification import load_crop_records, validate_fold_isolation
from rsdet.models.crop_classifier import CONVNEXT_TINY_WEIGHT_SHA256, sha256_file

EXPECTED_MANIFEST_SHA256 = "f259cd33542f4bfaad8f6af31cc71a87819fe3e4fd27ebd9a8b3da5922a0e26e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 P0-3 服务器环境、manifest、数据和权重")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--expected-manifest-sha256",
        default=EXPECTED_MANIFEST_SHA256,
        help="默认保持 P0-2 历史合同；正式复验必须显式传 formal manifest SHA",
    )
    parser.add_argument(
        "--verify-source-count",
        type=int,
        default=32,
        help="校验的唯一源图数；0 表示全部",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    import torchvision
    from PIL import Image

    if torch.__version__.split("+")[0] != "2.5.1":
        raise RuntimeError(f"torch 应为 2.5.1，当前 {torch.__version__}")
    if torchvision.__version__.split("+")[0] != "0.20.1":
        raise RuntimeError(f"torchvision 应为 0.20.1，当前 {torchvision.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    weight_sha = sha256_file(args.weights)
    if weight_sha != CONVNEXT_TINY_WEIGHT_SHA256:
        raise RuntimeError(f"权重 SHA256 不匹配: {weight_sha}")
    manifest_sha = sha256_file(args.manifest)
    if manifest_sha != args.expected_manifest_sha256:
        raise RuntimeError(
            "manifest SHA256 不匹配: "
            f"expected={args.expected_manifest_sha256}, actual={manifest_sha}"
        )
    if args.verify_source_count < 0:
        raise ValueError("--verify-source-count 不得为负")

    data_root = Path(args.data_root).expanduser().resolve()
    checks: dict[str, int] = {}
    all_records = []
    for fold in range(3):
        train = load_crop_records(
            args.manifest, crop_policy="tight", held_out_fold=fold, split="train"
        )
        val = load_crop_records(args.manifest, crop_policy="tight", held_out_fold=fold, split="val")
        validate_fold_isolation(train, val)
        checks[f"fold{fold}_train"] = len(train)
        checks[f"fold{fold}_val"] = len(val)
        if fold == 0:
            all_records = train + val

    by_source: dict[str, object] = {}
    for record in all_records:
        previous = by_source.get(record.source_relative_path)
        if previous is not None and (
            previous.source_checksum_sha256 != record.source_checksum_sha256
            or previous.source_width != record.source_width
            or previous.source_height != record.source_height
        ):
            raise RuntimeError(f"{record.source_relative_path} 的源信息在 manifest 中冲突")
        by_source[record.source_relative_path] = record
    source_records = [by_source[key] for key in sorted(by_source)]
    if args.verify_source_count:
        source_records = source_records[: args.verify_source_count]
    for record in source_records:
        path = (data_root / record.source_relative_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            if image.size != (record.source_width, record.source_height):
                raise RuntimeError(f"{path} 尺寸与 manifest 不一致")
        actual_sha = sha256_file(path)
        if actual_sha != record.source_checksum_sha256:
            raise RuntimeError(f"{path} SHA256 与 manifest 不一致")

    result = {
        "status": "pass",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "weight_sha256": weight_sha,
        "manifest_sha256": manifest_sha,
        "fold_counts": checks,
        "source_files_checked": len(source_records),
        "source_file_sha256_verified": True,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
