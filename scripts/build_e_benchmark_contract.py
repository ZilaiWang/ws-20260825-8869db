#!/usr/bin/env python3
"""生成 E 正式测速的 benchmark contract（runtime_10k_benchmark_v1）。

汇总各输入的 SHA，生成 benchmark_10k_pipeline.py 严格校验的冻结合同：

  contract_version / image_source_type / model_key / image_manifest_sha256
  / checkpoint_sha256 / checkpoint_provenance_sha256 / config_sha256
  / hardware_sha256 / engineering_checkpoint_only / cuda_synchronized
  / timing_method / tile_size / expected_tile_count / warmup_runs
  / minimum_measured_runs / overlap

其中 hardware_sha256 由 --hardware 指定的 hardware.json（服务器现场采集）
计算；其余 SHA 从输入文件直接计算。纯 CPU。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

IMAGE_SOURCE_TYPES = {"real_official", "real_project_proxy", "synthetic", "stitched"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_contract(
    *,
    image_manifest: Path,
    checkpoint: Path,
    checkpoint_provenance: Path,
    config: Path,
    hardware: Path,
    model_key: str,
    image_source_type: str,
    engineering_checkpoint_only: bool,
    tile_size: int,
    overlap: int,
    expected_tile_count: int,
    warmup_runs: int,
    minimum_measured_runs: int,
) -> dict[str, Any]:
    for path, label in (
        (image_manifest, "image_manifest"),
        (checkpoint, "checkpoint"),
        (checkpoint_provenance, "checkpoint_provenance"),
        (config, "config"),
        (hardware, "hardware"),
    ):
        if not path.expanduser().resolve().is_file():
            raise FileNotFoundError(f"{label} 不存在: {path}")
    if image_source_type not in IMAGE_SOURCE_TYPES:
        raise ValueError(f"image_source_type 非法: {image_source_type!r}")
    if not 0 <= overlap < tile_size:
        raise ValueError(f"overlap 必须在 [0, {tile_size}) 内")
    return {
        "contract_version": "runtime_10k_benchmark_v1",
        "image_source_type": image_source_type,
        "model_key": model_key.strip().upper(),
        "image_manifest_sha256": _sha256(image_manifest.expanduser().resolve()),
        "checkpoint_sha256": _sha256(checkpoint.expanduser().resolve()),
        "checkpoint_provenance_sha256": _sha256(checkpoint_provenance.expanduser().resolve()),
        "config_sha256": _sha256(config.expanduser().resolve()),
        "hardware_sha256": _sha256(hardware.expanduser().resolve()),
        "engineering_checkpoint_only": bool(engineering_checkpoint_only),
        "cuda_synchronized": True,
        "timing_method": "perf_counter_with_torch_cuda_synchronize",
        "tile_size": int(tile_size),
        "overlap": int(overlap),
        "expected_tile_count": int(expected_tile_count),
        "warmup_runs": int(warmup_runs),
        "minimum_measured_runs": int(minimum_measured_runs),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-provenance", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--model-key", default="M1")
    parser.add_argument("--image-source-type", default="synthetic")
    parser.add_argument("--engineering-checkpoint-only", action="store_true")
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--expected-tile-count", type=int, default=100)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--minimum-measured-runs", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    contract = build_contract(
        image_manifest=args.image_manifest,
        checkpoint=args.checkpoint,
        checkpoint_provenance=args.checkpoint_provenance,
        config=args.config,
        hardware=args.hardware,
        model_key=args.model_key,
        image_source_type=args.image_source_type,
        engineering_checkpoint_only=args.engineering_checkpoint_only,
        tile_size=args.tile_size,
        overlap=args.overlap,
        expected_tile_count=args.expected_tile_count,
        warmup_runs=args.warmup_runs,
        minimum_measured_runs=args.minimum_measured_runs,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(contract, ensure_ascii=False, indent=2))
    print(f"CONTRACT_SHA256={_sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
