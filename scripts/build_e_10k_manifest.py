#!/usr/bin/env python3
"""生成 E 正式测速的 image manifest（e_10k_image_manifest_v1）。

扫描 data_root 下的 10000×10000 合成图（不同内容，SHA 互异），生成
benchmark_10k_pipeline.py 严格校验的 manifest：

  {"version": "e_10k_image_manifest_v1",
   "samples": [{"image_id", "width", "height", "image_source_type",
                "relative_path", "sha256"}, ...]}

image_source_type 支持 real_official / real_project_proxy / synthetic /
stitched（由 --source-type 指定；合成图用 synthetic，官方图用 real_official）。

纯 CPU。输出 manifest + 其 SHA256。
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


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size  # (width, height)


def build_manifest(
    data_root: Path,
    *,
    expected_width: int,
    expected_height: int,
    source_type: str,
    min_count: int,
) -> dict[str, Any]:
    """扫描 data_root 下所有 10K 图，返回 e_10k_image_manifest_v1 文档。"""
    if source_type not in IMAGE_SOURCE_TYPES:
        raise ValueError(f"image_source_type 非法: {source_type!r}")
    candidates: list[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        candidates.extend(data_root.rglob(suffix))
    samples: list[dict[str, Any]] = []
    seen_shas: set[str] = set()
    for path in sorted(candidates):
        width, height = _image_size(path)
        if width != expected_width or height != expected_height:
            continue
        sha = _sha256(path)
        if sha in seen_shas:
            continue  # 内容重复的不进入 manifest（防充数）
        seen_shas.add(sha)
        samples.append(
            {
                "image_id": len(samples) + 1,
                "width": width,
                "height": height,
                "image_source_type": source_type,
                "relative_path": str(path.relative_to(data_root)),
                "sha256": sha,
            }
        )
    if len(samples) < min_count:
        raise ValueError(
            f"data_root 下找到 {len(samples)} 张 10K 图（需 >= {min_count} 张）: {data_root}"
        )
    return {"version": "e_10k_image_manifest_v1", "samples": samples}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-width", type=int, default=10000)
    parser.add_argument("--expected-height", type=int, default=10000)
    parser.add_argument("--source-type", choices=sorted(IMAGE_SOURCE_TYPES), default="synthetic")
    parser.add_argument("--min-count", type=int, default=10)
    args = parser.parse_args(argv)

    manifest = build_manifest(
        args.data_root,
        expected_width=args.expected_width,
        expected_height=args.expected_height,
        source_type=args.source_type,
        min_count=args.min_count,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"MANIFEST_SHA256={_sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
