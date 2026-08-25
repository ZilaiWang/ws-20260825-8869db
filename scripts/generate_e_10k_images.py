#!/usr/bin/env python3
"""生成 E 测速用的合成 10K 大图（不同 seed，内容互异）。

调用 ``rsdet.tiling.synthetic.generate_synthetic_scene`` 生成 N 张 10000x10000
带 GT 的合成图，保存为 JPEG，供 E 三层 smoke 与跨 tile 路径验证使用。

用法：
    PYTHONPATH=src python scripts/generate_e_10k_images.py \
        --output-dir /workspace/data/10k --count 10 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rsdet.tiling.synthetic import generate_synthetic_scene
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="generate_e_10k_images")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=10000)
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--overlap", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_samples: list[dict[str, Any]] = []
    for index in range(args.count):
        scene = generate_synthetic_scene(
            image_size=args.image_size,
            tile_size=args.tile_size,
            overlap=args.overlap,
            num_ships=10,
            num_aircraft=30,
            num_vehicles=5,
            seed=args.seed + index,
        )
        image = np.asarray(scene.image, dtype=np.uint8)
        path = output_dir / f"synthetic_10k_{index:02d}.jpg"
        Image.fromarray(image).save(path, quality=95)
        logger.info("生成 %s (%dx%d)", path.name, image.shape[1], image.shape[0])
        manifest_samples.append(
            {
                "image_id": index + 1,
                "relative_path": path.name,
                "width": image.shape[1],
                "height": image.shape[0],
            }
        )

    manifest = {"version": "e_10k_synthetic_manifest_v1", "samples": manifest_samples}
    manifest_path = output_dir / "image_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"GENERATED_E_10K count={args.count} dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
