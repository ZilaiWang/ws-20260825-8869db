#!/usr/bin/env python3
"""在同一输入目录上执行提交配置矩阵并保存标准 COCO 预测与时延。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rsdet.submission.competition import run_submission


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []
    for source in args.configs:
        config = json.loads(source.read_text(encoding="utf-8"))
        config["model"]["weight_path"] = str(args.weights.resolve())
        name = source.stem
        run_dir = args.output_dir / name
        materialized = run_dir / "config.materialized.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        materialized.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        start = time.perf_counter()
        payload = run_submission(args.input_dir, run_dir, materialized)
        elapsed = time.perf_counter() - start
        predictions: list[dict[str, object]] = []
        for image_index, image in enumerate(payload["images"], start=1):
            for item in image["objects"]:
                x1, y1, x2, y2 = item["bbox"]
                predictions.append(
                    {
                        "image_id": image_index,
                        "category_id": item["category_id"],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": item["score"],
                    }
                )
        (run_dir / "predictions.json").write_text(
            json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summary.append(
            {
                "name": name,
                "workpoint_id": config["workpoint_id"],
                "images": len(payload["images"]),
                "predictions": len(predictions),
                "wall_seconds": elapsed,
                "average_seconds": elapsed / max(1, len(payload["images"])),
            }
        )
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
