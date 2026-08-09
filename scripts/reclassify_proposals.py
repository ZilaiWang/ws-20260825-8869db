#!/usr/bin/env python3
"""N2-2：对象学生重分类 M1 候选（GPU）。

用法（GPU）：
    PYTHONPATH=src python scripts/reclassify_proposals.py \
        --manifest outputs/N2-PROPO-CROP/proposal_crop_manifest.csv \
        --data-root /workspace/data \
        --checkpoint /workspace/results/N2-OBJECT-STUDENT/fold0/best_checkpoint.pt \
        --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
        --output /workspace/results/N2-RECLASS/reclassified_fold0.json \
        --mode reclassify \
        --fold 0

产出：每折一个 reclassified JSON（含 proposal_uid、新类别、student_score、
dropped 标记），供后续融合与 cross-fit 评估。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from rsdet.analysis.proposal_reclassification import (
    MODE_BACKGROUND,
    MODE_JOINT,
    MODE_RECLASSIFY,
    reclassify_proposals,
)
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="n2_reclassify")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="N2-2 对象学生重分类")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(MODE_RECLASSIFY, MODE_BACKGROUND, MODE_JOINT),
    )
    parser.add_argument("--fold", type=int, default=None, help="只处理某折")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args(argv)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    """运行 N2-2 重分类。"""
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("设备: %s", device)
    try:
        rows = _load_rows(args.manifest)
        if args.fold is not None:
            rows = [row for row in rows if int(row["fold"]) == args.fold]
            if not rows:
                raise ValueError(f"fold {args.fold} 无候选")
        logger.info("处理 %d 条候选（fold=%s）", len(rows), args.fold)
        results = reclassify_proposals(
            manifest_rows=rows,
            data_root=args.data_root,
            checkpoint_path=args.checkpoint,
            weights_path=args.weights,
            resolution=args.resolution,
            batch_size=args.batch_size,
            device=device,
            mode=args.mode,
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        logger.error("重分类失败: %s", error)
        return 1

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    dropped = sum(1 for r in results if r.get("dropped"))
    output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("=== N2-2 重分类完成 ===")
    logger.info("处理 %d 条，丢弃 %d（背景拒识）", len(results), dropped)
    logger.info("已保存: %s", output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
