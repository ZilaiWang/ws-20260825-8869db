#!/usr/bin/env python3
"""V1-FULL-P2：高分辨率候选通路（stride-4 P2 检测头）验证实验。

对齐 M1（yolo26s-1024-CV3）严格单因素：
- 模型：yolo26-p2s.pt（官方 P2 变体，Detect 四输入 P2/P3/P4/P5）
- 数据：CV3 三折，与 M1 完全相同的 fold 划分
- 训练：与 M1 相同超参（imgsz 1024 / batch 12 / AdamW / 160 epoch）
- 唯一变化：模型多了 P2 检测通路

用法（GPU）：
    PYTHONPATH=src python scripts/n1c_p2_train.py \
        --cv3-manifest data/splits/cv3_airport_proxy_k60_v2.json \
        --data-root /workspace/data \
        --fold 0 \
        --output-dir /workspace/results/P2-FOLD0 \
        --epochs 160
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from rsdet.utils.logging import setup_logging

logger = setup_logging(name="n1c_p2_train")

CLASS_NAMES_25 = [str(i) for i in range(25)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V1-FULL-P2 高分辨率通路实验")
    parser.add_argument("--cv3-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument(
        "--model-yaml",
        default="yolo26-p2s.yaml",
        help="P2 模型定义（s 尺度，ultralytics 内置）",
    )
    parser.add_argument(
        "--pretrained",
        default="yolo26s.pt",
        help="迁移来源权重（yolo26s COCO 预训练，公共层复用，P2 头随机初始化）",
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args(argv)


def build_dataset_yaml(
    cv3_manifest: Path,
    data_root: Path,
    fold: int,
    output_dir: Path,
) -> Path:
    """从 CV3 划分生成 fold 的 ultralytics dataset.yaml。

    fold X：训练 = 其余两折图像，验证 = fold X 图像（与 M1 一致）。
    返回 dataset.yaml 路径。
    """
    manifest = json.loads(cv3_manifest.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if len(samples) != 4481:
        raise ValueError(f"CV3 划分图像数 {len(samples)} != 4481")

    train_paths: list[str] = []
    val_paths: list[str] = []
    for sample in samples:
        rel = sample["relative_path"]
        if sample["fold"] == fold:
            val_paths.append(rel)
        else:
            train_paths.append(rel)

    if not train_paths or not val_paths:
        raise ValueError(f"fold {fold}: 训练 {len(train_paths)} / 验证 {len(val_paths)} 有空集合")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_txt = output_dir / f"train_fold{fold}.txt"
    val_txt = output_dir / f"val_fold{fold}.txt"
    train_txt.write_text(
        "\n".join(str(data_root / p) for p in train_paths) + "\n", encoding="utf-8"
    )
    val_txt.write_text("\n".join(str(data_root / p) for p in val_paths) + "\n", encoding="utf-8")

    dataset_yaml = output_dir / f"dataset_fold{fold}.yaml"
    dataset_yaml.write_text(
        f"# CV3 airport_proxy_k60_v2 fold{fold} (与 M1 相同划分, git 可回溯)\n"
        f"path: {data_root}\n"
        f"train: {train_txt}\n"
        f"val: {val_txt}\n"
        f"nc: 25\n"
        f"names: {CLASS_NAMES_25}\n",
        encoding="utf-8",
    )
    logger.info("训练图 %d / 验证图 %d", len(train_paths), len(val_paths))
    return dataset_yaml


def main(argv: list[str] | None = None) -> int:
    """运行 P2 训练。"""
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        logger.error("输出目录非空，禁止覆盖: %s", output_dir)
        return 1
    try:
        from ultralytics import YOLO

        dataset_yaml = build_dataset_yaml(
            args.cv3_manifest,
            Path(args.data_root).expanduser().resolve(),
            args.fold,
            output_dir,
        )

        # 配置记录（可回溯）。
        config_payload = {
            "experiment": "V1-FULL-P2-fold" + str(args.fold),
            "model_yaml": args.model_yaml,
            "pretrained": args.pretrained,
            "data_yaml": str(dataset_yaml),
            "fold": args.fold,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "epochs": args.epochs,
            "cv3_manifest": str(args.cv3_manifest),
            "cv3_manifest_sha256": _sha256(args.cv3_manifest),
            "checkpoint_selection": "fixed_epoch_last",
            "evaluation_contract": "formal_comparison_requires_same_160_epochs_and_last_pt",
            "note": "与 M1 单因素对照：唯一差异 = P2 stride-4 检测通路；"
            "P2 头随机初始化，公共层从 yolo26s COCO 预训练迁移",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "p2_config.json").write_text(
            json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        model = YOLO(args.model_yaml)
        if args.pretrained:
            logger.info("从 %s 迁移公共层权重（P2 头随机初始化）", args.pretrained)
            model.load(args.pretrained)
        model.train(
            data=str(dataset_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            project=str(output_dir),
            name="p2_run",
            exist_ok=False,
            seed=42,
            deterministic=True,
            optimizer="AdamW",
            lr0=0.002,
            lrf=0.01,
            weight_decay=0.0005,
            warmup_epochs=3,
            cos_lr=True,
            amp=True,
            patience=0,
            close_mosaic=20,
        )
    except Exception as error:
        logger.exception("P2 训练失败: %s", error)
        return 1
    logger.info("=== P2 fold %d 完成 ===", args.fold)
    logger.info("固定最后权重: %s", output_dir / "p2_run" / "weights" / "last.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
