#!/usr/bin/env python3
"""CV3 OOF 正式训练引擎（ultralytics 后端）。

补齐 model 仓库 ``train.py`` 的契约：读 ``materialize_cv3_oof_config.py``
生成的 ``train_request.yaml``，从本折 ``split_view.json`` 生成 ultralytics
``dataset.yaml``，用 ultralytics 原生训练，输出：

- ``output_dir/runs/foundation/weights/last.pt``（固定 last checkpoint）；
- ``output_dir/train_summary.json``（满足 ``_validate_train_summary`` 契约）；
- ``output_dir/resolved_config.yaml``（路径绝对化的 resolved 训练配置）。

支持 ``family=rtdetr`` 与 ``family=yolo``。所有 ultralytics 导入延迟到
函数内，保证无深度学习环境也能 import 本模块。训练契约严格遵循
``src/rsdet/experiments/cv3_oof.py`` 的 ``FORMAL_TRAINING_CONTRACTS``。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from rsdet.utils.logging import setup_logging

logger = setup_logging(name="train_cv3_oof")

CLASS_NAMES_25 = [str(i) for i in range(25)]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"训练配置必须是 YAML 对象: {path}")
    return payload


def _resolve(path: str | Path, base: Path) -> Path:
    result = Path(str(path)).expanduser()
    if not result.is_absolute():
        result = base.parent / result
    return result.resolve()


def build_dataset_yaml(split_view: Path, data_root: Path, output_dir: Path) -> Path:
    """从 split_view.json 生成 ultralytics dataset.yaml 与 train/val 路径表。"""
    view = json.loads(split_view.read_text(encoding="utf-8"))
    samples = view.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("split_view.json 缺少 samples")
    train_rel = [s["relative_path"] for s in samples if s.get("split") == "train"]
    val_rel = [s["relative_path"] for s in samples if s.get("split") == "val"]
    if not train_rel or not val_rel:
        raise ValueError(f"split view 训练 {len(train_rel)} / 验证 {len(val_rel)} 有空集合")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_txt = output_dir / "train.txt"
    val_txt = output_dir / "val.txt"
    train_txt.write_text(
        "\n".join(str(data_root / p) for p in train_rel) + "\n", encoding="utf-8"
    )
    val_txt.write_text("\n".join(str(data_root / p) for p in val_rel) + "\n", encoding="utf-8")

    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        f"# CV3 airport_proxy_k60_v2 fold split view (git 可回溯)\n"
        f"path: {data_root}\n"
        f"train: {train_txt}\n"
        f"val: {val_txt}\n"
        f"nc: 25\n"
        f"names: {CLASS_NAMES_25}\n",
        encoding="utf-8",
    )
    logger.info("训练图 %d / 验证图 %d", len(train_rel), len(val_rel))
    return dataset_yaml


def _build_train_arguments(config: Mapping[str, Any], dataset_yaml: Path) -> dict[str, Any]:
    """构造实际传给 ultralytics train() 的参数，与冻结合同严格一致。"""
    train = config["train"]
    stage = train["stages"][0]
    train_args = dict(train.get("args") or {})
    stage_args = dict(stage.get("args") or {})
    arguments = {
        **train_args,
        **stage_args,
        "epochs": int(stage["epochs"]),
        "device": str(train["device"]),
        "seed": int(config["seed"]),
        "data": str(dataset_yaml),
        "project": str(_resolve(config["output_dir"], Path.cwd()) / "runs"),
        "name": "foundation",
        "exist_ok": False,
    }
    return arguments


def _load_ultralytics_model(family: str, weights: str):
    from ultralytics import RTDETR, YOLO

    normalized = family.strip().lower()
    if normalized == "rtdetr":
        return RTDETR(str(weights))
    if normalized == "yolo":
        return YOLO(str(weights))
    raise ValueError(f"不支持的 ultralytics family: {family!r}")


def _write_resolved_config(config: Mapping[str, Any], output_dir: Path) -> Path:
    resolved = json.loads(json.dumps(config))
    resolved["model"] = dict(resolved["model"])
    resolved["model"]["weights"] = str(_resolve(config["model"]["weights"], Path.cwd()))
    resolved["data"] = dict(resolved["data"])
    resolved["data"]["root"] = str(_resolve(config["data"]["root"], Path.cwd()))
    resolved["data"]["manifest"] = str(_resolve(config["data"]["manifest"], Path.cwd()))
    resolved["output_dir"] = str(_resolve(config["output_dir"], Path.cwd()))
    path = output_dir / "resolved_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _write_train_summary(
    config: Mapping[str, Any],
    arguments: Mapping[str, Any],
    output_dir: Path,
    *,
    dry_run: bool,
) -> None:
    weights_abs = _resolve(config["model"]["weights"], Path.cwd())
    checkpoint = output_dir / "runs" / "foundation" / "weights" / "last.pt"
    stage = config["train"]["stages"][0]
    summary = {
        "dry_run": dry_run,
        "seed": int(config["seed"]),
        "model_family": str(config["model"]["family"]).strip().lower(),
        "checkpoint_selection": "last",
        "initial_weights": str(weights_abs),
        "stages": [
            {
                "name": "foundation",
                "balanced": bool(stage.get("balanced", False)),
                "input_weights": str(weights_abs),
                "arguments": dict(arguments),
                "checkpoint_selection": "last",
                "last": str(checkpoint),
                "selected_checkpoint": str(checkpoint),
            }
        ],
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="只审计配置与数据，不训练")
    parser.add_argument("--resume", type=Path, default=None, help="禁止：正式 OOF 不允许 resume")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume is not None:
        raise ValueError("正式 OOF 训练禁止 --resume")
    config = _load_yaml(args.config)
    output_dir = _resolve(config["output_dir"], args.config)
    data_root = _resolve(config["data"]["root"], args.config)
    split_view = _resolve(config["data"]["manifest"], args.config)
    weights = _resolve(config["model"]["weights"], args.config)

    dataset_yaml = build_dataset_yaml(split_view, data_root, output_dir)
    arguments = _build_train_arguments(config, dataset_yaml)
    _write_resolved_config(config, output_dir)

    if args.dry_run:
        logger.info("dry-run：仅审计，不启动训练")
        _write_train_summary(config, arguments, output_dir, dry_run=True)
        print("TRAIN_DRY_RUN_PASS")
        return 0

    model = _load_ultralytics_model(config["model"]["family"], str(weights))
    model.train(**arguments)

    checkpoint = output_dir / "runs" / "foundation" / "weights" / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"训练未产出 last.pt: {checkpoint}")
    # ultralytics 可能把 last.pt 放在 runs/foundation/weights/last.pt，无需移动。
    _write_train_summary(config, arguments, output_dir, dry_run=False)
    logger.info("训练完成: %s", checkpoint)
    print("TRAIN_OOF_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
