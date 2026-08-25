"""基于 Ultralytics 的可复现多阶段训练编排。

从 xh-202625-model 仓库合并而来（2026-08-15，源自 weste 旧服务器抢救回传）。
与主仓库 engine/trainer.py（BHC-DETR 论文引擎）并存：本模块用于 M1(YOLO26-s)
与 M3(RT-DETR-L) 的正式 CV3 OOF 训练，model.family 支持 rtdetr / yolo。
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from rsdet.data.imbalance import PreparedUltralyticsData, prepare_ultralytics_data
from rsdet.models.ultralytics_adapter import create_ultralytics_model
from rsdet.utils.seed import set_seed


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} 必须是映射")
    return dict(value)


def _required_text(config: Mapping[str, Any], key: str, scope: str) -> str:
    value = str(config.get(key, "")).strip()
    if not value:
        raise ValueError(f"{scope}.{key} 不能为空")
    return value


def _checkpoint_from_model(model: Any) -> tuple[str, str]:
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        raise RuntimeError("Ultralytics 训练结束后没有 trainer，无法定位权重")

    def existing_checkpoint(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        path = Path(text).expanduser()
        if not path.is_file() or path.stat().st_size <= 0:
            return ""
        return str(path.resolve())

    raw_best = getattr(trainer, "best", "")
    raw_last = getattr(trainer, "last", "")
    best = existing_checkpoint(raw_best)
    last = existing_checkpoint(raw_last)
    if not best and not last:
        raise RuntimeError(
            "Ultralytics trainer 未生成非空 best/last 权重；"
            f"reported best={raw_best!s}, last={raw_last!s}"
        )
    return best, last


def _prepare_data(config: Mapping[str, Any], output_dir: Path, seed: int) -> PreparedUltralyticsData:
    data_config = _mapping(config.get("data", {}), "data")
    imbalance = _mapping(config.get("imbalance", {}), "imbalance")
    return prepare_ultralytics_data(
        data_root=_required_text(data_config, "root", "data"),
        manifest_path=_required_text(data_config, "manifest", "data"),
        output_dir=output_dir / "prepared_data",
        seed=seed,
        repeat_threshold=float(imbalance.get("repeat_threshold", 0.05)),
        max_repeat=float(imbalance.get("max_repeat", 4.0)),
        effective_beta=float(imbalance.get("effective_beta", 0.9999)),
        max_class_weight=float(imbalance.get("max_class_weight", 5.0)),
    )


def validate_training_config(config: Mapping[str, Any]) -> None:
    """在访问数据/GPU 前校验训练配置结构。"""
    model = _mapping(config.get("model", {}), "model")
    _required_text(model, "family", "model")
    _required_text(model, "weights", "model")
    data = _mapping(config.get("data", {}), "data")
    _required_text(data, "root", "data")
    _required_text(data, "manifest", "data")
    train_config = _mapping(config.get("train", {}), "train")
    checkpoint_selection = str(
        train_config.get("checkpoint_selection", "best")
    ).strip()
    if checkpoint_selection not in {"best", "last"}:
        raise ValueError("train.checkpoint_selection 只允许 best 或 last")
    stages = train_config.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("train.stages 必须是非空列表")
    seen_names: set[str] = set()
    for index, stage_value in enumerate(stages):
        stage = _mapping(stage_value, f"train.stages[{index}]")
        name = _required_text(stage, "name", f"train.stages[{index}]")
        if name in seen_names:
            raise ValueError(f"训练阶段名称重复: {name}")
        if int(stage.get("epochs", 0)) <= 0:
            raise ValueError(f"训练阶段 {name} 的 epochs 必须 > 0")
        arguments = stage.get("args", {})
        if not isinstance(arguments, Mapping):
            raise TypeError(f"训练阶段 {name} 的 args 必须是映射")
        seen_names.add(name)


def train(
    config: dict[str, Any],
    *,
    resume: str | Path | None = None,
    device: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """准备不均衡数据并依次运行基础训练与稀有类再平衡微调。

    ``dry_run`` 仍会读取 manifest、检查标签并生成派生清单，但不会导入
    Ultralytics 或启动 GPU 训练。
    """
    validate_training_config(config)
    resolved = deepcopy(config)
    seed = int(resolved.get("seed", 42))
    set_seed(seed)
    output_dir = Path(str(resolved.get("output_dir", "outputs/train"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = _prepare_data(resolved, output_dir, seed)

    resolved_path = output_dir / "resolved_config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    model_config = _mapping(resolved["model"], "model")
    train_config = _mapping(resolved["train"], "train")
    stages = list(train_config["stages"])
    checkpoint_selection = str(
        train_config.get("checkpoint_selection", "best")
    ).strip()
    summary: dict[str, Any] = {
        "dry_run": dry_run,
        "seed": seed,
        "model_family": model_config["family"],
        "initial_weights": str(resume or model_config["weights"]),
        "checkpoint_selection": checkpoint_selection,
        "prepared_data": {
            "base_yaml": str(prepared.base_yaml),
            "balanced_yaml": str(prepared.balanced_yaml),
            "statistics": str(prepared.statistics_json),
        },
        "stages": [],
    }

    common_args = _mapping(train_config.get("args", {}), "train.args")
    current_weights = str(resume or model_config["weights"])
    selected_device = device or str(train_config.get("device", "cuda:0"))
    for index, stage_value in enumerate(stages):
        stage = _mapping(stage_value, f"train.stages[{index}]")
        name = str(stage["name"])
        use_balanced = bool(stage.get("balanced", False))
        data_yaml = prepared.balanced_yaml if use_balanced else prepared.base_yaml
        arguments = dict(common_args)
        arguments.update(_mapping(stage.get("args", {}), f"train.stages[{index}].args"))
        arguments.update(
            {
                "data": str(data_yaml),
                "epochs": int(stage["epochs"]),
                "device": selected_device,
                "seed": seed,
                "project": str(output_dir / "runs"),
                "name": name,
                "exist_ok": True,
            }
        )
        if index == 0 and resume is not None:
            arguments["resume"] = True

        stage_summary: dict[str, Any] = {
            "name": name,
            "balanced": use_balanced,
            "input_weights": current_weights,
            "data_yaml": str(data_yaml),
            "arguments": arguments,
        }
        if not dry_run:
            model = create_ultralytics_model(str(model_config["family"]), current_weights)
            model.train(**arguments)
            best, last = _checkpoint_from_model(model)
            selected = (
                last
                if checkpoint_selection == "last"
                else best or last
            )
            if not selected:
                raise RuntimeError(
                    f"训练阶段 {name} 未生成 checkpoint_selection="
                    f"{checkpoint_selection} 所需权重"
                )
            current_weights = selected
            stage_summary.update(
                {
                    "best": best,
                    "last": last,
                    "selected_checkpoint": selected,
                    "checkpoint_selection": checkpoint_selection,
                }
            )
        summary["stages"].append(stage_summary)

    summary_path = output_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


__all__ = ["train", "validate_training_config"]
