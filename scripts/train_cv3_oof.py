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


def build_dataset_yaml(split_view: Path, data_root: Path, output_dir: Path,
                       hard_image_ids: set[int] | None = None) -> Path:
    """从 split_view.json 生成 ultralytics dataset.yaml 与 train/val 路径表。

    hard_image_ids: E7 困难样本课程——这些图在 train.txt 中重复出现,
    使模型以更高频率采样困难图(默认重复 1 次 = 权重 2x)。
    """
    view = json.loads(split_view.read_text(encoding="utf-8"))
    samples = view.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("split_view.json 缺少 samples")
    id_to_rel = {int(s["image_id"]): s["relative_path"] for s in samples}
    train_rel = [s["relative_path"] for s in samples if s.get("split") == "train"]
    val_rel = [s["relative_path"] for s in samples if s.get("split") == "val"]
    if not train_rel or not val_rel:
        raise ValueError(f"split view 训练 {len(train_rel)} / 验证 {len(val_rel)} 有空集合")

    if hard_image_ids:
        hard_rel = [id_to_rel[i] for i in hard_image_ids if i in id_to_rel]
        if hard_rel:
            logger.info("E7 困难课程: %d 张困难图重复 1 次(训练图 %d→%d)",
                        len(hard_rel), len(train_rel), len(train_rel) + len(hard_rel))
            train_rel = train_rel + hard_rel

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


def _load_ultralytics_model(family: str, weights: str, architecture: str | None = None):
    from ultralytics import RTDETR, YOLO

    normalized = family.strip().lower()
    if normalized == "rtdetr":
        return RTDETR(str(weights))
    if normalized == "yolo":
        if architecture:
            # 自定义结构（如 yolo26s-p2.yaml）：按结构初始化后加载预训练权重迁移。
            return YOLO(str(architecture)).load(str(weights))
        return YOLO(str(weights))
    raise ValueError(f"不支持的 ultralytics family: {family!r}")


def _load_afss_suff_list(suff_json: Path, split_view: Path) -> list[float]:
    """从充分度表 + split_view 生成按训练集顺序的充分度列表（Y4 采样器输入）。

    ``suff_json`` 为 ``afss_diagnose.py`` 输出（含 ``per_image_suff``），
    顺序须与 ultralytics dataset 索引一致，即 ``split_view`` 中
    ``split == "train"`` 的样本顺序（与 ``build_dataset_yaml`` 的 train.txt 一致）。
    """
    payload = json.loads(suff_json.read_text(encoding="utf-8"))
    per_image = payload.get("per_image_suff")
    if not isinstance(per_image, dict) or not per_image:
        raise ValueError(f"充分度表缺少 per_image_suff: {suff_json}")
    view = json.loads(split_view.read_text(encoding="utf-8"))
    suff_list: list[float] = []
    for s in view["samples"]:
        if s.get("split") != "train":
            continue
        key = str(s["image_id"])
        if key not in per_image:
            raise ValueError(f"充分度表缺少 image_id={key}（须用同一 split 计算充分度）")
        suff_list.append(float(per_image[key]))
    return suff_list


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
    # 创新训练期模块（材料 19 Y3/Y4/Y5）
    parser.add_argument(
        "--innovation",
        choices=("none", "y3", "y4", "y5", "coph", "dfd", "family", "worstgroup",
                 "v2veh", "confpair", "obs", "tailw"),
        default="none",
        help="创新训练期模块：none=基线 / y3=层次粗类辅助损失 / y4=AFSS采样 / y5=90°旋转增强 / coph=COPH存在性正则 / dfd=DFD密集前景监督 / family=F2三层(fine+family+coarse)辅助损失 / worstgroup=D4 worst-group加权 / v2veh=V2车辆中心-周围对比 / confpair=F3混淆组内判别 / obs=F4可观测性掩码 / tailw=F6尾类重加权",
    )
    parser.add_argument("--coarse-gain", type=float, default=0.5, help="Y3 粗类辅助损失权重")
    parser.add_argument("--family-gain", type=float, default=0.5, help="F2 family 中间层辅助损失权重")
    parser.add_argument("--wg-gain", type=float, default=1.5, help="D4 worst-group 样本 cls 损失放大倍数")
    parser.add_argument("--confusion-gain", type=float, default=0.5, help="F3/F4 混淆组内判别辅助权重")
    parser.add_argument("--center-gain", type=float, default=0.5, help="V2 车辆中心-周围对比权重")
    parser.add_argument("--tail-weight", type=float, default=2.0, help="F6 尾类 cls 损失放大倍数")
    parser.add_argument("--coph-presence-gain", type=float, default=1.0, help="E8 COPH 存在性正则权重")
    parser.add_argument("--coph-coarse-gain", type=float, default=0.0, help="E8 COPH 粗类辅助权重(默认关)")
    parser.add_argument("--dfd-gain", type=float, default=1.0, help="E9/B4 DFD 密集前景监督权重")
    parser.add_argument("--dfd-sigma-scale", type=float, default=0.15, help="DFD 高斯 sigma 相对 box 尺寸比例")
    parser.add_argument("--dfd-focal-alpha", type=float, default=2.0, help="DFD focal 调制指数")
    parser.add_argument("--dfd-neg-gamma", type=float, default=4.0, help="DFD 负样本 (1-heatmap)^gamma 调制")
    parser.add_argument("--dfd-only-classes", default=None, help="DFD 只对这些细类生成热力图(逗号分隔, V1 车辆种子用 24)")
    parser.add_argument("--hard-curriculum", type=Path, default=None,
                        help="E7 困难样本课程 JSON(目标清单, 这些图训练重复 1 次)")
    parser.add_argument("--suff-json", type=Path, default=None, help="Y4 充分度表(afss_diagnose.py 输出)")
    parser.add_argument("--easy-floor", type=float, default=0.05, help="Y4 容易图最低回看权重")
    parser.add_argument("--rotate90-p", type=float, default=1.0, help="Y5 旋转概率")
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

    hard_image_ids: set[int] | None = None
    if args.hard_curriculum is not None:
        hard_image_ids = {int(r["image_id"]) for r in json.loads(
            Path(args.hard_curriculum).read_text(encoding="utf-8"))}
        logger.info("E7 困难课程: 加载 %d 个困难目标, %d 张图",
                    len(hard_image_ids), 0)
    dataset_yaml = build_dataset_yaml(split_view, data_root, output_dir, hard_image_ids)
    arguments = _build_train_arguments(config, dataset_yaml)
    _write_resolved_config(config, output_dir)

    if args.dry_run:
        logger.info("dry-run：仅审计，不启动训练")
        _write_train_summary(config, arguments, output_dir, dry_run=True)
        print("TRAIN_DRY_RUN_PASS")
        return 0

    architecture = config["model"].get("architecture")
    model = _load_ultralytics_model(config["model"]["family"], str(weights), architecture)

    innovation = getattr(args, "innovation", "none")
    if innovation == "y3":
        from rsdet.innovation.trainers import hierarchical_trainer

        logger.info("启用 Y3 层次粗类辅助损失 (coarse_gain=%.2f)", args.coarse_gain)
        model.train(trainer=hierarchical_trainer(coarse_gain=args.coarse_gain), **arguments)
    elif innovation == "y4":
        if args.suff_json is None:
            raise ValueError("Y4 需要 --suff-json（afss_diagnose.py 输出）")
        from rsdet.innovation.trainers import afss_trainer

        suff_list = _load_afss_suff_list(args.suff_json, split_view)
        logger.info("启用 Y4 AFSS 采样 (%d 图, easy_floor=%.2f)", len(suff_list), args.easy_floor)
        model.train(trainer=afss_trainer(suff_list, easy_floor=args.easy_floor), **arguments)
    elif innovation == "y5":
        from rsdet.innovation.trainers import rotate90_augmentations

        logger.info("启用 Y5 90° 旋转增强 (p=%.2f)", args.rotate90_p)
        model.train(augmentations=rotate90_augmentations(p=args.rotate90_p), **arguments)
    elif innovation == "coph":
        from rsdet.innovation.coph_presence import coph_trainer

        logger.info(
            "启用 E8 COPH 存在性正则 (presence_gain=%.2f, coarse_gain=%.2f)",
            args.coph_presence_gain,
            args.coph_coarse_gain,
        )
        model.train(
            trainer=coph_trainer(
                coarse_gain=args.coph_coarse_gain,
                presence_gain=args.coph_presence_gain,
            ),
            **arguments,
        )
    elif innovation == "dfd":
        from rsdet.innovation.dfd_presence import dfd_trainer

        only_cls = None
        if args.dfd_only_classes:
            only_cls = tuple(int(x) for x in args.dfd_only_classes.split(","))
        logger.info(
            "启用 DFD 密集前景监督 (dfd_gain=%.2f, sigma_scale=%.2f, "
            "presence_gain=%.2f, only_classes=%s)",
            args.dfd_gain,
            args.dfd_sigma_scale,
            args.coph_presence_gain,
            only_cls,
        )
        model.train(
            trainer=dfd_trainer(
                coarse_gain=args.coph_coarse_gain,
                presence_gain=args.coph_presence_gain,
                dfd_gain=args.dfd_gain,
                dfd_sigma_scale=args.dfd_sigma_scale,
                dfd_focal_alpha=args.dfd_focal_alpha,
                dfd_neg_gamma=args.dfd_neg_gamma,
                only_classes=only_cls,
            ),
            **arguments,
        )
    elif innovation == "family":
        from rsdet.innovation.family_loss import family_trainer

        logger.info(
            "启用 F2 family 三层辅助损失 (coarse_gain=%.2f, family_gain=%.2f)",
            args.coarse_gain,
            args.family_gain,
        )
        model.train(
            trainer=family_trainer(
                coarse_gain=args.coarse_gain,
                family_gain=args.family_gain,
            ),
            **arguments,
        )
    elif innovation == "worstgroup":
        if args.hard_curriculum is None:
            raise ValueError("D4 worstgroup 需要 --hard-curriculum(提供 worst-group image_id 列表)")
        from rsdet.innovation.worst_group_loss import worst_group_trainer

        # 构建 worst-group 的 im_file 绝对路径集合(split_view: image_id -> relative_path)
        hard_ids = hard_image_ids  # 已在前面从 hard_curriculum 读取
        view = json.loads(split_view.read_text(encoding="utf-8"))
        id_to_rel = {int(s["image_id"]): s["relative_path"] for s in view.get("samples", [])}
        hard_paths = {str(data_root / id_to_rel[i]) for i in hard_ids if i in id_to_rel}
        logger.info(
            "启用 D4 worst-group 加权 (hard_paths=%d, wg_gain=%.2f, coarse_gain=%.2f)",
            len(hard_paths),
            args.wg_gain,
            args.coarse_gain,
        )
        model.train(
            trainer=worst_group_trainer(
                hard_paths=hard_paths,
                coarse_gain=args.coarse_gain,
                wg_gain=args.wg_gain,
            ),
            **arguments,
        )
    elif innovation == "v2veh":
        from rsdet.innovation.v2_vehicle_seed import v2_vehicle_seed_trainer

        logger.info(
            "启用 V2 车辆中心-周围对比 (center_gain=%.2f, dfd_gain=%.2f)",
            args.center_gain,
            args.dfd_gain,
        )
        model.train(
            trainer=v2_vehicle_seed_trainer(
                coarse_gain=args.coarse_gain,
                dfd_gain=args.dfd_gain,
                center_gain=args.center_gain,
            ),
            **arguments,
        )
    elif innovation == "confpair":
        from rsdet.innovation.confusion_pair import confusion_pair_trainer

        logger.info(
            "启用 F3 混淆组内判别 (confusion_gain=%.2f, family_gain=%.2f)",
            args.confusion_gain,
            args.family_gain,
        )
        model.train(
            trainer=confusion_pair_trainer(
                coarse_gain=args.coarse_gain,
                family_gain=args.family_gain,
                confusion_gain=args.confusion_gain,
            ),
            **arguments,
        )
    elif innovation == "obs":
        from rsdet.innovation.observability import observability_trainer

        logger.info(
            "启用 F4 可观测性掩码 (confusion_gain=%.2f, family_gain=%.2f)",
            args.confusion_gain,
            args.family_gain,
        )
        model.train(
            trainer=observability_trainer(
                coarse_gain=args.coarse_gain,
                family_gain=args.family_gain,
                confusion_gain=args.confusion_gain,
            ),
            **arguments,
        )
    elif innovation == "tailw":
        from rsdet.innovation.tail_weight import tail_weight_trainer

        logger.info(
            "启用 F6 尾类重加权 (tail_weight=%.2f, family_gain=%.2f)",
            args.tail_weight,
            args.family_gain,
        )
        model.train(
            trainer=tail_weight_trainer(
                coarse_gain=args.coarse_gain,
                family_gain=args.family_gain,
                tail_weight=args.tail_weight,
            ),
            **arguments,
        )
    else:
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
