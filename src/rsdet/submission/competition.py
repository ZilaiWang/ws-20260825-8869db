"""官方 Docker 合同的独立推理实现。

本模块只负责比赛部署：读取 ``/input`` 第一层图片、一次加载模型、逐图执行
大图切片推理，并把结果写为 ``/output/result.json``。训练、评估和实验代码不应
从这里反向依赖。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rsdet.contracts import InferenceSample, Prediction
from rsdet.data.xh_dataset import COARSE_NAMES, FINE_NAMES
from rsdet.models.base import BaseDetector
from rsdet.pipeline.large_image import PipelineConfig, run_pipeline
from rsdet.postprocess.nms import nms
from rsdet.postprocess.thresholds import normalize_fine_thresholds
from rsdet.submission.agreement import apply_label_agreement
from rsdet.submission.aircraft_d4 import (
    AircraftD4ClassifierRuntime,
    filter_prediction_by_score,
)
from rsdet.submission.class_resolution_router import (
    DualPipelineResolutionRuntime,
    PrimaryLabelRescue,
    ResolutionLabelRoute,
)
from rsdet.submission.vehicle_rescue import (
    SelectiveVehicleRescueDetector,
    VehicleRescueConfig,
)

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
DEFAULT_CONFIG_PATH = Path("/app/config.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} 必须是 JSON 对象")
    return dict(value)


def _validate_yolo_model_section(model: Mapping[str, Any], name: str) -> None:
    if str(model.get("family", "")).lower() != "yolo":
        raise ValueError(f"{name}.family 只允许 yolo")
    adapter = str(model.get("inference_adapter", "legacy_tta"))
    if adapter not in {"legacy_tta", "shared_offline"}:
        raise ValueError(f"{name}.inference_adapter 非法")
    if adapter == "shared_offline" and model.get("rot90_views", [0]) != [0]:
        raise ValueError("shared_offline 只允许恒等视图，不能悄悄忽略 TTA")
    weight_path = Path(str(model.get("weight_path", "")))
    if not weight_path.is_absolute():
        raise ValueError(f"{name}.weight_path 必须是容器内绝对路径")
    expected_sha = str(model.get("expected_sha256", "")).strip().lower()
    if len(expected_sha) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha
    ):
        raise ValueError(f"{name}.expected_sha256 必须是 64 位十六进制 SHA256")
    if int(model.get("imgsz", 0)) <= 0:
        raise ValueError(f"{name}.imgsz 必须 > 0")
    if not 0.0 <= float(model.get("confidence", -1.0)) <= 1.0:
        raise ValueError(f"{name}.confidence 必须在 [0, 1]")
    if not 0.0 <= float(model.get("iou", -1.0)) <= 1.0:
        raise ValueError(f"{name}.iou 必须在 [0, 1]")
    if int(model.get("max_detections", 0)) <= 0:
        raise ValueError(f"{name}.max_detections 必须 > 0")
    rot90_views = model.get("rot90_views", [0])
    if (
        not isinstance(rot90_views, list)
        or not rot90_views
        or any(isinstance(value, bool) or not isinstance(value, int) for value in rot90_views)
        or any(value not in {0, 1, 2, 3} for value in rot90_views)
        or len(set(rot90_views)) != len(rot90_views)
    ):
        raise ValueError(f"{name}.rot90_views 必须是 [0,1,2,3] 的非空无重复子集")
    if 0 not in rot90_views:
        raise ValueError(f"{name}.rot90_views 必须包含恒等视图 0")
    if not 0.0 <= float(model.get("tta_nms_iou", 0.55)) <= 1.0:
        raise ValueError(f"{name}.tta_nms_iou 必须在 [0, 1]")


def _validate_pipeline_section(pipeline: Mapping[str, Any], name: str) -> None:
    tile_size = int(pipeline.get("tile_size", 0))
    overlap = int(pipeline.get("overlap", -1))
    if tile_size <= 0 or not 0 <= overlap < tile_size:
        raise ValueError(f"{name} 必须满足 tile_size > overlap >= 0")
    if int(pipeline.get("batch_size", 0)) <= 0:
        raise ValueError(f"{name}.batch_size 必须 > 0")
    if str(pipeline.get("fusion", "")) not in {"tile", "global", "safe"}:
        raise ValueError(f"{name}.fusion 只允许 tile、global 或 safe")
    if not 0.0 <= float(pipeline.get("score_threshold", -1.0)) <= 1.0:
        raise ValueError(f"{name}.score_threshold 必须在 [0, 1]")
    output_threshold = pipeline.get("output_score_threshold")
    owner_slack = pipeline.get("owner_logit_slack")
    safe_category_ids = pipeline.get("threshold_safe_category_ids")
    if output_threshold is not None:
        value = float(output_threshold)
        if (
            str(pipeline.get("fusion")) != "safe"
            or not math.isfinite(value)
            or not float(pipeline["score_threshold"]) <= value <= 1.0
        ):
            raise ValueError(
                f"{name}.output_score_threshold 仅适用于 safe，且必须不低于候选底阈值"
            )
    if owner_slack is not None:
        value = float(owner_slack)
        if output_threshold is None or not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{name}.owner_logit_slack 必须非负且要求 output_score_threshold"
            )
    if safe_category_ids is not None:
        if (
            output_threshold is None
            or not isinstance(safe_category_ids, list)
            or not safe_category_ids
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in safe_category_ids
            )
            or len(set(safe_category_ids)) != len(safe_category_ids)
        ):
            raise ValueError(
                f"{name}.threshold_safe_category_ids 必须是非空、无重复的"
                "非负整数列表，且要求 output_score_threshold"
            )


def load_submission_config(path: str | Path) -> dict[str, Any]:
    """读取并校验部署配置；不允许运行时静默猜测权重或设备。"""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"部署配置不存在: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = _as_mapping(payload, "config")
    if str(config.get("metric_protocol", "platform_observed_20260831")) != (
        "platform_observed_20260831"
    ):
        raise ValueError("Docker 只接受 metric_protocol=platform_observed_20260831")
    model = _as_mapping(config.get("model"), "model")
    pipeline = _as_mapping(config.get("pipeline"), "pipeline")

    _validate_yolo_model_section(model, "model")
    agreement_model = config.get("agreement_model")
    resolution_expert = config.get("resolution_expert_model")
    if agreement_model is not None and resolution_expert is not None:
        raise ValueError("agreement_model 与 resolution_expert_model 不允许同时启用")
    if agreement_model is not None:
        agreement_model = _as_mapping(agreement_model, "agreement_model")
        if str(agreement_model.get("family", "")).lower() != "dfine":
            raise ValueError("agreement_model.family 只允许 dfine")
        for field in ("root_path", "config_path", "weight_path"):
            if not Path(str(agreement_model.get(field, ""))).is_absolute():
                raise ValueError(f"agreement_model.{field} 必须是容器内绝对路径")
        agreement_sha = str(agreement_model.get("expected_sha256", "")).lower()
        if len(agreement_sha) != 64 or any(
            character not in "0123456789abcdef" for character in agreement_sha
        ):
            raise ValueError("agreement_model.expected_sha256 必须是 64 位十六进制 SHA256")
        if int(agreement_model.get("imgsz", 0)) <= 0:
            raise ValueError("agreement_model.imgsz 必须 > 0")
        if not 0.0 <= float(agreement_model.get("score_floor", -1.0)) <= 1.0:
            raise ValueError("agreement_model.score_floor 必须在 [0, 1]")
        if not 0.0 <= float(agreement_model.get("support_iou", -1.0)) <= 1.0:
            raise ValueError("agreement_model.support_iou 必须在 [0, 1]")
        raw_labels = agreement_model.get("apply_labels", [24])
        if not isinstance(raw_labels, list) or not raw_labels:
            raise ValueError("agreement_model.apply_labels 必须是非空整数列表")
        if any(
            isinstance(label, bool)
            or not isinstance(label, int)
            or not 0 <= label < len(FINE_NAMES)
            for label in raw_labels
        ):
            raise ValueError("agreement_model.apply_labels 包含非法类别")
        if len(raw_labels) != len(set(raw_labels)):
            raise ValueError("agreement_model.apply_labels 不允许重复")
        agreement_mode = str(agreement_model.get("mode", "multiplicative"))
        if agreement_mode not in {"multiplicative", "vehicle_reject_rescue"}:
            raise ValueError("agreement_model.mode 只允许 multiplicative 或 vehicle_reject_rescue")
        if agreement_mode == "vehicle_reject_rescue":
            VehicleRescueConfig(
                vehicle_label=int(agreement_model.get("vehicle_label", 24)),
                core_threshold=float(agreement_model["core_threshold"]),
                candidate_floor=float(agreement_model["candidate_floor"]),
                support_iou=float(agreement_model["support_iou"]),
                rescue_product_threshold=float(agreement_model["rescue_product_threshold"]),
                promoted_score=float(agreement_model["promoted_score"]),
            )

    _validate_pipeline_section(pipeline, "pipeline")
    post_threshold = config.get("post_fusion_score_threshold")
    if post_threshold is not None:
        value = float(post_threshold)
        if not math.isfinite(value) or not float(pipeline["score_threshold"]) <= value <= 1:
            raise ValueError("post_fusion_score_threshold 必须有限且不低于候选底阈值")
        if resolution_expert is not None or any(
            pipeline.get(k) is not None
            for k in ("score_threshold_by_coarse", "score_threshold_by_fine")
        ):
            raise ValueError("融合后统一阈值不能混合分辨率路由或类别阈值")
        output_threshold = pipeline.get("output_score_threshold")
        if output_threshold is not None and not math.isclose(
            float(output_threshold), value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "BATIS output_score_threshold 必须与 post_fusion_score_threshold 相等"
            )
    coarse_thresholds = pipeline.get("score_threshold_by_coarse")
    if coarse_thresholds is not None:
        coarse_thresholds = _as_mapping(coarse_thresholds, "pipeline.score_threshold_by_coarse")
        if set(coarse_thresholds) != set(COARSE_NAMES):
            raise ValueError(
                "pipeline.score_threshold_by_coarse 必须恰好覆盖 ship/aircraft/vehicle"
            )
        for name, value in coarse_thresholds.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"pipeline.score_threshold_by_coarse.{name} 必须在 [0, 1]")
    fine_thresholds = pipeline.get("score_threshold_by_fine")
    if fine_thresholds is not None:
        normalize_fine_thresholds(
            _as_mapping(fine_thresholds, "pipeline.score_threshold_by_fine"),
            require_complete=True,
        )
    if resolution_expert is not None:
        expert = _as_mapping(resolution_expert, "resolution_expert_model")
        _validate_yolo_model_section(expert, "resolution_expert_model")
        expert_pipeline = _as_mapping(
            config.get("resolution_expert_pipeline"), "resolution_expert_pipeline"
        )
        _validate_pipeline_section(expert_pipeline, "resolution_expert_pipeline")
        route = _as_mapping(config.get("resolution_route"), "resolution_route")
        parsed_route = ResolutionLabelRoute(
            primary_labels=frozenset(route.get("primary_labels", [])),
            expert_labels=frozenset(route.get("expert_labels", [])),
            primary_threshold=float(route.get("primary_threshold", -1.0)),
            expert_threshold=float(route.get("expert_threshold", -1.0)),
        )
        rescue_raw = config.get("resolution_primary_rescue")
        if rescue_raw is not None:
            rescue = _as_mapping(rescue_raw, "resolution_primary_rescue")
            parsed_rescue = PrimaryLabelRescue(
                labels=frozenset(rescue.get("labels", [])),
                threshold=float(rescue.get("threshold", -1.0)),
                dedup_iou=float(rescue.get("dedup_iou", -1.0)),
            )
            if not parsed_rescue.labels <= parsed_route.expert_labels:
                raise ValueError(
                    "resolution_primary_rescue.labels 必须属于 resolution_route.expert_labels"
                )
            if float(pipeline["score_threshold"]) > parsed_rescue.threshold:
                raise ValueError(
                    "pipeline.score_threshold 不能高于 primary rescue threshold"
                )
        for section_name, section, threshold in (
            ("pipeline", pipeline, parsed_route.primary_threshold),
            (
                "resolution_expert_pipeline",
                expert_pipeline,
                parsed_route.expert_threshold,
            ),
        ):
            if (
                section.get("score_threshold_by_coarse") is not None
                or section.get("score_threshold_by_fine") is not None
            ):
                raise ValueError(
                    f"{section_name} 的类别阈值必须关闭；分支阈值只能由 resolution_route 在融合后执行"
                )
            if float(section["score_threshold"]) > threshold:
                raise ValueError(
                    f"{section_name}.score_threshold 不能高于对应的 resolution_route 融合后阈值"
                )
    elif config.get("resolution_primary_rescue") is not None:
        raise ValueError("resolution_primary_rescue 只能与 resolution_expert_model 一起启用")
    aircraft_classifier = config.get("aircraft_classifier_model")
    if aircraft_classifier is not None:
        aircraft_classifier = _as_mapping(
            aircraft_classifier, "aircraft_classifier_model"
        )
        full_state = bool(aircraft_classifier.get("checkpoint_contains_full_state", False))
        path_fields = ["weight_path"]
        sha_fields = ["expected_sha256"]
        if not full_state:
            path_fields.append("imagenet_weight_path")
            sha_fields.append("imagenet_expected_sha256")
        for field in path_fields:
            if not Path(str(aircraft_classifier.get(field, ""))).is_absolute():
                raise ValueError(f"aircraft_classifier_model.{field} 必须是容器内绝对路径")
        for field in sha_fields:
            value = str(aircraft_classifier.get(field, "")).lower()
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"aircraft_classifier_model.{field} 必须是 SHA256")
        if str(aircraft_classifier.get("method", "")) != "view_consistency_d4":
            raise ValueError("aircraft_classifier_model.method 只允许 view_consistency_d4")
        if int(aircraft_classifier.get("batch_objects", 0)) <= 0:
            raise ValueError("aircraft_classifier_model.batch_objects 必须 > 0")
        for field in ("channels_last", "tensorized_views"):
            value = aircraft_classifier.get(field, False)
            if not isinstance(value, bool):
                raise ValueError(f"aircraft_classifier_model.{field} 必须是布尔值")
        for field in ("relabel_min_probability", "nms_iou"):
            if not 0.0 <= float(aircraft_classifier.get(field, -1.0)) <= 1.0:
                raise ValueError(f"aircraft_classifier_model.{field} 必须在 [0, 1]")
    return config


def check_gpu(device: str) -> None:
    """启动时硬检查 CUDA，避免评测时悄悄退回 CPU。"""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；赛事镜像禁止回退到 CPU 推理")
    if not device.startswith("cuda"):
        raise ValueError(f"部署 device 必须是 CUDA 设备，当前为 {device!r}")
    index = 0
    if ":" in device:
        index = int(device.split(":", 1)[1])
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA 设备 {device} 不存在，共 {torch.cuda.device_count()} 张卡")
    torch.cuda.set_device(index)
    print(
        f"[submission] CUDA ready: device={device} "
        f"name={torch.cuda.get_device_name(index)} torch={torch.__version__}",
        flush=True,
    )


class _SubmissionYoloDetector(BaseDetector):
    """部署专用的批量 YOLO 适配器。

    PIL 被统一转为 RGB，而 Ultralytics 对 numpy 输入采用 BGR 约定，因此此处显式
    反转通道。该行为与训练/文件路径推理一致，不能省略。
    """

    def __init__(self, model_config: Mapping[str, Any], device: str) -> None:
        self.config = dict(model_config)
        self.device = device
        self.model: Any | None = None

    def load(self, checkpoint_path: str) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("镜像缺少 ultralytics") from error
        self.model = YOLO(checkpoint_path)
        raw_names = getattr(self.model, "names", {})
        names = tuple(str(raw_names[index]) for index in range(len(raw_names)))
        if names != FINE_NAMES:
            raise RuntimeError(
                f"权重类别表与赛事 25 类合同不一致: expected={FINE_NAMES!r}, actual={names!r}"
            )

    def to(self, device: str) -> None:
        self.device = device
        if self.model is not None:
            self.model.to(device)

    def eval(self) -> None:
        if self.model is None:
            return
        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "eval"):
            inner.eval()

    @staticmethod
    def _invert_rot90_box(
        box: Sequence[float], rotation: int, width: int, height: int
    ) -> list[float]:
        """Map an xyxy box from ``np.rot90(image, rotation)`` back to the source."""
        x1, y1, x2, y2 = (float(value) for value in box)
        rotation %= 4
        if rotation == 0:
            restored = [x1, y1, x2, y2]
        elif rotation == 1:
            restored = [width - y2, x1, width - y1, x2]
        elif rotation == 2:
            restored = [width - x2, height - y2, width - x1, height - y1]
        else:
            restored = [y1, height - x2, y2, height - x1]
        restored[0] = min(max(restored[0], 0.0), float(width))
        restored[2] = min(max(restored[2], 0.0), float(width))
        restored[1] = min(max(restored[1], 0.0), float(height))
        restored[3] = min(max(restored[3], 0.0), float(height))
        return restored

    def _predict_view(self, batch: Sequence[InferenceSample], rotation: int) -> list[Any]:
        if self.model is None:
            raise RuntimeError("YOLO 权重尚未加载")
        sources = []
        for sample in batch:
            image = np.asarray(sample.image, dtype=np.uint8)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"YOLO 输入必须是 HWC RGB，实际 {image.shape}")
            rotated = np.rot90(image, rotation, axes=(0, 1))
            sources.append(np.ascontiguousarray(rotated[..., ::-1]))
        return list(
            self.model.predict(
                source=sources,
                imgsz=int(self.config["imgsz"]),
                conf=float(self.config["confidence"]),
                iou=float(self.config["iou"]),
                max_det=int(self.config["max_detections"]),
                device=self.device,
                half=bool(self.config.get("half", True)),
                verbose=False,
                stream=False,
            )
        )

    def predict(self, batch: Sequence[InferenceSample]) -> list[Prediction]:
        if self.model is None:
            raise RuntimeError("YOLO 权重尚未加载")
        if not batch:
            return []
        rotations = tuple(int(value) for value in self.config.get("rot90_views", [0]))
        accumulated = [([], [], []) for _ in batch]
        for rotation in rotations:
            results = self._predict_view(batch, rotation)
            if len(results) != len(batch):
                raise RuntimeError(
                    f"Ultralytics 返回 {len(results)} 张结果，但输入 batch 为 {len(batch)}"
                )
            for index, (sample, result) in enumerate(zip(batch, results)):
                boxes = getattr(result, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    continue
                raw_xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float64).tolist()
                scores = boxes.conf.detach().cpu().numpy().astype(np.float64).tolist()
                labels = boxes.cls.detach().cpu().numpy().astype(np.int64).tolist()
                if len(boxes) >= int(self.config["max_detections"]):
                    print(
                        "[submission][warning] tile reached max_det: "
                        f"tile_id={sample.image_id} rotation={rotation} count={len(boxes)} "
                        f"limit={self.config['max_detections']}",
                        flush=True,
                    )
                height, width = np.asarray(sample.image).shape[:2]
                restored = [
                    self._invert_rot90_box(box, rotation, width, height) for box in raw_xyxy
                ]
                out_boxes, out_scores, out_labels = accumulated[index]
                out_boxes.extend(restored)
                out_scores.extend(scores)
                out_labels.extend(labels)

        outputs: list[Prediction] = []
        tta_nms_iou = float(self.config.get("tta_nms_iou", 0.55))
        limit = int(self.config["max_detections"])
        for sample, (boxes, scores, labels) in zip(batch, accumulated):
            keep: list[int] = []
            for label in sorted(set(labels)):
                indices = [index for index, value in enumerate(labels) if value == label]
                local = nms(
                    [boxes[index] for index in indices],
                    [scores[index] for index in indices],
                    tta_nms_iou,
                )
                keep.extend(indices[index] for index in local)
            keep.sort(key=lambda index: (-float(scores[index]), index))
            keep = keep[:limit]
            outputs.append(
                Prediction(
                    sample.image_id,
                    [boxes[index] for index in keep],
                    [scores[index] for index in keep],
                    [labels[index] for index in keep],
                )
            )
        return outputs


class _SubmissionDfineDetector(BaseDetector):
    """Deployment-only D-FINE adapter with the frozen square-resize contract."""

    def __init__(self, model_config: Mapping[str, Any], device: str) -> None:
        self.config = dict(model_config)
        self.device = device
        self.model: Any | None = None

    def load(self, checkpoint_path: str) -> None:
        import torch
        from torch import nn

        root = str(self.config["root_path"])
        if root not in sys.path:
            sys.path.insert(0, root)
        try:
            from src.core import YAMLConfig
        except ImportError as error:
            raise RuntimeError("镜像缺少冻结 D-FINE 源码") from error

        cfg = YAMLConfig(str(self.config["config_path"]), resume=checkpoint_path)
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        expected_epoch = int(self.config.get("expected_checkpoint_epoch", 39))
        if int(checkpoint.get("last_epoch", -1)) != expected_epoch:
            raise RuntimeError(
                "D-FINE checkpoint epoch 不匹配: "
                f"expected={expected_epoch}, actual={checkpoint.get('last_epoch')!r}"
            )
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        cfg.model.load_state_dict(state)

        class DeployModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images: Any, sizes: Any) -> Any:
                return self.postprocessor(self.inner(images), sizes)

        self.model = DeployModel()

    def to(self, device: str) -> None:
        self.device = device
        if self.model is not None:
            self.model.to(device)

    def eval(self) -> None:
        if self.model is not None:
            self.model.eval()

    def predict(self, batch: Sequence[InferenceSample]) -> list[Prediction]:
        if self.model is None:
            raise RuntimeError("D-FINE 权重尚未加载")
        if not batch:
            return []
        import torch
        from torchvision.transforms import functional as tvf

        tensors = []
        sizes = []
        image_size = int(self.config["imgsz"])
        for sample in batch:
            image = np.asarray(sample.image, dtype=np.uint8)
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"D-FINE 输入必须是 HWC RGB，实际 {image.shape}")
            # Keep this byte-for-byte preprocessing contract aligned with
            # scripts/infer_dfine_coco.py, which generated the evidence used
            # to freeze the product threshold.  Tensor interpolation (even
            # with bilinear mode) is not numerically identical to PIL resize.
            resized = tvf.resize(Image.fromarray(image), [image_size, image_size])
            tensors.append(tvf.to_tensor(resized))
            sizes.append([sample.width, sample.height])
        images = torch.stack(tensors).to(self.device, non_blocking=True)
        original_sizes = torch.tensor(sizes, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            labels, boxes, scores = self.model(images, original_sizes)
        floor = float(self.config["score_floor"])
        outputs: list[Prediction] = []
        for sample, item_labels, item_boxes, item_scores in zip(
            batch, labels, boxes, scores, strict=True
        ):
            keep = item_scores >= floor
            output_boxes = item_boxes[keep].detach().cpu().float().tolist()
            output_scores = item_scores[keep].detach().cpu().float().tolist()
            output_labels = item_labels[keep].detach().cpu().long().tolist()
            clipped = [
                [
                    min(max(float(box[0]), 0.0), float(sample.width)),
                    min(max(float(box[1]), 0.0), float(sample.height)),
                    min(max(float(box[2]), 0.0), float(sample.width)),
                    min(max(float(box[3]), 0.0), float(sample.height)),
                ]
                for box in output_boxes
            ]
            outputs.append(
                Prediction(
                    image_id=sample.image_id,
                    boxes_xyxy=clipped,
                    scores=output_scores,
                    labels=output_labels,
                )
            )
        return outputs


class _SubmissionAgreementDetector(BaseDetector):
    """Run both detectors but retain only Y5 geometry and fine labels."""

    def __init__(
        self,
        primary: BaseDetector,
        specialist: BaseDetector,
        *,
        support_iou: float,
        apply_labels: Sequence[int],
    ) -> None:
        self.primary = primary
        self.specialist = specialist
        self.support_iou = support_iou
        self.apply_labels = tuple(int(label) for label in apply_labels)

    def load(self, checkpoint_path: str) -> None:
        del checkpoint_path

    def to(self, device: str) -> None:
        self.primary.to(device)
        self.specialist.to(device)

    def eval(self) -> None:
        self.primary.eval()
        self.specialist.eval()

    def predict(self, batch: Sequence[InferenceSample]) -> list[Prediction]:
        primary = self.primary.predict(batch)
        specialist = self.specialist.predict(batch)
        if len(primary) != len(specialist):
            raise RuntimeError("Y5 与 D-FINE batch 输出数量不一致")
        return [
            apply_label_agreement(
                first,
                second,
                labels=self.apply_labels,
                support_iou=self.support_iou,
            )
            for first, second in zip(primary, specialist, strict=True)
        ]


def _pipeline_config_from_mapping(pipeline: Mapping[str, Any], name: str) -> PipelineConfig:
    return PipelineConfig(
        tile_size=int(pipeline["tile_size"]),
        overlap=int(pipeline["overlap"]),
        batch_size=int(pipeline["batch_size"]),
        score_threshold=float(pipeline["score_threshold"]),
        score_threshold_by_coarse=(
            None
            if pipeline.get("score_threshold_by_coarse") is None
            else {
                str(key): float(value)
                for key, value in _as_mapping(
                    pipeline["score_threshold_by_coarse"],
                    f"{name}.score_threshold_by_coarse",
                ).items()
            }
        ),
        score_threshold_by_fine=normalize_fine_thresholds(
            None
            if pipeline.get("score_threshold_by_fine") is None
            else _as_mapping(
                pipeline["score_threshold_by_fine"],
                f"{name}.score_threshold_by_fine",
            ),
            require_complete=pipeline.get("score_threshold_by_fine") is not None,
        ),
        fine_nms_iou=float(pipeline.get("fine_nms_iou", 0.55)),
        coarse_nms_iou=(
            None
            if pipeline.get("coarse_nms_iou") is None
            else float(pipeline.get("coarse_nms_iou", 0.85))
        ),
        max_detections=int(pipeline.get("max_detections", 2000)),
        fusion=str(pipeline["fusion"]),
        cluster_eps=float(pipeline.get("cluster_eps", 50.0)),
        merge_iou=float(pipeline.get("merge_iou", 0.3)),
        nms_iou=float(pipeline.get("nms_iou", 0.5)),
        merge_ios=float(pipeline.get("merge_ios", 0.75)),
        border_margin=float(pipeline.get("border_margin", 8.0)),
        output_score_threshold=(
            None
            if pipeline.get("output_score_threshold") is None
            else float(pipeline["output_score_threshold"])
        ),
        owner_logit_slack=(
            None
            if pipeline.get("owner_logit_slack") is None
            else float(pipeline["owner_logit_slack"])
        ),
        threshold_safe_category_ids=(
            None
            if pipeline.get("threshold_safe_category_ids") is None
            else tuple(int(value) for value in pipeline["threshold_safe_category_ids"])
        ),
    )


class CompetitionDetector:
    """一次加载权重、逐图返回官方 objects 列表。"""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.device = str(self.config.get("device", "cuda:0"))
        check_gpu(self.device)
        model_config = _as_mapping(self.config["model"], "model")
        weight_path = Path(str(model_config["weight_path"]))
        if not weight_path.is_file():
            raise FileNotFoundError(f"模型权重不存在: {weight_path}")
        actual_sha = _sha256(weight_path)
        expected_sha = str(model_config["expected_sha256"]).lower()
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"模型权重 SHA256 不匹配: expected={expected_sha}, actual={actual_sha}"
            )
        if model_config.get("inference_adapter") == "shared_offline":
            # Opt-in only: reproduce the candidate generator that made the
            # frozen offline ledger. In particular, do not append legacy TTA NMS.
            from rsdet.models.ultralytics_adapter import UltralyticsDetector

            primary = UltralyticsDetector(
                family="yolo", imgsz=int(model_config["imgsz"]),
                confidence=float(model_config["confidence"]),
                iou=float(model_config["iou"]),
                max_detections=int(model_config["max_detections"]),
                half=bool(model_config.get("half", True)), agnostic_nms=False,
            )
            primary.load(str(weight_path))
            raw_names = getattr(primary._model, "names", {})
            if tuple(str(raw_names[i]) for i in range(len(raw_names))) != FINE_NAMES:
                raise RuntimeError("权重类别表与赛事 25 类合同不一致")
        else:
            primary = _SubmissionYoloDetector(model_config, self.device)
            primary.load(str(weight_path))
        agreement_config_raw = self.config.get("agreement_model")
        if agreement_config_raw is None:
            self.detector = primary
        else:
            agreement_config = _as_mapping(agreement_config_raw, "agreement_model")
            agreement_weight = Path(str(agreement_config["weight_path"]))
            agreement_root = Path(str(agreement_config["root_path"]))
            agreement_model_config = Path(str(agreement_config["config_path"]))
            for path, name in (
                (agreement_weight, "weight_path"),
                (agreement_root, "root_path"),
                (agreement_model_config, "config_path"),
            ):
                if not path.exists():
                    raise FileNotFoundError(f"agreement_model.{name} 不存在: {path}")
            agreement_actual_sha = _sha256(agreement_weight)
            agreement_expected_sha = str(agreement_config["expected_sha256"]).lower()
            if agreement_actual_sha != agreement_expected_sha:
                raise RuntimeError(
                    "D-FINE 权重 SHA256 不匹配: "
                    f"expected={agreement_expected_sha}, actual={agreement_actual_sha}"
                )
            specialist = _SubmissionDfineDetector(agreement_config, self.device)
            specialist.load(str(agreement_weight))
            if str(agreement_config.get("mode", "multiplicative")) == "vehicle_reject_rescue":
                self.detector = SelectiveVehicleRescueDetector(
                    primary,
                    specialist,
                    config=VehicleRescueConfig(
                        vehicle_label=int(agreement_config.get("vehicle_label", 24)),
                        core_threshold=float(agreement_config["core_threshold"]),
                        candidate_floor=float(agreement_config["candidate_floor"]),
                        support_iou=float(agreement_config["support_iou"]),
                        rescue_product_threshold=float(
                            agreement_config["rescue_product_threshold"]
                        ),
                        promoted_score=float(agreement_config["promoted_score"]),
                    ),
                )
            else:
                self.detector = _SubmissionAgreementDetector(
                    primary,
                    specialist,
                    support_iou=float(agreement_config["support_iou"]),
                    apply_labels=[
                        int(label) for label in agreement_config.get("apply_labels", [24])
                    ],
                )
        self.resolution_runtime: DualPipelineResolutionRuntime | None = None
        pipeline = _as_mapping(self.config["pipeline"], "pipeline")
        self.pipeline_config = _pipeline_config_from_mapping(pipeline, "pipeline")
        resolution_config_raw = self.config.get("resolution_expert_model")
        if resolution_config_raw is None:
            self.detector.to(self.device)
            self.detector.eval()
        else:
            expert_config = _as_mapping(resolution_config_raw, "resolution_expert_model")
            expert_weight = Path(str(expert_config["weight_path"]))
            if not expert_weight.is_file():
                raise FileNotFoundError(
                    f"resolution_expert_model.weight_path 不存在: {expert_weight}"
                )
            expert_sha = _sha256(expert_weight)
            if expert_sha != str(expert_config["expected_sha256"]).lower():
                raise RuntimeError("resolution expert 权重 SHA256 不匹配")
            expert = _SubmissionYoloDetector(expert_config, self.device)
            expert.load(str(expert_weight))
            expert_pipeline_raw = _as_mapping(
                self.config["resolution_expert_pipeline"],
                "resolution_expert_pipeline",
            )
            route_raw = _as_mapping(self.config["resolution_route"], "resolution_route")
            rescue_raw = self.config.get("resolution_primary_rescue")
            primary_rescue = None
            if rescue_raw is not None:
                rescue_config = _as_mapping(rescue_raw, "resolution_primary_rescue")
                primary_rescue = PrimaryLabelRescue(
                    labels=frozenset(rescue_config["labels"]),
                    threshold=float(rescue_config["threshold"]),
                    dedup_iou=float(rescue_config["dedup_iou"]),
                )
            self.resolution_runtime = DualPipelineResolutionRuntime(
                primary,
                expert,
                primary_pipeline=self.pipeline_config,
                expert_pipeline=_pipeline_config_from_mapping(
                    expert_pipeline_raw, "resolution_expert_pipeline"
                ),
                route=ResolutionLabelRoute(
                    primary_labels=frozenset(route_raw["primary_labels"]),
                    expert_labels=frozenset(route_raw["expert_labels"]),
                    primary_threshold=float(route_raw["primary_threshold"]),
                    expert_threshold=float(route_raw["expert_threshold"]),
                ),
                primary_rescue=primary_rescue,
            )
            self.resolution_runtime.to(self.device)
            self.resolution_runtime.eval()
        if int(pipeline["tile_size"]) > int(model_config["imgsz"]):
            print(
                "[submission][warning] tile is downscaled before inference: "
                f"tile_size={pipeline['tile_size']} imgsz={model_config['imgsz']}",
                flush=True,
            )
        self.aircraft_classifier: AircraftD4ClassifierRuntime | None = None
        aircraft_classifier_raw = self.config.get("aircraft_classifier_model")
        if aircraft_classifier_raw is not None:
            aircraft_classifier = _as_mapping(
                aircraft_classifier_raw, "aircraft_classifier_model"
            )
            asset_fields = [("weight_path", "expected_sha256")]
            if not bool(
                aircraft_classifier.get("checkpoint_contains_full_state", False)
            ):
                asset_fields.append(
                    ("imagenet_weight_path", "imagenet_expected_sha256")
                )
            for path_field, sha_field in asset_fields:
                path = Path(str(aircraft_classifier[path_field]))
                if not path.is_file():
                    raise FileNotFoundError(f"aircraft classifier asset missing: {path}")
                actual = _sha256(path)
                if actual != str(aircraft_classifier[sha_field]).lower():
                    raise RuntimeError(f"aircraft classifier asset SHA mismatch: {path}")
            self.aircraft_classifier = AircraftD4ClassifierRuntime(
                aircraft_classifier, self.device
            )
        print(
            f"[submission] model loaded: {weight_path.name} sha256={actual_sha} "
            f"workpoint={self.config.get('workpoint_id', 'unspecified')}",
            flush=True,
        )

    def predict(self, image: Image.Image) -> list[dict[str, Any]]:
        rgb = np.asarray(image, dtype=np.uint8).copy()
        if self.resolution_runtime is None:
            prediction, _ = run_pipeline(
                rgb,
                self.detector,
                config=self.pipeline_config,
                parent_image_id=0,
            )
        else:
            prediction, _ = self.resolution_runtime.predict_image(rgb, parent_image_id=0)
        aircraft_classifier = getattr(self, "aircraft_classifier", None)
        if aircraft_classifier is not None:
            # D4 never changes scores and its same-class NMS is score-descending.
            # Filtering the final workpoint first is therefore output-equivalent,
            # while avoiding crop/classification work for rejected proposals.
            post_threshold = self.config.get("post_fusion_score_threshold")
            if post_threshold is not None:
                prediction = filter_prediction_by_score(
                    prediction, float(post_threshold)
                )
            prediction = aircraft_classifier.refine(rgb, prediction)
        objects: list[dict[str, Any]] = []
        order = sorted(
            range(len(prediction.scores)),
            key=lambda index: (-float(prediction.scores[index]), index),
        )
        width, height = image.size
        for index in order:
            label = int(prediction.labels[index])
            score = float(prediction.scores[index])
            box = [float(value) for value in prediction.boxes_xyxy[index]]
            if not 0 <= label < len(FINE_NAMES):
                raise ValueError(f"模型输出非法 category_id={label}")
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"模型输出非法 score={score}")
            post_threshold = self.config.get("post_fusion_score_threshold")
            if post_threshold is not None and score < float(post_threshold):
                continue
            if len(box) != 4 or not all(math.isfinite(value) for value in box):
                raise ValueError(f"模型输出非法 bbox={box}")
            x1 = max(0.0, min(box[0], float(width)))
            y1 = max(0.0, min(box[1], float(height)))
            x2 = max(0.0, min(box[2], float(width)))
            y2 = max(0.0, min(box[3], float(height)))
            if x2 <= x1 or y2 <= y1:
                continue
            objects.append(
                {
                    "category_id": label,
                    "category_name": FINE_NAMES[label],
                    "score": score,
                    "bbox": [x1, y1, x2, y2],
                }
            )
        return objects


def discover_images(input_dir: str | Path) -> list[Path]:
    """只发现输入目录第一层、官方允许后缀的普通文件。"""
    root = Path(input_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"输入目录不存在: {root}")
    paths = sorted(
        path
        for path in root.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_EXTS
    )
    stems = [path.stem for path in paths]
    if len(stems) != len(set(stems)):
        raise ValueError("输入目录中存在文件主名重复，无法生成唯一 image_id")
    return paths


def validate_result_payload(payload: Mapping[str, Any]) -> dict[str, int]:
    """严格校验官方 result.json；也供本地提交前脚本复用。"""
    if payload.get("status") != "success":
        raise ValueError("result.status 必须为 success")
    images = payload.get("images")
    if not isinstance(images, list):
        raise ValueError("result.images 必须是列表")
    seen: set[str] = set()
    object_count = 0
    for image_index, item in enumerate(images):
        image = _as_mapping(item, f"images[{image_index}]")
        image_id = str(image.get("image_id", ""))
        if not image_id or image_id in seen:
            raise ValueError(f"images[{image_index}].image_id 为空或重复")
        seen.add(image_id)
        if not str(image.get("file_name", "")):
            raise ValueError(f"images[{image_index}].file_name 不能为空")
        if int(image.get("width", 0)) <= 0 or int(image.get("height", 0)) <= 0:
            raise ValueError(f"images[{image_index}] 尺寸非法")
        if int(image.get("run_end_timestamp", 0)) <= 0:
            raise ValueError(f"images[{image_index}].run_end_timestamp 非法")
        objects = image.get("objects")
        if not isinstance(objects, list):
            raise ValueError(f"images[{image_index}].objects 必须是列表")
        for object_index, raw in enumerate(objects):
            obj = _as_mapping(raw, f"images[{image_index}].objects[{object_index}]")
            category_id = obj.get("category_id")
            if isinstance(category_id, bool) or not isinstance(category_id, int):
                raise ValueError("category_id 必须是整数")
            if not 0 <= category_id < len(FINE_NAMES):
                raise ValueError(f"category_id 越界: {category_id}")
            if obj.get("category_name") != FINE_NAMES[category_id]:
                raise ValueError("category_name 与 category_id 不一致")
            score = float(obj.get("score", -1.0))
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("score 必须是 [0,1] 内有限数")
            bbox = obj.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("bbox 必须是四元素列表")
            box = [float(value) for value in bbox]
            if not all(math.isfinite(value) for value in box):
                raise ValueError("bbox 必须全为有限数")
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("bbox 必须满足 x2>x1 且 y2>y1")
            if box[0] < 0 or box[1] < 0 or box[2] > image["width"] or box[3] > image["height"]:
                raise ValueError("bbox 超出原图边界")
            object_count += 1
    return {"images": len(images), "objects": object_count}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_submission(
    input_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    *,
    detector_factory: Any = CompetitionDetector,
) -> dict[str, Any]:
    """执行完整官方输入输出流程。``detector_factory`` 只用于无 GPU 单测。"""
    config = load_submission_config(config_path)
    detector = detector_factory(config)
    paths = discover_images(input_dir)

    # 对齐官方模板：推理开始前先将输入全部解码到内存，防止读盘混入逐图运行阶段。
    images: list[tuple[Path, Image.Image]] = []
    for path in paths:
        with Image.open(path) as source:
            images.append((path, source.convert("RGB").copy()))

    final_results: list[dict[str, Any]] = []
    for path, image in images:
        objects = detector.predict(image)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
        run_end_timestamp = time.time_ns() // 1_000_000
        final_results.append(
            {
                "image_id": path.stem,
                "file_name": path.name,
                "width": image.width,
                "height": image.height,
                "run_end_timestamp": run_end_timestamp,
                "objects": objects,
            }
        )
        print(
            f"[submission] finished {path.name}: objects={len(objects)} "
            f"timestamp={run_end_timestamp}",
            flush=True,
        )

    payload: dict[str, Any] = {"status": "success", "images": final_results}
    validate_result_payload(payload)
    _write_json_atomic(Path(output_dir) / "result.json", payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XH-202625 赛事目标检测 Docker 入口")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(os.environ.get("RSDET_SUBMISSION_CONFIG", str(DEFAULT_CONFIG_PATH)))
    run_submission(args.input, args.output, config_path)
    return 0


__all__ = [
    "CompetitionDetector",
    "IMAGE_EXTS",
    "discover_images",
    "load_submission_config",
    "main",
    "run_submission",
    "validate_result_payload",
]
