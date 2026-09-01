"""Ultralytics YOLO/RT-DETR 的统一推理适配器。

依赖采用延迟导入，因此没有安装 PyTorch/Ultralytics 时，数据、评测和单元测试
仍可正常使用。只有真正构建或加载该检测器时才要求模型环境。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rsdet.contracts import InferenceSample, Prediction
from rsdet.data.xh_dataset import coarse_name
from rsdet.models.base import BaseDetector
from rsdet.models.registry import register_model


def coarse_purity_sqrt_scores(
    scores: Sequence[float], labels: Sequence[int], class_probabilities: np.ndarray
) -> list[float]:
    """Combine fine confidence with same-coarse probability purity.

    Aircraft is an explicit identity bypass because its formal metrics are
    already strong. Ship and vehicle receive the fixed geometric mean of the
    deployed fine score and the probability mass assigned to their coarse
    family relative to all 25 classes.
    """
    probabilities = np.asarray(class_probabilities, dtype=np.float64)
    if probabilities.shape != (len(scores), 25):
        raise ValueError("class_probabilities must have shape (detections, 25)")
    if len(labels) != len(scores):
        raise ValueError("scores and labels must have equal length")
    if any(label < 0 or label >= 25 for label in labels):
        raise ValueError("labels must be fine category ids in [0, 24]")
    mass = np.stack(
        [
            probabilities[:, 0:4].sum(axis=1),
            probabilities[:, 4:24].sum(axis=1),
            probabilities[:, 24],
        ],
        axis=1,
    )
    total = mass.sum(axis=1).clip(min=1e-12)
    transformed: list[float] = []
    for index, (score, label) in enumerate(zip(scores, labels, strict=True)):
        coarse_index = 0 if 0 <= label <= 3 else 1 if 4 <= label <= 23 else 2
        if coarse_index == 1:
            transformed.append(float(score))
            continue
        purity = float(np.clip(mass[index, coarse_index] / total[index], 0.0, 1.0))
        transformed.append(math.sqrt(max(float(score), 0.0) * purity))
    return transformed


def create_ultralytics_model(family: str, weights: str | Path):
    """按模型族创建 Ultralytics 模型，集中处理可选依赖错误。"""
    try:
        from ultralytics import RTDETR, YOLO
    except ImportError as error:
        raise ImportError('Ultralytics 模型依赖未安装。请执行 pip install -e ".[model]"') from error

    normalized = family.strip().lower().replace("-", "")
    if normalized == "yolo":
        return YOLO(str(weights))
    if normalized in {"rtdetr", "detr"}:
        return RTDETR(str(weights))
    raise ValueError(f"不支持的 Ultralytics 模型族: {family!r}")


@register_model("ultralytics")
class UltralyticsDetector(BaseDetector):
    """把 Ultralytics 结果转换为项目统一 ``Prediction``。"""

    def __init__(
        self,
        *,
        family: str = "yolo",
        imgsz: int = 1024,
        confidence: float = 0.05,
        iou: float = 0.6,
        max_detections: int = 500,
        half: bool = True,
        agnostic_nms: bool = False,
        label_map: Mapping[int, int] | None = None,
        refiner: Mapping[str, Any] | None = None,
        agreement: Mapping[str, Any] | None = None,
        score_transform: str | None = None,
    ) -> None:
        if imgsz <= 0:
            raise ValueError("imgsz 必须 > 0")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence 必须在 [0, 1]")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou 必须在 [0, 1]")
        if max_detections <= 0:
            raise ValueError("max_detections 必须 > 0")
        self.family = family
        self.imgsz = int(imgsz)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.max_detections = int(max_detections)
        self.half = bool(half)
        self.agnostic_nms = bool(agnostic_nms)
        self.label_map = (
            {int(source): int(target) for source, target in label_map.items()}
            if label_map is not None
            else None
        )
        self.refiner_config = dict(refiner or {})
        self.agreement_config = dict(agreement or {})
        if score_transform not in {None, "coarse_purity_sqrt"}:
            raise ValueError("unsupported score_transform")
        if score_transform is not None and family.strip().lower().replace("-", "") != "yolo":
            raise ValueError("score_transform is supported only for YOLO")
        self.score_transform = score_transform
        self._device = "cpu"
        self._model: Any | None = None
        self._refiner: Any | None = None
        self._refiner_metadata: dict[str, Any] = {}
        self._agreement: Any | None = None

    def load(self, checkpoint_path: str) -> None:
        checkpoint = Path(checkpoint_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"模型权重不存在: {checkpoint}")
        self._model = create_ultralytics_model(self.family, checkpoint)
        if self.agreement_config:
            if self.family.strip().lower().replace("-", "") != "yolo":
                raise ValueError("agreement runtime is supported only for YOLO")
            from rsdet.innovation.agreement_runtime import TileAgreementRuntime

            raw_adapter_path = str(self.agreement_config.get("checkpoint", "")).strip()
            if not raw_adapter_path:
                raise ValueError("model.agreement 已启用但 checkpoint 为空")
            adapter_path = Path(raw_adapter_path).expanduser()
            expected_sha256 = str(
                self.agreement_config.get("expected_sha256", "")
            ).strip() or None
            self._agreement = TileAgreementRuntime(
                self._model.model,
                adapter_path,
                expected_sha256=expected_sha256,
                category_id=int(self.agreement_config.get("category_id", 24)),
            )
        if self.refiner_config:
            from rsdet.models.prototype_refiner import load_refiner_checkpoint

            refiner_checkpoint = str(self.refiner_config.get("checkpoint", "")).strip()
            if not refiner_checkpoint:
                raise ValueError("model.refiner 已启用但 checkpoint 为空")
            self._refiner, self._refiner_metadata = load_refiner_checkpoint(
                refiner_checkpoint, map_location="cpu"
            )

    def to(self, device: str) -> None:
        self._device = str(device)
        if self._model is not None and hasattr(self._model, "to"):
            self._model.to(self._device)
        if self._refiner is not None:
            self._refiner.to(self._device)
        if self._agreement is not None:
            self._agreement.to(self._device)

    def eval(self) -> None:
        if self._model is None:
            return
        torch_model = getattr(self._model, "model", None)
        if torch_model is not None and hasattr(torch_model, "eval"):
            torch_model.eval()
        if self._refiner is not None:
            self._refiner.eval()

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)

    @staticmethod
    def _prepare_source(image: Any) -> Any:
        """Convert the project's RGB ndarray contract to Ultralytics BGR input."""
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Ultralytics 输入必须是 HWC RGB: {image.shape}")
            return np.ascontiguousarray(image[..., ::-1])
        return image

    def predict(self, batch: Sequence[InferenceSample]) -> list[Prediction]:
        if self._model is None:
            raise RuntimeError("模型尚未 load()")
        if not batch:
            return []

        if self._agreement is not None:
            self._agreement.clear()
        captured: list[Any] = []
        hook = None
        if self.score_transform is not None:
            hook = self._model.model.register_forward_hook(
                lambda _module, _inputs, output: captured.append(output)
            )
        try:
            results = self._model.predict(
                source=[self._prepare_source(sample.image) for sample in batch],
                imgsz=self.imgsz,
                conf=self.confidence,
                iou=self.iou,
                max_det=self.max_detections,
                device=self._device,
                quantize=16 if self.half and self._device != "cpu" else None,
                agnostic_nms=self.agnostic_nms,
                verbose=False,
                stream=False,
                batch=len(batch),
            )
        finally:
            if hook is not None:
                hook.remove()
        results = list(results)
        if len(results) != len(batch):
            raise ValueError(f"Ultralytics 返回 {len(results)} 个结果，输入 batch 为 {len(batch)}")

        probability_rows: list[np.ndarray] | None = None
        if self.score_transform == "coarse_purity_sqrt":
            if not captured:
                raise RuntimeError("score transform did not capture YOLO raw output")
            output = captured[-1]
            if not isinstance(output, tuple) or not isinstance(output[1], dict):
                raise TypeError("unexpected YOLO raw output for score transform")
            import torch

            processed = output[0]
            raw_scores = output[1]["one2one"]["scores"].sigmoid().permute(0, 2, 1)
            batch_size, anchors, classes = raw_scores.shape
            if batch_size != len(batch) or classes != 25:
                raise ValueError("unexpected YOLO score tensor shape")
            k = min(processed.shape[1], anchors)
            anchor_indices = raw_scores.max(dim=-1).values.topk(k, dim=1).indices
            selected = raw_scores.gather(
                1, anchor_indices[:, :, None].expand(-1, -1, classes)
            )
            flat_indices = selected.flatten(1).topk(k, dim=1).indices
            selected_rows = selected.gather(
                1, (flat_indices // classes)[:, :, None].expand(-1, -1, classes)
            )
            selected_scores = selected.flatten(1).gather(1, flat_indices)
            if not torch.allclose(
                selected_scores, processed[:, :k, 4], atol=2e-3, rtol=0
            ):
                raise RuntimeError("raw score reconstruction does not match YOLO output")
            probability_rows = []
            for batch_index in range(batch_size):
                mask = processed[batch_index, :k, 4] >= self.confidence
                probability_rows.append(
                    selected_rows[batch_index, mask].float().cpu().numpy()
                )

        predictions: list[Prediction] = []
        for batch_index, (sample, result) in enumerate(zip(batch, results)):
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                predictions.append(Prediction(sample.image_id, [], [], []))
                continue
            raw_boxes = [
                [float(coordinate) for coordinate in box] for box in self._as_list(boxes.xyxy)
            ]
            # 防御性 clamp 到原图边界：ultralytics 的 clip 在 RT-DETR 低置信度边缘
            # 目标上可能失效，导致 bbox 左上角越界为负（实测 -94px）。这里按原图
            # 尺寸强制 clamp，后续 valid 检查会过滤 clamp 后宽/高 <= 0 的退化框。
            img_h = float(getattr(sample, "height", None) or sample.image.shape[0])
            img_w = float(getattr(sample, "width", None) or sample.image.shape[1])
            raw_boxes = [
                [
                    max(0.0, min(box[0], img_w)),
                    max(0.0, min(box[1], img_h)),
                    max(0.0, min(box[2], img_w)),
                    max(0.0, min(box[3], img_h)),
                ]
                for box in raw_boxes
            ]
            raw_scores = [float(score) for score in self._as_list(boxes.conf)]
            raw_labels = [int(label) for label in self._as_list(boxes.cls)]
            if probability_rows is not None:
                if len(probability_rows[batch_index]) != len(raw_scores):
                    raise RuntimeError("raw probability rows do not align with detections")
                raw_scores = coarse_purity_sqrt_scores(
                    raw_scores, raw_labels, probability_rows[batch_index]
                )
            if self.label_map is not None:
                unknown = sorted(set(raw_labels) - set(self.label_map))
                if unknown:
                    raise ValueError(f"Ultralytics 返回未配置映射的类别: {unknown}")
                raw_labels = [self.label_map[label] for label in raw_labels]
            valid = [
                index
                for index, box in enumerate(raw_boxes)
                if len(box) == 4
                and all(math.isfinite(value) for value in box)
                and box[2] > box[0]
                and box[3] > box[1]
            ]
            predictions.append(
                Prediction(
                    image_id=sample.image_id,
                    boxes_xyxy=[raw_boxes[index] for index in valid],
                    scores=[raw_scores[index] for index in valid],
                    labels=[raw_labels[index] for index in valid],
                )
            )
        if self._agreement is not None:
            predictions = self._agreement.rescore(batch, predictions)
        return self._refine_predictions(batch, predictions)

    def _refine_predictions(
        self,
        batch: Sequence[InferenceSample],
        predictions: list[Prediction],
    ) -> list[Prediction]:
        """仅对不确定的舰船/飞机候选启动 HPR，避免固定增加全部候选时延。"""
        if self._refiner is None:
            return predictions
        import torch

        from rsdet.data.object_crops import (
            as_rgb_image,
            crop_and_resize,
            crop_to_tensor,
        )

        enabled_coarse = set(self.refiner_config.get("coarse_classes", ["ship", "aircraft"]))
        min_confidence = float(self.refiner_config.get("min_base_confidence", 0.0))
        max_confidence = float(self.refiner_config.get("max_base_confidence", 0.75))
        prototype_weight = float(self.refiner_config.get("prototype_weight", 0.35))
        score_blend = float(self.refiner_config.get("score_blend", 0.0))
        min_refined_confidence = float(self.refiner_config.get("min_refined_confidence", 0.0))
        min_refined_margin = float(self.refiner_config.get("min_refined_margin", 0.0))
        refiner_batch_size = int(self.refiner_config.get("batch_size", 128))
        input_size = int(
            self.refiner_config.get("input_size", self._refiner_metadata.get("input_size", 128))
        )
        context_ratio = float(
            self.refiner_config.get(
                "context_ratio", self._refiner_metadata.get("context_ratio", 0.15)
            )
        )
        if not 0.0 <= min_confidence <= max_confidence <= 1.0:
            raise ValueError(
                "refiner 置信度范围必须满足 0 <= min_base_confidence <= max_base_confidence <= 1"
            )
        if not 0.0 <= score_blend <= 1.0:
            raise ValueError("refiner.score_blend 必须在 [0, 1]")
        if not 0.0 <= min_refined_confidence <= 1.0:
            raise ValueError("refiner.min_refined_confidence 必须在 [0, 1]")
        if not 0.0 <= min_refined_margin <= 1.0:
            raise ValueError("refiner.min_refined_margin 必须在 [0, 1]")
        if refiner_batch_size <= 0:
            raise ValueError("refiner.batch_size 必须 > 0")

        refined = [
            Prediction(
                prediction.image_id,
                [list(box) for box in prediction.boxes_xyxy],
                list(prediction.scores),
                list(prediction.labels),
            )
            for prediction in predictions
        ]
        candidates: list[tuple[int, int, str]] = []
        crops = []
        for prediction_index, (sample, prediction) in enumerate(zip(batch, predictions)):
            source_image = as_rgb_image(sample.image)
            for detection_index, (box, score, label) in enumerate(
                zip(prediction.boxes_xyxy, prediction.scores, prediction.labels)
            ):
                group = coarse_name(int(label))
                if (
                    group not in enabled_coarse
                    or float(score) < min_confidence
                    or float(score) > max_confidence
                ):
                    continue
                crop = crop_and_resize(
                    source_image,
                    box,
                    output_size=input_size,
                    context_ratio=context_ratio,
                )
                crops.append(crop_to_tensor(crop))
                candidates.append((prediction_index, detection_index, group))
        if not crops:
            return refined

        device = next(self._refiner.parameters()).device
        for start in range(0, len(crops), refiner_batch_size):
            crop_batch = torch.stack(crops[start : start + refiner_batch_size]).to(device)
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=self.half and device.type == "cuda",
                ),
            ):
                outputs = self._refiner(crop_batch)
                logits = self._refiner.fused_logits(outputs, prototype_weight=prototype_weight)
            for local_index, row in enumerate(logits.float().cpu()):
                prediction_index, detection_index, group = candidates[start + local_index]
                if group == "ship":
                    allowed = list(range(0, 4))
                elif group == "aircraft":
                    allowed = list(range(4, 24))
                else:
                    allowed = [24]
                probabilities = row[allowed].softmax(dim=0)
                best_local = int(probabilities.argmax().item())
                refined_label = allowed[best_local]
                refined_confidence = float(probabilities[best_local].item())
                base_label = int(refined[prediction_index].labels[detection_index])
                base_local = allowed.index(base_label)
                base_probability = float(probabilities[base_local].item())
                if refined_label != base_label and (
                    refined_confidence < min_refined_confidence
                    or refined_confidence - base_probability < min_refined_margin
                ):
                    continue
                base_score = float(refined[prediction_index].scores[detection_index])
                refined[prediction_index].labels[detection_index] = refined_label
                if score_blend > 0.0:
                    refined[prediction_index].scores[detection_index] = (
                        base_score ** (1.0 - score_blend)
                        * max(refined_confidence, 1e-6) ** score_blend
                    )
        return refined


__all__ = [
    "UltralyticsDetector",
    "coarse_purity_sqrt_scores",
    "create_ultralytics_model",
]
