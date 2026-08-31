"""Tile-level runtime for the in-model D-FINE agreement residual.

The agreement branch is evaluated before large-image tile fusion.  This is
important: fused boxes no longer have a one-to-one ROI in any detector feature
map, whereas each raw tile proposal does.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TileAgreementRuntime:
    """Load an agreement adapter and rescore vehicle proposals on tile FPNs."""

    def __init__(
        self,
        detection_model: Any,
        adapter_path: Path,
        *,
        expected_sha256: str | None = None,
        category_id: int = 24,
    ) -> None:
        import torch

        from rsdet.innovation.in_model_agreement import AgreementResidualHead
        from rsdet.innovation.yolo_feature_quality import (
            MultiScaleROIFeatureEncoder,
            YoloPyramidTap,
        )

        if not adapter_path.is_file():
            raise FileNotFoundError(adapter_path)
        actual_sha256 = sha256_file(adapter_path)
        if expected_sha256 is not None and actual_sha256 != expected_sha256.lower():
            raise ValueError("agreement adapter SHA mismatch")
        try:
            state = torch.load(adapter_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(adapter_path, map_location="cpu")
        if state.get("protocol") not in {
            "in_model_dfine_agreement_distillation_v1",
            "in_model_dfine_agreement_and_official_match_distillation_v2",
        }:
            raise ValueError("unsupported agreement adapter protocol")
        indices = tuple(int(value) for value in state["layer_indices"])
        detect_indices = tuple(int(value) for value in detection_model.model[-1].f)
        if indices != detect_indices:
            raise ValueError("adapter feature layers do not match detector architecture")
        self.adapter_path = adapter_path.resolve()
        self.adapter_sha256 = actual_sha256
        self.category_id = int(category_id)
        self.imgsz = int(state["imgsz"])
        self.residual_alpha = float(state["residual_alpha"])
        self.tap = YoloPyramidTap(detection_model, indices)
        self.roi_encoder = MultiScaleROIFeatureEncoder(
            state["channels"],
            state["strides"],
            projection_dim=int(state["projection_dim"]),
            output_size=3,
            context_ratio=float(state["context_ratio"]),
        )
        self.quality_head = AgreementResidualHead(
            self.roi_encoder.output_dim,
            hidden_dim=int(state["hidden_dim"]),
        )
        self.roi_encoder.load_state_dict(state["roi_encoder_state_dict"])
        self.quality_head.load_state_dict(state["quality_head_state_dict"])
        self.roi_encoder.eval()
        self.quality_head.eval()

    def to(self, device: str) -> None:
        self.roi_encoder.to(device)
        self.quality_head.to(device)

    def clear(self) -> None:
        self.tap.clear()

    def close(self) -> None:
        self.tap.close()

    def rescore(
        self,
        samples: Sequence[Any],
        predictions: Sequence[Any],
    ) -> list[Any]:
        """Return copied predictions with vehicle scores changed in tile space."""
        import torch

        from rsdet.contracts import Prediction
        from rsdet.innovation.in_model_agreement import (
            apply_logit_residual,
            boxes_to_letterbox,
            letterbox_geometry,
            proposal_metadata,
        )

        if len(samples) != len(predictions):
            raise ValueError("agreement sample/prediction rows are not aligned")
        feature_maps = self.tap.features(detach=True)
        if any(int(feature.shape[0]) != len(samples) for feature in feature_maps):
            raise RuntimeError(
                "YOLO predictor did not preserve the requested tile batch; "
                "agreement features cannot be aligned safely"
            )
        boxes_with_batch = []
        detector_scores = []
        row_locations: list[tuple[int, int]] = []
        for sample_index, (sample, prediction) in enumerate(zip(samples, predictions, strict=True)):
            width = int(getattr(sample, "width", None) or sample.image.shape[1])
            height = int(getattr(sample, "height", None) or sample.image.shape[0])
            geometry = letterbox_geometry(width, height, self.imgsz)
            selected_boxes = []
            selected_scores = []
            selected_locations = []
            for detection_index, (box, score, label) in enumerate(
                zip(
                    prediction.boxes_xyxy,
                    prediction.scores,
                    prediction.labels,
                    strict=True,
                )
            ):
                if int(label) != self.category_id:
                    continue
                selected_boxes.append([float(value) for value in box])
                selected_scores.append(float(score))
                selected_locations.append((sample_index, detection_index))
            if not selected_boxes:
                continue
            box_tensor = torch.tensor(
                selected_boxes,
                dtype=torch.float32,
                device=feature_maps[0].device,
            )
            model_boxes = boxes_to_letterbox(box_tensor, geometry)
            batch_column = torch.full(
                (len(model_boxes), 1),
                float(sample_index),
                dtype=model_boxes.dtype,
                device=model_boxes.device,
            )
            boxes_with_batch.append(torch.cat((batch_column, model_boxes), dim=1))
            detector_scores.append(
                torch.tensor(
                    selected_scores,
                    dtype=torch.float32,
                    device=feature_maps[0].device,
                )
            )
            row_locations.extend(selected_locations)
        copied = [
            Prediction(
                prediction.image_id,
                [list(box) for box in prediction.boxes_xyxy],
                list(prediction.scores),
                list(prediction.labels),
            )
            for prediction in predictions
        ]
        if not boxes_with_batch:
            return copied
        all_boxes = torch.cat(boxes_with_batch, dim=0)
        all_scores = torch.cat(detector_scores, dim=0)
        roi = self.roi_encoder(
            feature_maps,
            all_boxes,
            image_height=self.imgsz,
            image_width=self.imgsz,
        )
        metadata = proposal_metadata(all_scores, all_boxes[:, 1:], image_size=self.imgsz)
        with torch.inference_mode():
            residual = self.quality_head(roi, metadata)
            rescored = apply_logit_residual(
                all_scores,
                residual,
                alpha=self.residual_alpha,
            )
        if not torch.isfinite(rescored).all():
            raise RuntimeError("agreement runtime produced NaN/Inf")
        for (sample_index, detection_index), score in zip(
            row_locations, rescored.detach().cpu().tolist(), strict=True
        ):
            copied[sample_index].scores[detection_index] = float(score)
        return copied


__all__ = ["TileAgreementRuntime", "sha256_file"]
