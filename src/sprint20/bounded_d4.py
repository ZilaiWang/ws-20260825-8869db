"""Opt-in wrapper around the EXISTING loaded AircraftD4ClassifierRuntime."""

from __future__ import annotations

import numpy as np

from .policy import AIRCRAFT, _apply_aircraft_labels, certify_keep, full_view_labels


class BoundedD4Runtime:
    def __init__(self, delegate, *, margin: float = 2e-4):
        self.delegate = delegate
        self.margin = float(margin)
        self.last_audit = {}
        if not delegate.tensorized_views:
            raise ValueError("bounded D4 requires the already-audited tensorized_views=true")
        if not np.isfinite(self.margin) or self.margin < 0:
            raise ValueError("invalid margin")

    def refine(self, rgb, prediction):
        import torch
        from PIL import Image

        from rsdet.data.crop_classification import render_crop
        from rsdet.postprocess.nms import nms
        from rsdet.submission.aircraft_d4 import _normalize, _tensorized_d4_views

        d = self.delegate
        indices = [i for i, label in enumerate(prediction.labels) if int(label) in AIRCRAFT]
        self.last_audit = {
            "aircraft_objects": len(indices),
            "certified_keep": 0,
            "full_d4_objects": 0,
            "view_evaluations": 0,
        }
        if not indices:
            return prediction
        tau = float(d.config.get("relabel_min_probability", 0.9))
        nms_iou = float(d.config.get("nms_iou", 0.5))
        batch_size = int(d.config.get("batch_objects", 64))
        if batch_size <= 0:
            raise ValueError("batch_objects must be positive")
        source = Image.fromarray(rgb)
        decisions = []
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            crops = [render_crop(source, prediction.boxes_xyxy[i], 224) for i in batch_indices]
            base = torch.stack([_normalize(crop) for crop in crops]).to(d.device, non_blocking=True)
            first_images = (
                base.contiguous(memory_format=torch.channels_last) if d.channels_last else base
            )
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=d.device.type, dtype=torch.float16, enabled=d.device.type == "cuda"
                ),
            ):
                first_logits = d.model(first_images)
            first = first_logits.float()[:, 4:24].softmax(1).cpu().numpy()
            original = np.asarray(
                [int(prediction.labels[i]) - 4 for i in batch_indices], dtype=np.int64
            )
            keep = certify_keep(first, original, threshold=tau, margin=self.margin)
            chosen = original.copy()
            pending = np.flatnonzero(~keep)
            if len(pending):
                # Recompute ALL EIGHT views for unresolved objects, exactly the
                # original formula. No approximation by a one-view decision.
                selected = base.index_select(0, torch.as_tensor(pending, device=d.device))
                images = _tensorized_d4_views(selected)
                if d.channels_last:
                    images = images.contiguous(memory_format=torch.channels_last)
                with (
                    torch.inference_mode(),
                    torch.autocast(
                        device_type=d.device.type,
                        dtype=torch.float16,
                        enabled=d.device.type == "cuda",
                    ),
                ):
                    logits = d.model(images).reshape(len(pending), 8, 25)
                probs = logits.float()[:, :, 4:24].softmax(2).mean(1).cpu().numpy()
                chosen[pending] = full_view_labels(probs, original[pending], tau)
            decisions.extend(chosen.tolist())
            self.last_audit["certified_keep"] += int(keep.sum())
            self.last_audit["full_d4_objects"] += len(pending)
            self.last_audit["view_evaluations"] += len(batch_indices) + 8 * len(pending)
        self.last_audit["original_view_evaluations"] = 8 * len(indices)
        return _apply_aircraft_labels(prediction, decisions, nms, nms_iou=nms_iou)
