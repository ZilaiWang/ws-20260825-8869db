"""Pinned YOLO26 head readout using the existing repository and native decoder.

The shared mode is experimental: native OTO and native OTM parity must BOTH
pass on the server before replacing two native passes by one shared backbone.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import importlib.metadata
import time
from typing import Any

import numpy as np

from . import PINNED_ULTRALYTICS
from .policy import route_after_fusion


def require_version() -> str:
    version = importlib.metadata.version("ultralytics")
    if version != PINNED_ULTRALYTICS:
        raise RuntimeError(
            f"Audited ultralytics={PINNED_ULTRALYTICS}; found {version}. "
            "Do NOT upgrade the competition environment for this patch."
        )
    return version


def inspect_head(wrapper: Any) -> dict:
    head = wrapper.model.model[-1]
    result = {
        "type": type(head).__name__,
        "nc": int(head.nc),
        "end2end": bool(head.end2end),
        "reg_max": int(head.reg_max),
    }
    for attr in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        value = getattr(head, attr, None)
        result[attr] = value is not None
        if value is not None:
            result[attr + "_parameter_count"] = sum(p.numel() for p in value.parameters())
    return result


def assert_available(wrapper: Any, mode: str) -> None:
    info = inspect_head(wrapper)
    if info["nc"] != 25:
        raise ValueError(f"Expected the trained 25-class P40 checkpoint, got {info}")
    if mode in {"otm", "shared"} and not (info["cv2"] and info["cv3"]):
        raise RuntimeError(
            "OTM head absent/already fused away. Reload an UNFUSED "
            "checkpoint from the SAME P40 epoch; never random-initialize it."
        )
    if mode == "oto" and not info["end2end"]:
        raise RuntimeError(
            "Checkpoint is not OTO-default: native-OTO baseline must be explicitly re-audited"
        )
    if mode in {"oto", "shared"} and not (info["one2one_cv2"] and info["one2one_cv3"]):
        raise RuntimeError("OTO head absent")


def select_native_otm(detector: Any) -> Any:
    """Opt into the library's native OTM path BEFORE its first prediction."""
    require_version()
    wrapper = detector._model
    if getattr(wrapper, "predictor", None) is not None:
        raise RuntimeError("Native head selection requires a fresh, unused detector")
    assert_available(wrapper, "otm")
    # Native setup_model consumes end2end=False BEFORE calling model.fuse().
    wrapper.predict = functools.partial(wrapper.predict, end2end=False)
    wrapper.model.end2end = False
    if bool(wrapper.model.model[-1].end2end):
        raise RuntimeError("Model-to-head end2end setter did not propagate")
    return detector


def sample_key(sample: Any) -> tuple:
    # A single-image cache is also cleared at begin_image(). Pixel hashes
    # prevent a same-ID/same-size but different-image accidental replay.
    image = np.ascontiguousarray(sample.image)
    digest = hashlib.blake2b(image.view(np.uint8), digest_size=16).hexdigest()
    return (int(sample.image_id), int(sample.width), int(sample.height), digest)


class SharedHeadCapture:
    """Wrap the real UltralyticsDetector; keep native OTO output untouched.

    Conv/BN is fused with end2end temporarily disabled, which prevents the
    library from deleting the co-trained OTM head. Subsequent AutoBackend
    fusion sees an already fused model. Raw OTM is decoded with the exact
    v8.4.103 Detect._inference and native NMS, NOT a handwritten approximation.
    """

    def __init__(self, delegate: Any):
        require_version()
        if getattr(delegate._model, "predictor", None) is not None:
            raise RuntimeError("Shared readout must be attached BEFORE first predict")
        if delegate.refiner_config or delegate.agreement_config or delegate.score_transform:
            raise ValueError("Shared P40 probe requires the plain identity detector")
        if delegate.label_map is not None or delegate.drop_labels:
            raise ValueError("Label remapping/filtering is not supported by this P40 probe")
        assert_available(delegate._model, "shared")
        self.delegate = delegate
        self.cache: dict[tuple, Any] = {}
        self._prepared = False
        self._captured = None
        self._input_hw = None
        self._hook = None
        self.ordered_otm = []
        self._ordered_only = False

    def begin_image(self, *, ordered_only: bool = False):
        self.cache.clear()
        self.ordered_otm.clear()
        self._ordered_only = bool(ordered_only)
        self._captured = None
        self._input_hw = None

    def load(self, path):
        raise RuntimeError("Construct with an already loaded, unused detector")

    def to(self, device):
        self.delegate.to(device)

    def eval(self):
        self.delegate.eval()

    def _prepare(self):
        if self._prepared:
            return
        model = self.delegate._model.model
        head = model.model[-1]
        if bool(getattr(head, "export", False)) or bool(getattr(head, "xyxy", False)):
            raise RuntimeError(
                "Export/xyxy-altered checkpoints are not supported by shared OTM decoding"
            )
        was = bool(head.end2end)
        if not was:
            raise RuntimeError("Shared mode requires original OTO-default P40")
        # Called AFTER delegate.to(device), matching native fusion device/order.
        head.end2end = False
        try:
            model.fuse(verbose=False)
        finally:
            head.end2end = was
        if not model.is_fused():
            raise RuntimeError("Conv/BN fusion incomplete; unsafe to preserve both heads")
        assert_available(self.delegate._model, "shared")
        self._hook = model.register_forward_hook(self._capture)
        self._prepared = True

    def _capture(self, module, args, output):
        del module
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("Pinned raw forward contract changed")
        if not isinstance(output[1], dict) or not output[1].get("one2many"):
            raise RuntimeError("OTM raw predictions disappeared (possibly fused away)")
        self._captured = output[1]["one2many"]
        self._input_hw = tuple(int(x) for x in args[0].shape[-2:])

    def predict(self, batch):
        if not batch:
            return []
        import torch
        from ultralytics.utils import ops
        from ultralytics.utils.nms import non_max_suppression

        from rsdet.contracts import Prediction

        self._prepare()
        self._captured = None
        # Exact original adapter including its conversion, clipping and ordering.
        primary = self.delegate.predict(batch)
        raw, input_hw = self._captured, self._input_hw
        if raw is None or raw["scores"].shape[0] != len(batch):
            raise RuntimeError("Captured OTM batch does not align with native OTO results")
        head = self.delegate._model.model.model[-1]
        was = bool(head.end2end)
        try:
            # Native OTM decode needs xywh, rather than the OTO xyxy contract.
            head.end2end = False
            with torch.inference_mode():
                dense = head._inference(raw)
                detections = non_max_suppression(
                    dense,
                    conf_thres=self.delegate.confidence,
                    iou_thres=self.delegate.iou,
                    agnostic=False,
                    max_det=self.delegate.max_detections,
                    nc=25,
                    end2end=False,
                )
        finally:
            head.end2end = was
            self._captured = None
        if len(detections) != len(batch):
            raise RuntimeError("OTM NMS batch mismatch")
        for sample, det in zip(batch, detections, strict=True):
            if len(det):
                # ``non_max_suppression`` runs under inference mode and may
                # return inference tensors.  ``scale_boxes`` performs an
                # in-place coordinate update, which PyTorch rejects once the
                # inference-mode context has ended.  Clone to a normal tensor
                # before scaling; this also leaves the captured decode output
                # untouched for parity diagnostics.
                det = det.clone()
                det[:, :4] = ops.scale_boxes(input_hw, det[:, :4], sample.image.shape)
            boxes = det[:, :4].detach().cpu().tolist()
            scores = det[:, 4].detach().cpu().tolist()
            labels = det[:, 5].detach().cpu().long().tolist()
            clipped = [
                [
                    max(0.0, min(float(b[0]), sample.width)),
                    max(0.0, min(float(b[1]), sample.height)),
                    max(0.0, min(float(b[2]), sample.width)),
                    max(0.0, min(float(b[3]), sample.height)),
                ]
                for b in boxes
            ]
            valid = [
                i
                for i, b in enumerate(clipped)
                if np.isfinite(b).all() and b[2] > b[0] and b[3] > b[1]
            ]
            prediction = Prediction(
                sample.image_id,
                [clipped[i] for i in valid],
                [float(scores[i]) for i in valid],
                [int(labels[i]) for i in valid],
            )
            self.ordered_otm.append(prediction)
            if not self._ordered_only:
                key = sample_key(sample)
                if key in self.cache:
                    raise RuntimeError("Duplicate tile in one parent image")
                self.cache[key] = prediction
        return primary

    def close(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        self.cache.clear()


class CachedOTM:
    def __init__(self, capture: SharedHeadCapture):
        self.capture = capture
        self.used = set()

    def eval(self):
        """Satisfy the repository ``BaseDetector`` inference contract.

        The cached branch has no live module to switch, but
        ``predict_batches`` deliberately calls ``eval`` on every detector.
        Keeping this as an explicit no-op avoids bypassing that common path.
        """
        return None

    def predict(self, batch):
        output = []
        for sample in batch:
            key = sample_key(sample)
            if key not in self.capture.cache or key in self.used:
                raise RuntimeError("Missing/stale/reused OTM tile cache")
            self.used.add(key)
            output.append(copy.deepcopy(self.capture.cache[key]))
        return output


class SharedHeadPipeline:
    def __init__(
        self,
        capture,
        config,
        *,
        otm_labels,
        primary_threshold,
        otm_threshold,
        primary_threshold_by_fine=None,
        optimized=False,
    ):
        self.capture, self.config = capture, config
        self.otm_labels = tuple(otm_labels)
        self.primary_threshold, self.otm_threshold = float(primary_threshold), float(otm_threshold)
        self.primary_threshold_by_fine = dict(primary_threshold_by_fine or {})
        self.optimized = bool(optimized)
        self.last_primary = None
        self.last_otm = None

    def predict_image(self, rgb, parent_image_id=0):
        if self.optimized:
            return self._predict_image_optimized(rgb, parent_image_id=parent_image_id)
        from rsdet.pipeline.large_image import run_pipeline

        self.capture.begin_image()
        primary, primary_time = run_pipeline(
            rgb, self.capture, config=self.config, parent_image_id=parent_image_id
        )
        cached = CachedOTM(self.capture)
        otm, fusion_time = run_pipeline(
            rgb, cached, config=self.config, parent_image_id=parent_image_id
        )
        if len(cached.used) != len(self.capture.cache):
            raise RuntimeError("Second pipeline did not consume exactly the first grid")
        self.last_primary, self.last_otm = primary, otm
        routed = route_after_fusion(
            primary,
            otm,
            alternative_labels=self.otm_labels,
            primary_threshold=self.primary_threshold,
            alternative_threshold=self.otm_threshold,
            primary_threshold_by_fine=self.primary_threshold_by_fine,
        )
        # Includes OTM decode, both fusions and cache/hash overhead. It is NOT
        # an official endpoint latency and must not be substituted for one.
        return routed, {
            "primary": primary_time.to_dict(),
            "cached_otm_pipeline": fusion_time.to_dict(),
        }

    def _predict_image_optimized(self, rgb, *, parent_image_id):
        """Fuse both captured heads from one materialized tile grid.

        The scientific candidate is unchanged: OTO and OTM still receive two
        independent Safe Fusion passes before class-disjoint routing.  This
        path removes only the second crop/copy/hash traversal used by the
        parity-oriented cache implementation.
        """
        from rsdet.contracts import InferenceSample
        from rsdet.engine.predictor import predict_batches
        from rsdet.pipeline.large_image import _extract_tile_image
        from rsdet.postprocess.safe_tile_fusion import fuse_safe_tile_predictions
        from rsdet.tiling.slicer import generate_tiles

        if self.config.fusion != "safe":
            raise ValueError("optimized shared head path is frozen to Safe Fusion")
        height, width = rgb.shape[:2]
        started = time.perf_counter()
        tiles = generate_tiles(
            image_width=width,
            image_height=height,
            tile_size=self.config.tile_size,
            overlap=self.config.overlap,
        )
        samples = []
        for tile in tiles:
            tile.image_id = tile.tile_id
            tile.parent_image_id = parent_image_id
            patch = _extract_tile_image(rgb, tile)
            samples.append(
                InferenceSample(
                    image_id=tile.tile_id,
                    image=patch,
                    width=tile.width,
                    height=tile.height,
                    metadata={
                        "tile_x_offset": tile.x_offset,
                        "tile_y_offset": tile.y_offset,
                        "tile_width": tile.width,
                        "tile_height": tile.height,
                    },
                )
            )
        tiling_seconds = time.perf_counter() - started
        self.capture.begin_image(ordered_only=True)
        model_started = time.perf_counter()
        primary_tiles = predict_batches(
            self.capture,
            samples,
            batch_size=self.config.batch_size,
        )
        model_seconds = time.perf_counter() - model_started
        otm_tiles = self.capture.ordered_otm
        if len(primary_tiles) != len(tiles) or len(otm_tiles) != len(tiles):
            raise RuntimeError("shared head tile predictions do not align with the frozen grid")
        for index, (primary, alternative, tile) in enumerate(
            zip(primary_tiles, otm_tiles, tiles, strict=True)
        ):
            if primary.image_id != tile.tile_id or alternative.image_id != tile.tile_id:
                raise RuntimeError(f"shared head tile id mismatch at index {index}")

        def fuse(rows):
            return fuse_safe_tile_predictions(
                rows,
                tiles,
                image_width=width,
                image_height=height,
                parent_image_id=parent_image_id,
                score_threshold=self.config.score_threshold,
                score_threshold_by_coarse=self.config.score_threshold_by_coarse,
                score_threshold_by_fine=self.config.score_threshold_by_fine,
                merge_iou=self.config.merge_iou,
                merge_ios=self.config.merge_ios,
                fine_nms_iou=self.config.fine_nms_iou,
                border_margin=self.config.border_margin,
                max_detections=self.config.max_detections,
                output_score_threshold=self.config.output_score_threshold,
                owner_logit_slack=self.config.owner_logit_slack,
                threshold_safe_category_ids=self.config.threshold_safe_category_ids,
            )

        fusion_started = time.perf_counter()
        primary = fuse(primary_tiles)
        otm = fuse(otm_tiles)
        fusion_seconds = time.perf_counter() - fusion_started
        self.last_primary, self.last_otm = primary, otm
        routed = route_after_fusion(
            primary,
            otm,
            alternative_labels=self.otm_labels,
            primary_threshold=self.primary_threshold,
            alternative_threshold=self.otm_threshold,
            primary_threshold_by_fine=self.primary_threshold_by_fine,
        )
        return routed, {
            "optimized_shared_pipeline": {
                "tiling_s": tiling_seconds,
                "model_and_head_decode_s": model_seconds,
                "two_safe_fusions_s": fusion_seconds,
                "n_tiles": len(tiles),
            }
        }
