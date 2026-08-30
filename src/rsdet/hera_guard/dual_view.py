"""Seven-channel object view for HERA-Guard V3 pixel verification.

Channels are ``tight RGB + object-masked context RGB + proposal mask``.  The
context branch cannot copy object pixels: they are replaced with a robust ring
mean.  Out-of-image context is reflection padded before bicubic sampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


def square_box(box: Sequence[float], ratio: float = 1.0) -> tuple[float, float, float, float]:
    if len(box) != 4 or ratio <= 0:
        raise ValueError("box must be xyxy and ratio must be positive")
    x0, y0, x1, y1 = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
        raise ValueError("box must be finite and non-empty")
    side = max(x1 - x0, y1 - y0) * ratio
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0


def _reflect_render(
    image: Image.Image, box: Sequence[float], resolution: int
) -> Image.Image:
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    rgb_image = image if image.mode == "RGB" else image.convert("RGB")
    rgb = np.asarray(rgb_image)
    x0, y0, x1, y1 = (float(value) for value in box)
    ix0, iy0, ix1, iy1 = math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)
    clipped_x0, clipped_y0 = max(0, ix0), max(0, iy0)
    clipped_x1, clipped_y1 = min(rgb.shape[1], ix1), min(rgb.shape[0], iy1)
    patch = rgb[clipped_y0:clipped_y1, clipped_x0:clipped_x1]
    left = clipped_x0 - ix0
    top = clipped_y0 - iy0
    right = ix1 - clipped_x1
    bottom = iy1 - clipped_y1
    if not patch.size:
        raise ValueError("crop lies completely outside the source image")
    if left or top or right or bottom:
        mode = "reflect" if min(patch.shape[:2]) > 1 else "edge"
        patch = np.pad(patch, ((top, bottom), (left, right), (0, 0)), mode=mode)
    shifted = (x0 - ix0, y0 - iy0, x1 - ix0, y1 - iy0)
    return Image.fromarray(patch).transform(
        (resolution, resolution),
        Image.Transform.EXTENT,
        shifted,
        resample=Image.Resampling.BICUBIC,
    )


def proposal_mask_in_context(
    proposal_xyxy: Sequence[float],
    context_xyxy: Sequence[float],
    resolution: int,
) -> np.ndarray:
    px0, py0, px1, py1 = (float(value) for value in proposal_xyxy)
    cx0, cy0, cx1, cy1 = (float(value) for value in context_xyxy)
    scale_x, scale_y = resolution / (cx1 - cx0), resolution / (cy1 - cy0)
    x0 = max(0, min(resolution, math.floor((px0 - cx0) * scale_x)))
    y0 = max(0, min(resolution, math.floor((py0 - cy0) * scale_y)))
    x1 = max(0, min(resolution, math.ceil((px1 - cx0) * scale_x)))
    y1 = max(0, min(resolution, math.ceil((py1 - cy0) * scale_y)))
    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def _masked_context(context: np.ndarray, mask: np.ndarray) -> np.ndarray:
    foreground = mask > 0
    if not foreground.any():
        raise ValueError("proposal mask is empty")
    y, x = np.nonzero(foreground)
    margin = max(2, int(round(max(x.max() - x.min(), y.max() - y.min()) * 0.15)))
    y0, y1 = max(0, y.min() - margin), min(mask.shape[0], y.max() + margin + 1)
    x0, x1 = max(0, x.min() - margin), min(mask.shape[1], x.max() + margin + 1)
    ring = np.zeros_like(foreground)
    ring[y0:y1, x0:x1] = True
    ring &= ~foreground
    samples = context[ring]
    if not len(samples):
        samples = context[~foreground]
    fill = np.median(samples, axis=0) if len(samples) else np.full(3, 127.0)
    output = context.copy()
    output[foreground] = np.clip(np.rint(fill), 0, 255).astype(np.uint8)
    return output


def render_seven_channel_view(
    image: Image.Image,
    proposal_xyxy: Sequence[float],
    *,
    resolution: int = 224,
    context_ratio: float = 1.75,
) -> np.ndarray:
    """Return a deterministic ``uint8[7,H,W]`` deployment input."""

    tight_box = square_box(proposal_xyxy)
    context_box = square_box(proposal_xyxy, ratio=context_ratio)
    tight = np.asarray(_reflect_render(image, tight_box, resolution), dtype=np.uint8)
    context = np.asarray(_reflect_render(image, context_box, resolution), dtype=np.uint8)
    mask = proposal_mask_in_context(proposal_xyxy, context_box, resolution)
    masked = _masked_context(context, mask)
    output = np.concatenate(
        [tight.transpose(2, 0, 1), masked.transpose(2, 0, 1), mask[None]], axis=0
    )
    if output.shape != (7, resolution, resolution):
        raise AssertionError("seven-channel view shape contract failed")
    return output


try:
    from torch import nn as _nn
except ImportError:  # pragma: no cover

    class _ModuleBase:  # type: ignore[no-redef]
        pass
else:
    _ModuleBase = _nn.Module


@dataclass(frozen=True)
class DualViewOutput:
    foreground_logit: Any
    fine_logits: Any
    quality_logit: Any
    risk_residual_logit: Any
    embedding: Any


class DualViewVerifier(_ModuleBase):
    def __init__(self, backbone: Any, *, feature_dim: int = 768, num_fine: int = 25) -> None:
        from torch import nn

        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.foreground_head = nn.Linear(feature_dim, 1)
        self.fine_head = nn.Linear(feature_dim, num_fine)
        self.quality_head = nn.Linear(feature_dim, 1)
        self.risk_head = nn.Linear(feature_dim, 1)

    def forward(self, image: Any) -> DualViewOutput:
        feature_map = self.backbone.features(image)
        pooled = self.backbone.avgpool(feature_map)
        embedding = self.backbone.classifier[0](pooled).flatten(1)
        return DualViewOutput(
            foreground_logit=self.foreground_head(embedding).squeeze(1),
            fine_logits=self.fine_head(embedding),
            quality_logit=self.quality_head(embedding).squeeze(1),
            risk_residual_logit=self.risk_head(embedding).squeeze(1),
            embedding=embedding,
        )


def build_dual_view_verifier(
    *, weight_path: str | Path, verify_weight_sha256: bool = True
) -> DualViewVerifier:
    """Load project ConvNeXt-T weights and expand its stem 3→7 channels."""

    from torch import nn

    from rsdet.hera_guard.verifier import build_convnext_tiny_backbone

    backbone = build_convnext_tiny_backbone(
        weight_path=weight_path,
        freeze="full",
        verify_weight_sha256=verify_weight_sha256,
    )
    old = backbone.features[0][0]
    if not isinstance(old, nn.Conv2d) or old.in_channels != 3:
        raise ValueError("unexpected ConvNeXt stem contract")
    new = nn.Conv2d(
        7,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
    )
    with __import__("torch").no_grad():
        new.weight[:, :3].copy_(old.weight * 0.5)
        new.weight[:, 3:6].copy_(old.weight * 0.5)
        new.weight[:, 6:].zero_()
        if old.bias is not None:
            new.bias.copy_(old.bias)
    backbone.features[0][0] = new
    return DualViewVerifier(backbone)


__all__ = [
    "DualViewOutput",
    "DualViewVerifier",
    "build_dual_view_verifier",
    "proposal_mask_in_context",
    "render_seven_channel_view",
    "square_box",
]
