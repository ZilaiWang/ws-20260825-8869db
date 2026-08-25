"""FRFDet-inspired symmetric sampling for the YOLO26 P2 neck.

The mathematical structure follows the Apache-2.0 FRFDet reference at commit
``d424df831da98f0184a8316e73b545add2b0f7a5``.  This project adapts the
implementation to plain PyTorch and limits it to one P2 up/down pair, rather
than copying the full FRFDet detector.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

IBS_VARIANT = "ibs_p2_pair_v1"
FRFDET_REFERENCE_COMMIT = "d424df831da98f0184a8316e73b545add2b0f7a5"


class ConvBNAct(nn.Sequential):
    """Bias-free convolution, BatchNorm and SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        groups: int = 1,
    ) -> None:
        if in_channels <= 0 or out_channels <= 0 or kernel_size <= 0:
            raise ValueError("卷积通道和核尺寸必须为正整数")
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


def _space_to_channels(x: torch.Tensor, factor: int) -> torch.Tensor:
    """FRFDet unfold/rearrange order: ``kh, kw, channel``."""

    if x.ndim != 4:
        raise ValueError("IBS 输入必须是 BCHW")
    batch, channels, height, width = x.shape
    if height % factor or width % factor:
        raise ValueError(f"IBS-D 空间尺寸必须被 factor={factor} 整除")
    patches = x.unfold(2, factor, factor).unfold(3, factor, factor)
    return (
        patches.permute(0, 4, 5, 1, 2, 3)
        .contiguous()
        .view(batch, factor * factor * channels, height // factor, width // factor)
    )


def _channels_to_space(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Exact inverse of :func:`_space_to_channels`."""

    if x.ndim != 4:
        raise ValueError("IBS 输入必须是 BCHW")
    batch, channels, height, width = x.shape
    square = factor * factor
    if channels % square:
        raise ValueError(f"IBS-U 通道数必须被 factor^2={square} 整除")
    reduced = channels // square
    return (
        x.view(batch, factor, factor, reduced, height, width)
        .permute(0, 3, 4, 1, 5, 2)
        .contiguous()
        .view(batch, reduced, height * factor, width * factor)
    )


class IBSDown(nn.Module):
    """Expansion-compression followed by learnable spatial reorganization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        factor: int = 2,
        expansion_ratio: int = 2,
    ) -> None:
        super().__init__()
        if factor <= 1 or out_channels % (factor * factor):
            raise ValueError("IBS-D out_channels 必须被 factor^2 整除")
        hidden = out_channels * expansion_ratio
        self.factor = factor
        self.expand = ConvBNAct(in_channels, hidden, 1)
        self.depthwise = ConvBNAct(hidden, hidden, 3, groups=hidden)
        self.compress = ConvBNAct(hidden, out_channels // (factor * factor), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expanded = self.expand(x)
        compressed = self.compress(expanded + self.depthwise(expanded))
        return _space_to_channels(compressed, self.factor)


class IBSUp(nn.Module):
    """Inverse reorganization followed by expansion-compression without residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        factor: int = 2,
        expansion_ratio: int = 2,
    ) -> None:
        super().__init__()
        if factor <= 1 or in_channels % (factor * factor):
            raise ValueError("IBS-U in_channels 必须被 factor^2 整除")
        hidden = out_channels * expansion_ratio
        self.factor = factor
        self.expand = ConvBNAct(in_channels // (factor * factor), hidden, 1)
        self.depthwise = ConvBNAct(hidden, hidden, 3, groups=hidden)
        self.compress = ConvBNAct(hidden, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rearranged = _channels_to_space(x, self.factor)
        return self.compress(self.depthwise(self.expand(rearranged)))


def _copy_ultralytics_routing(source: nn.Module, target: nn.Module) -> None:
    for attribute in ("i", "f"):
        if hasattr(source, attribute):
            setattr(target, attribute, getattr(source, attribute))
    target.type = f"rsdet.{type(target).__name__}"
    target.np = sum(parameter.numel() for parameter in target.parameters())


def inject_p2_ibs_pair(model: Any) -> dict[str, Any]:
    """Replace only P3->P2 upsampling and P2->P3 downsampling.

    YOLO26-P2 layer indices are frozen by the official 8.4.103 YAML.  All
    other backbone, neck and Detect layers remain untouched.
    """

    layers = getattr(model, "model", None)
    if layers is None or len(layers) != 30:
        raise ValueError("仅支持 Ultralytics 8.4.103 官方 YOLO26-P2 30 层结构")
    if isinstance(layers[17], IBSUp) and isinstance(layers[20], IBSDown):
        return dict(getattr(model, "_rsdet_ibs_audit"))
    original_up = layers[17]
    original_down = layers[20]
    conv = getattr(original_down, "conv", None)
    if not isinstance(conv, nn.Conv2d):
        raise ValueError("YOLO26-P2 layer20 不是预期 Conv 下采样")
    down_in = int(conv.in_channels)
    down_out = int(conv.out_channels)

    captured: dict[str, int] = {}

    def capture(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured["up_in"] = int(inputs[0].shape[1])

    hook = original_up.register_forward_pre_hook(capture)
    training = bool(model.training)
    try:
        model.eval()
        first_parameter = next(model.parameters())
        with torch.no_grad():
            model(torch.zeros(1, 3, 64, 64, device=first_parameter.device))
    finally:
        hook.remove()
        model.train(training)
    up_in = captured.get("up_in")
    if up_in is None:
        raise ValueError("无法推断 YOLO26-P2 layer17 输入通道")

    replacement_up = IBSUp(up_in, up_in, factor=2, expansion_ratio=2)
    replacement_down = IBSDown(down_in, down_out, factor=2, expansion_ratio=2)
    _copy_ultralytics_routing(original_up, replacement_up)
    _copy_ultralytics_routing(original_down, replacement_down)
    layers[17] = replacement_up
    layers[20] = replacement_down
    audit = {
        "variant": IBS_VARIANT,
        "up_layer": 17,
        "down_layer": 20,
        "up_channels": [up_in, up_in],
        "down_channels": [down_in, down_out],
        "factor": 2,
        "expansion_ratio": 2,
        "reference_commit": FRFDET_REFERENCE_COMMIT,
    }
    model._rsdet_ibs_audit = audit
    return dict(audit)


def build_ibs_detection_trainer():
    """Return an Ultralytics trainer that injects the pair after weight load."""

    try:
        from ultralytics.models.yolo.detect import DetectionTrainer
    except ImportError as error:
        raise ImportError("IBS trainer 需要 ultralytics") from error

    class IBSP2Trainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            result = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
            inject_p2_ibs_pair(result)
            return result

    IBSP2Trainer.__name__ = "IBSP2Trainer"
    return IBSP2Trainer


__all__ = [
    "FRFDET_REFERENCE_COMMIT",
    "IBS_VARIANT",
    "IBSDown",
    "IBSUp",
    "build_ibs_detection_trainer",
    "inject_p2_ibs_pair",
]
