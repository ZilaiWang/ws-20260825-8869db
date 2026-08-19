"""N2-CFG 模型：ConvNeXt-T shared trunk + 粗类条件式前景头。

结构（对应《改进方案 1》第 2.5 节）：

::

    context_1.25 crop (224x224)
      -> ConvNeXt-T shared trunk (features + avgpool)
      -> shared foreground head      : Linear(in_features, 1)
      -> coarse residual head        : Linear(in_features, 1) x {ship, aircraft, vehicle}
      -> p_fg[coarse] = sigmoid(shared + residual[coarse])

三种快筛模式只取不同输出：

- S1 用 ``shared_logit``（不含粗类残差，coarse-agnostic）；
- S2 用 ``shared_logit + coarse_residual[coarse]``（coarse-conditioned）；
- S0 不用模型输出（``beta=0``）。

所有 PyTorch/torchvision 导入延迟到函数内，保持仓库在无深度学习环境
也可导入本模块的其他部分（与 ``crop_classifier.py`` 约定一致）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from rsdet.analysis.background_gate import COARSE_CLASSES
from rsdet.models.crop_classifier import (
    CONVNEXT_TINY_WEIGHT_SHA256,
    sha256_file,
)

# 冻结策略。
FREEZE_BACKBONE = "freeze_backbone"  # 快筛 Level-E：只训 shared + coarse heads
FREEZE_FIRST_THREE_STAGES = "freeze_first_three_stages"  # 正式：训最后 stage + heads
FULL_FINETUNE = "full"  # 参考，不冻结

ALLOWED_FREEZE: frozenset[str] = frozenset(
    {FREEZE_BACKBONE, FREEZE_FIRST_THREE_STAGES, FULL_FINETUNE}
)


@dataclass(frozen=True)
class ForegroundGateOutput:
    """模型前向输出：shared 前景 logit 与三粗类条件 logit。"""

    shared_logit: Any  # (B, 1)
    coarse_logits: Any  # (B, 3)，按 COARSE_CLASSES 顺序

    def fg_logit(self, coarse: str) -> Any:
        """返回某粗类的 ``shared + residual`` 前景 logit（S2 用）。"""

        index = COARSE_CLASSES.index(coarse)
        residual = self.coarse_logits[:, index : index + 1]
        return self.shared_logit + residual


def _freeze_backbone(model: Any) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False


def _freeze_first_three_stages(model: Any) -> None:
    """冻结 stem 与前三个 stage，只保留最后一个 stage 可训练。

    torchvision ConvNeXt 的 ``features`` 结构为
    ``[stem, stem_norm, stage1, stage2, stage3, stage4]``，最后一个元素即
    stage4。冻结除最后一个 stage 外的全部 feature 参数。
    """
    children = list(model.features.children())
    if len(children) < 2:
        raise ValueError("ConvNeXt features 结构异常")
    for child in children[:-1]:
        for parameter in child.parameters():
            parameter.requires_grad = False


def _apply_freeze(model: Any, freeze: str) -> None:
    if freeze == FREEZE_BACKBONE:
        _freeze_backbone(model)
    elif freeze == FREEZE_FIRST_THREE_STAGES:
        _freeze_first_three_stages(model)
    elif freeze == FULL_FINETUNE:
        for parameter in model.features.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"未知 freeze={freeze!r}")


def build_coarse_foreground_gate(
    *,
    weight_path: str | Mapping[str, str],
    freeze: str,
    verify_weight_sha256: bool = True,
    device: Any = None,
) -> Any:
    """构建粗类条件式前景门控模型。

    Args:
        weight_path: ImageNet ConvNeXt-T 权重路径；若为 Mapping，则键
            ``convnext`` 指向 ImageNet 权重，键 ``checkpoint`` 指向可选的
            N2-CFG 训练后 state_dict（含 heads）。
        freeze: 冻结策略，见 ALLOWED_FREEZE。
        verify_weight_sha256: 是否校验 ImageNet 权重 SHA（正式路径必须 True）。
        device: 可选，构建后移动到该设备。
    """
    if freeze not in ALLOWED_FREEZE:
        raise ValueError(f"freeze 必须是 {sorted(ALLOWED_FREEZE)}")

    try:
        import torch
        from torch import nn
        from torchvision.models import convnext_tiny
    except ImportError as error:
        raise RuntimeError(
            "N2-CFG 需要 PyTorch 2.5.1 与 torchvision 0.20.1"
        ) from error

    if isinstance(weight_path, Mapping):
        convnext_path = weight_path["convnext"]
        checkpoint_path = weight_path.get("checkpoint")
    else:
        convnext_path = weight_path
        checkpoint_path = None

    base = convnext_tiny(weights=None)
    if verify_weight_sha256:
        actual = sha256_file(convnext_path)
        if actual != CONVNEXT_TINY_WEIGHT_SHA256:
            raise ValueError(
                f"ConvNeXt-Tiny 权重 SHA256 不匹配: expected="
                f"{CONVNEXT_TINY_WEIGHT_SHA256}, actual={actual}"
            )
    try:
        state_dict = torch.load(convnext_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(convnext_path, map_location="cpu")
    base.load_state_dict(state_dict, strict=True)

    in_features = base.classifier[-1].in_features

    class CoarseForegroundGate(nn.Module):
        def __init__(self, backbone: Any, features_in: int) -> None:
            super().__init__()
            self.features = backbone.features
            self.avgpool = backbone.avgpool
            self.shared_head = nn.Linear(features_in, 1)
            self.coarse_heads = nn.ModuleDict(
                {name: nn.Linear(features_in, 1) for name in COARSE_CLASSES}
            )

        def forward(self, x: Any) -> ForegroundGateOutput:
            feature_map = self.features(x)
            pooled = self.avgpool(feature_map).flatten(1)
            shared = self.shared_head(pooled)
            coarse_logits = torch.cat(
                [self.coarse_heads[name](pooled) for name in COARSE_CLASSES],
                dim=1,
            )
            return ForegroundGateOutput(shared_logit=shared, coarse_logits=coarse_logits)

    model = CoarseForegroundGate(base, in_features)
    _apply_freeze(model, freeze)

    if checkpoint_path is not None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        model.load_state_dict(checkpoint, strict=True)

    if device is not None:
        model = model.to(device)
    return model


def parameter_summary(model: Any) -> dict[str, int]:
    """返回总参数与可训练参数。"""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


__all__ = [
    "ALLOWED_FREEZE",
    "FREEZE_BACKBONE",
    "FREEZE_FIRST_THREE_STAGES",
    "FULL_FINETUNE",
    "ForegroundGateOutput",
    "build_coarse_foreground_gate",
    "parameter_summary",
]
