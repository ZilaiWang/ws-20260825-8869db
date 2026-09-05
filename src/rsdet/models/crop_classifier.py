"""P0-3 crop 分类器构建函数。

所有 PyTorch/torchvision 导入均延迟到函数内，保持仓库核心模块在
无深度学习环境也可导入和测试。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

CONVNEXT_TINY_WEIGHT_SHA256 = "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_convnext_tiny_classifier(
    num_classes: int,
    *,
    weight_path: str | Path,
    regime: str,
    verify_weight_sha256: bool = True,
) -> Any:
    """从显式本地权重构建 ConvNeXt-Tiny，不在运行中隐式下载。"""

    if num_classes <= 1:
        raise ValueError("num_classes 必须大于 1")
    if regime not in {"linear_probe", "fine_tune"}:
        raise ValueError("regime 必须是 linear_probe 或 fine_tune")
    path = Path(weight_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ImageNet 预训练权重不存在: {path}")
    if verify_weight_sha256:
        actual = sha256_file(path)
        if actual != CONVNEXT_TINY_WEIGHT_SHA256:
            raise ValueError(
                f"ConvNeXt-Tiny 权重 SHA256 不匹配: expected="
                f"{CONVNEXT_TINY_WEIGHT_SHA256}, actual={actual}"
            )

    import torch

    model = build_convnext_tiny_architecture(num_classes, regime=regime)
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    return model


def build_convnext_tiny_architecture(num_classes: int, *, regime: str) -> Any:
    """Build the architecture without loading an ImageNet state dictionary.

    This is the deployment path for audited checkpoints that contain the full
    model state.  Keeping it separate prevents Docker startup from loading a
    109 MB ImageNet file only to overwrite every tensor immediately.
    """

    if num_classes <= 1:
        raise ValueError("num_classes 必须大于 1")
    if regime not in {"linear_probe", "fine_tune"}:
        raise ValueError("regime 必须是 linear_probe 或 fine_tune")
    try:
        from torch import nn
        from torchvision.models import convnext_tiny
    except ImportError as error:
        raise RuntimeError(
            "P0-3 需要 PyTorch 2.5.1 和 torchvision 0.20.1；"
            "请按 docs/server/P03_TASK_01_ENV_AND_LINEAR_PROBE.md 安装"
        ) from error

    model = convnext_tiny(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    trainable = regime == "fine_tune"
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    if regime == "linear_probe":
        for parameter in model.classifier[-1].parameters():
            parameter.requires_grad = True
    return model


def parameter_summary(model: Any) -> dict[str, int]:
    """返回总参数和可训练参数。"""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}
