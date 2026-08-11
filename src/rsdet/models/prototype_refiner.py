"""HPR：面向不均衡小样本细类的层次化原型重识别分支。

该网络由本项目实现，不依赖 torchvision 或 Ultralytics。检测器负责定位，HPR 对
候选框放大裁剪后进行 25 类细粒度判别；余弦分类器、EMA 类原型和大类辅助监督共同
降低少样本类别因分类器权重范数不足造成的偏置。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import torch
    import torch.nn.functional as F
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError("HPR 原型分支需要 PyTorch，请切换到 pytorch 环境")


if _TORCH_AVAILABLE:

    class ConvNormAct(nn.Sequential):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            *,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1,
            activate: bool = True,
        ) -> None:
            padding = dilation * (kernel_size - 1) // 2
            layers: list[nn.Module] = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            ]
            if activate:
                layers.append(nn.SiLU(inplace=True))
            super().__init__(*layers)

    class DepthwiseSeparableConv(nn.Sequential):
        def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
            super().__init__(
                ConvNormAct(
                    in_channels,
                    in_channels,
                    3,
                    stride=stride,
                    groups=in_channels,
                ),
                ConvNormAct(in_channels, out_channels, 1),
            )

    class MultiScaleDetailBlock(nn.Module):
        """局部纹理与空洞上下文双支路，保留飞机/舰船细粒度结构。"""

        def __init__(self, channels: int) -> None:
            super().__init__()
            if channels % 2:
                raise ValueError("MultiScaleDetailBlock 的 channels 必须为偶数")
            half = channels // 2
            self.pre = ConvNormAct(channels, channels, 1)
            self.local = ConvNormAct(half, half, 3, groups=half)
            self.context = ConvNormAct(half, half, 3, groups=half, dilation=2)
            hidden = max(8, channels // 8)
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, channels, 1),
                nn.Sigmoid(),
            )
            self.project = ConvNormAct(channels, channels, 1, activate=False)
            self.activate = nn.SiLU(inplace=True)

        def forward(self, inputs):
            first, second = self.pre(inputs).chunk(2, dim=1)
            features = torch.cat((self.local(first), self.context(second)), dim=1)
            features = self.project(features * self.channel_gate(features))
            return self.activate(inputs + features)

    class CosineClassifier(nn.Module):
        """消除类别权重范数偏置的归一化分类器。"""

        def __init__(self, embedding_dim: int, num_classes: int, scale: float = 20.0) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
            self.log_scale = nn.Parameter(torch.tensor(float(scale)).log())
            nn.init.normal_(self.weight, std=0.01)

        def forward(self, embeddings):
            scale = self.log_scale.exp().clamp(1.0, 100.0)
            return scale * F.linear(F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1))

    class HierarchicalPrototypeRefiner(nn.Module):
        """轻量多尺度纹理编码器 + 25 类余弦头 + 三大类辅助头 + EMA 原型。"""

        def __init__(
            self,
            *,
            num_classes: int = 25,
            embedding_dim: int = 128,
            dropout: float = 0.1,
            prototype_momentum: float = 0.9,
        ) -> None:
            super().__init__()
            if num_classes != 25:
                raise ValueError("XH HPR 当前固定为 25 个官方细类")
            if embedding_dim <= 0:
                raise ValueError("embedding_dim 必须 > 0")
            if not 0.0 <= prototype_momentum < 1.0:
                raise ValueError("prototype_momentum 必须在 [0, 1)")
            self.num_classes = int(num_classes)
            self.embedding_dim = int(embedding_dim)
            self.prototype_momentum = float(prototype_momentum)
            self.encoder = nn.Sequential(
                ConvNormAct(3, 32, 3, stride=2),
                DepthwiseSeparableConv(32, 48, stride=2),
                MultiScaleDetailBlock(48),
                DepthwiseSeparableConv(48, 72, stride=2),
                MultiScaleDetailBlock(72),
                DepthwiseSeparableConv(72, 96, stride=2),
                MultiScaleDetailBlock(96),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.embedding_head = nn.Sequential(
                nn.Linear(96, embedding_dim, bias=False),
                nn.LayerNorm(embedding_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.fine_classifier = CosineClassifier(embedding_dim, num_classes)
            self.coarse_classifier = nn.Linear(embedding_dim, 3)
            self.register_buffer("prototypes", torch.zeros(num_classes, embedding_dim))
            self.register_buffer("prototype_counts", torch.zeros(num_classes, dtype=torch.long))
            self.register_buffer(
                "fine_to_coarse",
                torch.tensor([0] * 4 + [1] * 20 + [2], dtype=torch.long),
            )

        def forward(self, images) -> dict[str, Any]:
            embeddings = F.normalize(self.embedding_head(self.encoder(images)), dim=1)
            fine_logits = self.fine_classifier(embeddings)
            scale = self.fine_classifier.log_scale.exp().clamp(1.0, 100.0)
            prototype_logits = scale * embeddings @ F.normalize(self.prototypes, dim=1).T
            available = self.prototype_counts > 0
            prototype_logits = prototype_logits.masked_fill(~available.unsqueeze(0), -1e4)
            return {
                "embeddings": embeddings,
                "fine_logits": fine_logits,
                "coarse_logits": self.coarse_classifier(embeddings),
                "prototype_logits": prototype_logits,
                "prototype_available": available,
            }

        def fused_logits(self, outputs: Mapping[str, Any], *, prototype_weight: float = 0.35):
            if not 0.0 <= prototype_weight <= 1.0:
                raise ValueError("prototype_weight 必须在 [0, 1]")
            fine_logits = outputs["fine_logits"]
            prototype_logits = outputs["prototype_logits"]
            available = outputs["prototype_available"].unsqueeze(0)
            fused = (1.0 - prototype_weight) * fine_logits + prototype_weight * prototype_logits
            return torch.where(available, fused, fine_logits)

        @torch.no_grad()
        def update_prototypes(self, embeddings, labels) -> None:
            embeddings = F.normalize(embeddings.detach(), dim=1)
            for class_id in labels.unique():
                class_index = int(class_id.item())
                class_mean = F.normalize(embeddings[labels == class_id].mean(dim=0), dim=0)
                batch_count = int((labels == class_id).sum().item())
                if self.prototype_counts[class_index] == 0:
                    updated = class_mean
                else:
                    updated = F.normalize(
                        self.prototype_momentum * self.prototypes[class_index]
                        + (1.0 - self.prototype_momentum) * class_mean,
                        dim=0,
                    )
                self.prototypes[class_index].copy_(updated)
                self.prototype_counts[class_index].add_(batch_count)

    class HierarchicalPrototypeLoss(nn.Module):
        """Class-balanced focal + LDAM + 大类辅助 + 原型/批内紧致约束。"""

        def __init__(
            self,
            class_counts: Sequence[int],
            *,
            effective_beta: float = 0.9999,
            max_class_weight: float = 3.0,
            focal_gamma: float = 1.5,
            max_margin: float = 0.3,
            coarse_weight: float = 0.25,
            compact_weight: float = 0.1,
            prototype_weight: float = 0.1,
        ) -> None:
            super().__init__()
            counts = torch.as_tensor(list(class_counts), dtype=torch.float32)
            if counts.numel() != 25 or torch.any(counts <= 0):
                raise ValueError("class_counts 必须包含 25 个正数")
            if not 0.0 <= effective_beta < 1.0:
                raise ValueError("effective_beta 必须在 [0, 1)")
            effective = (1.0 - effective_beta) / (1.0 - effective_beta**counts)
            weights = effective / effective.mean()
            weights = weights.clamp(max=max_class_weight)
            margins = counts.pow(-0.25)
            margins = margins / margins.max() * max_margin
            self.register_buffer("class_weights", weights)
            self.register_buffer("margins", margins)
            self.register_buffer(
                "fine_to_coarse",
                torch.tensor([0] * 4 + [1] * 20 + [2], dtype=torch.long),
            )
            self.focal_gamma = float(focal_gamma)
            self.coarse_weight = float(coarse_weight)
            self.compact_weight = float(compact_weight)
            self.prototype_weight = float(prototype_weight)

        @staticmethod
        def _compactness(embeddings, labels):
            losses = []
            for class_id in labels.unique():
                members = embeddings[labels == class_id]
                if len(members) < 2:
                    continue
                center = F.normalize(members.mean(dim=0), dim=0)
                losses.append((1.0 - members @ center).mean())
            return torch.stack(losses).mean() if losses else embeddings.sum() * 0.0

        def forward(self, outputs: Mapping[str, Any], labels) -> dict[str, Any]:
            fine_logits = outputs["fine_logits"]
            adjusted = fine_logits.clone()
            row_indices = torch.arange(len(labels), device=labels.device)
            adjusted[row_indices, labels] -= self.margins[labels]
            per_sample = F.cross_entropy(
                adjusted,
                labels,
                weight=self.class_weights,
                reduction="none",
            )
            target_probabilities = fine_logits.softmax(dim=1)[row_indices, labels]
            fine_loss = ((1.0 - target_probabilities) ** self.focal_gamma * per_sample).mean()
            coarse_labels = self.fine_to_coarse[labels]
            coarse_loss = F.cross_entropy(outputs["coarse_logits"], coarse_labels)
            compact_loss = self._compactness(outputs["embeddings"], labels)

            available = outputs["prototype_available"][labels]
            if torch.any(available):
                prototype_loss = F.cross_entropy(
                    outputs["prototype_logits"][available], labels[available]
                )
            else:
                prototype_loss = fine_logits.sum() * 0.0
            total = (
                fine_loss
                + self.coarse_weight * coarse_loss
                + self.compact_weight * compact_loss
                + self.prototype_weight * prototype_loss
            )
            return {
                "loss": total,
                "fine_loss": fine_loss.detach(),
                "coarse_loss": coarse_loss.detach(),
                "compact_loss": compact_loss.detach(),
                "prototype_loss": prototype_loss.detach(),
            }

else:

    class HierarchicalPrototypeRefiner:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()

    class HierarchicalPrototypeLoss:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            _require_torch()


def load_refiner_checkpoint(path: str | Path, *, map_location: str = "cpu"):
    """加载包含结构配置、类别计数和原型的 HPR checkpoint。"""
    _require_torch()
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"HPR checkpoint 不存在: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Mapping) or "model_state" not in checkpoint:
        raise ValueError("HPR checkpoint 必须包含 model_state")
    model_config = dict(checkpoint.get("model_config", {}))
    model = HierarchicalPrototypeRefiner(**model_config)
    model.load_state_dict(checkpoint["model_state"])
    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"model_state", "optimizer_state", "scheduler_state"}
    }
    return model, metadata


__all__ = [
    "HierarchicalPrototypeLoss",
    "HierarchicalPrototypeRefiner",
    "load_refiner_checkpoint",
]
