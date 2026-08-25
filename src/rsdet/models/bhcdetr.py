"""Paper-aligned BHC-DETR network.

The implementation follows Chen et al., *Balanced Hierarchical Contrastive
Learning with Decoupled Queries for Fine-grained Object Detection in Remote
Sensing Images* (CVPR 2026): object queries are split into classification and
localization streams, aligned by joint self-attention, and refined by separate
cross-attention/FFN branches.  BHCL itself lives in :mod:`rsdet.models.bhcl`.

The competition only provides horizontal boxes, so the paper's rotated box
head is replaced by a four-parameter normalized ``cxcywh`` head.  No other
detector family is used by this module.

The opt-in ``uhr_enabled`` path adds the Gain Map/LPM/ISSGA and bounded C4
local-memory adaptation documented in :mod:`rsdet.models.uhr_small_object`.
It deliberately preserves the BHC query/loss design and is not presented as a
full-image UHR-DETR reproduction.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rsdet.models.uhr_small_object import (
    GainMapHead,
    gain_map_expectation,
    select_sparse_local_tokens,
)


@dataclass(frozen=True)
class BHCDetrConfig:
    """Serializable architecture configuration stored in every checkpoint."""

    num_classes: int = 25
    image_size: int = 1024
    backbone: str = "resnet50"
    backbone_pretrained: bool = True
    backbone_weights: str | None = None
    train_backbone: bool = True
    hidden_dim: int = 256
    num_queries: int = 300
    encoder_layers: int = 6
    decoder_layers: int = 6
    nheads: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    projection_dim: int = 128
    prior_probability: float = 0.01
    # Project-compatible adaptation of UHR-DETR (arXiv:2604.21435v1).  It is
    # opt-in so checkpoints trained before the small-object extension remain
    # loadable without silently changing their architecture.
    uhr_enabled: bool = False
    uhr_gain_bin_limit: int = 6
    uhr_patch_size: int = 512
    uhr_patch_budget: int = 4
    uhr_max_local_tokens: int = 512
    uhr_gain_head_groups: int = 32

    def __post_init__(self) -> None:
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.backbone != "resnet50":
            raise ValueError("the paper configuration currently supports backbone=resnet50")
        if isinstance(self.nheads, bool) or not isinstance(self.nheads, int) or self.nheads <= 0:
            raise ValueError("nheads must be a positive integer")
        if (
            isinstance(self.hidden_dim, bool)
            or not isinstance(self.hidden_dim, int)
            or self.hidden_dim <= 0
            or self.hidden_dim % self.nheads
        ):
            raise ValueError("hidden_dim must be positive and divisible by nheads")
        if self.hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by 4 for 2D sine encoding")
        if min(self.num_queries, self.encoder_layers, self.decoder_layers) <= 0:
            raise ValueError("num_queries/encoder_layers/decoder_layers must be positive")
        if self.dim_feedforward <= 0 or self.projection_dim <= 0:
            raise ValueError("dim_feedforward/projection_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 < self.prior_probability < 1.0:
            raise ValueError("prior_probability must be in (0, 1)")
        if type(self.uhr_enabled) is not bool:
            raise ValueError("uhr_enabled must be bool")
        for name, value in (
            ("uhr_gain_bin_limit", self.uhr_gain_bin_limit),
            ("uhr_patch_size", self.uhr_patch_size),
            ("uhr_patch_budget", self.uhr_patch_budget),
            ("uhr_max_local_tokens", self.uhr_max_local_tokens),
            ("uhr_gain_head_groups", self.uhr_gain_head_groups),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.uhr_enabled and self.uhr_patch_size > self.image_size:
            raise ValueError("uhr_patch_size cannot exceed image_size")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "BHCDetrConfig":
        raw = dict(value or {})
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown BHCDetr model fields: {unknown}")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MLP(nn.Module):
    """Small feed-forward head used for box regression and projection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        dimensions = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1]) for index in range(num_layers)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        value = inputs
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index + 1 < len(self.layers):
                value = F.relu(value)
        return value


class PositionEmbeddingSine(nn.Module):
    """Standard normalized two-dimensional sine/cosine positional encoding."""

    def __init__(self, hidden_dim: int, temperature: float = 10000.0) -> None:
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even for 2D positional encoding")
        self.num_pos_feats = hidden_dim // 2
        self.temperature = float(temperature)

    def forward(self, padding_mask: Tensor) -> Tensor:
        if padding_mask.ndim != 3 or padding_mask.dtype is not torch.bool:
            raise ValueError("padding_mask must be bool [B,H,W]")
        not_mask = ~padding_mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * (2.0 * math.pi)
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * (2.0 * math.pi)
        dim_t = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
            device=padding_mask.device,
        )
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4)
        pos_x = pos_x.flatten(3)
        pos_y = pos_y.flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class ResNet50Backbone(nn.Module):
    """ResNet-50 C5 feature extractor with frozen batch-normalization stats."""

    output_channels = 2048

    def __init__(
        self,
        *,
        pretrained: bool,
        weights_path: str | None,
        train_backbone: bool,
    ) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet50_Weights, resnet50
        except (ImportError, OSError) as error:
            raise ImportError(
                "BHC-DETR requires a working torchvision installation; see requirements-model.txt"
            ) from error

        weights = None
        if pretrained and not weights_path:
            weights = ResNet50_Weights.DEFAULT
        network = resnet50(weights=weights)
        if weights_path:
            checkpoint = torch.load(
                Path(weights_path).expanduser(),
                map_location="cpu",
                weights_only=False,
            )
            state = (
                checkpoint.get("state_dict", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )
            if not isinstance(state, Mapping):
                raise TypeError("backbone checkpoint must contain a state_dict mapping")
            normalized = {
                str(key).removeprefix("module.").removeprefix("backbone."): value
                for key, value in state.items()
            }
            network.load_state_dict(normalized, strict=True)

        self.body = nn.Sequential(
            network.conv1,
            network.bn1,
            network.relu,
            network.maxpool,
            network.layer1,
            network.layer2,
            network.layer3,
            network.layer4,
        )
        if not train_backbone:
            for parameter in self.body.parameters():
                parameter.requires_grad_(False)
        else:
            # The small per-GPU batch used for 1024px imagery makes updating BN
            # statistics unstable.  Affine parameters and stats are both frozen.
            for module in self.body.modules():
                if isinstance(module, nn.BatchNorm2d):
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "ResNet50Backbone":
        super().train(mode)
        for module in self.body.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        return self.body(images)

    def forward_multiscale(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Return C4/stride-16 and C5/stride-32 without changing state keys."""

        features = images
        c4: Tensor | None = None
        for index, module in enumerate(self.body):
            features = module(features)
            # Sequential entries: conv1, bn1, relu, maxpool, layer1, layer2,
            # layer3, layer4.  layer3 is the C4 feature map.
            if index == 6:
                c4 = features
        if c4 is None:  # pragma: no cover - protects future backbone refactors
            raise RuntimeError("ResNet-50 backbone did not produce C4 features")
        return c4, features


class EncoderLayer(nn.Module):
    def __init__(self, config: BHCDetrConfig) -> None:
        super().__init__()
        dimension = config.hidden_dim
        self.self_attn = nn.MultiheadAttention(
            dimension,
            config.nheads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(dimension, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, dimension)
        self.dropout = nn.Dropout(config.dropout)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)

    def forward(
        self,
        source: Tensor,
        position: Tensor,
        padding_mask: Tensor | None,
    ) -> Tensor:
        query = key = source + position
        attended = self.self_attn(
            query,
            key,
            source,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        source = self.norm1(source + self.dropout1(attended))
        feedforward = self.linear2(self.dropout(F.relu(self.linear1(source))))
        return self.norm2(source + self.dropout2(feedforward))


class TransformerEncoder(nn.Module):
    def __init__(self, config: BHCDetrConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(EncoderLayer(config) for _ in range(config.encoder_layers))

    def forward(
        self,
        source: Tensor,
        position: Tensor,
        padding_mask: Tensor | None,
    ) -> Tensor:
        output = source
        for layer in self.layers:
            output = layer(output, position, padding_mask)
        return output


class TaskSpecificDecoderBranch(nn.Module):
    """Task-specific global/local cross-attention followed by one FFN.

    With ``uhr_enabled=False`` this is exactly the original BHC-DETR branch.
    With the small-object extension enabled, the existing cross-attention is
    the global macro-interaction and a second attention consumes bounded C4
    tokens selected by Gain Map + ISSGA.
    """

    def __init__(self, config: BHCDetrConfig) -> None:
        super().__init__()
        dimension = config.hidden_dim
        self.cross_attn = nn.MultiheadAttention(
            dimension,
            config.nheads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(dimension, config.dim_feedforward)
        self.linear2 = nn.Linear(config.dim_feedforward, dimension)
        self.dropout = nn.Dropout(config.dropout)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.local_cross_attn: nn.MultiheadAttention | None = None
        self.local_dropout: nn.Dropout | None = None
        self.local_norm: nn.LayerNorm | None = None
        if config.uhr_enabled:
            self.local_cross_attn = nn.MultiheadAttention(
                dimension,
                config.nheads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.local_dropout = nn.Dropout(config.dropout)
            self.local_norm = nn.LayerNorm(dimension)

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        memory_position: Tensor,
        memory_padding_mask: Tensor | None,
        local_memory: Tensor | None = None,
        local_position: Tensor | None = None,
        local_padding_mask: Tensor | None = None,
    ) -> Tensor:
        attended = self.cross_attn(
            queries,
            memory + memory_position,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )[0]
        queries = self.norm1(queries + self.dropout1(attended))
        if self.local_cross_attn is not None:
            if local_memory is None or local_position is None:
                raise ValueError("UHR decoder requires local memory and position")
            local_attended = self.local_cross_attn(
                queries,
                local_memory + local_position,
                local_memory,
                key_padding_mask=local_padding_mask,
                need_weights=False,
            )[0]
            if self.local_dropout is None or self.local_norm is None:  # pragma: no cover
                raise RuntimeError("local decoder branch is incompletely initialized")
            queries = self.local_norm(queries + self.local_dropout(local_attended))
        feedforward = self.linear2(self.dropout(F.relu(self.linear1(queries))))
        return self.norm2(queries + self.dropout2(feedforward))


class DecoupledDecoderLayer(nn.Module):
    """Joint self-attention followed by decoupled task-specific branches."""

    def __init__(self, config: BHCDetrConfig) -> None:
        super().__init__()
        joint_dimension = 2 * config.hidden_dim
        self.hidden_dim = config.hidden_dim
        self.self_attn = nn.MultiheadAttention(
            joint_dimension,
            config.nheads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.self_dropout = nn.Dropout(config.dropout)
        self.self_norm = nn.LayerNorm(joint_dimension)
        self.classification_branch = TaskSpecificDecoderBranch(config)
        self.localization_branch = TaskSpecificDecoderBranch(config)

    def forward(
        self,
        classification_queries: Tensor,
        localization_queries: Tensor,
        memory: Tensor,
        memory_position: Tensor,
        memory_padding_mask: Tensor | None,
        local_memory: Tensor | None = None,
        local_position: Tensor | None = None,
        local_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        joint = torch.cat((classification_queries, localization_queries), dim=-1)
        aligned = self.self_attn(joint, joint, joint, need_weights=False)[0]
        joint = self.self_norm(joint + self.self_dropout(aligned))
        classification_queries, localization_queries = joint.split(self.hidden_dim, dim=-1)
        classification_queries = self.classification_branch(
            classification_queries,
            memory,
            memory_position,
            memory_padding_mask,
            local_memory,
            local_position,
            local_padding_mask,
        )
        localization_queries = self.localization_branch(
            localization_queries,
            memory,
            memory_position,
            memory_padding_mask,
            local_memory,
            local_position,
            local_padding_mask,
        )
        return classification_queries, localization_queries


class DecoupledTransformerDecoder(nn.Module):
    def __init__(self, config: BHCDetrConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DecoupledDecoderLayer(config) for _ in range(config.decoder_layers)
        )

    def forward(
        self,
        classification_queries: Tensor,
        localization_queries: Tensor,
        memory: Tensor,
        memory_position: Tensor,
        memory_padding_mask: Tensor | None,
        local_memory: Tensor | None = None,
        local_position: Tensor | None = None,
        local_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        classification_outputs: list[Tensor] = []
        localization_outputs: list[Tensor] = []
        for layer in self.layers:
            classification_queries, localization_queries = layer(
                classification_queries,
                localization_queries,
                memory,
                memory_position,
                memory_padding_mask,
                local_memory,
                local_position,
                local_padding_mask,
            )
            classification_outputs.append(classification_queries)
            localization_outputs.append(localization_queries)
        return torch.stack(classification_outputs), torch.stack(localization_outputs)


class BHCDetr(nn.Module):
    """ResNet-50 DETR with the paper's decoupled-query decoder."""

    def __init__(self, config: BHCDetrConfig | Mapping[str, Any] | None = None) -> None:
        super().__init__()
        self.config = (
            config if isinstance(config, BHCDetrConfig) else BHCDetrConfig.from_mapping(config)
        )
        cfg = self.config
        self.backbone = ResNet50Backbone(
            pretrained=cfg.backbone_pretrained,
            weights_path=cfg.backbone_weights,
            train_backbone=cfg.train_backbone,
        )
        self.input_projection = nn.Conv2d(self.backbone.output_channels, cfg.hidden_dim, 1)
        self.local_projection: nn.Conv2d | None = None
        self.gain_map_head: GainMapHead | None = None
        if cfg.uhr_enabled:
            # ResNet-50 C4 has 1024 channels and stride 16.  Only a bounded
            # ISSGA-selected subset is exposed to decoder attention.
            self.local_projection = nn.Conv2d(1024, cfg.hidden_dim, 1)
            self.gain_map_head = GainMapHead(
                cfg.hidden_dim,
                bin_limit=cfg.uhr_gain_bin_limit,
                group_norm_groups=cfg.uhr_gain_head_groups,
            )
        self.position_embedding = PositionEmbeddingSine(cfg.hidden_dim)
        self.encoder = TransformerEncoder(cfg)
        self.classification_query_embedding = nn.Embedding(cfg.num_queries, cfg.hidden_dim)
        self.localization_query_embedding = nn.Embedding(cfg.num_queries, cfg.hidden_dim)
        self.decoder = DecoupledTransformerDecoder(cfg)
        self.classifier = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.box_regressor = MLP(cfg.hidden_dim, cfg.hidden_dim, 4, 3)
        self.projection_head = MLP(cfg.hidden_dim, cfg.hidden_dim, cfg.projection_dim, 2)

        prior_bias = -math.log((1.0 - cfg.prior_probability) / cfg.prior_probability)
        nn.init.constant_(self.classifier.bias, prior_bias)
        nn.init.normal_(self.classification_query_embedding.weight, std=0.02)
        nn.init.normal_(self.localization_query_embedding.weight, std=0.02)
        nn.init.zeros_(self.box_regressor.layers[-1].weight)
        nn.init.zeros_(self.box_regressor.layers[-1].bias)

    def forward(
        self,
        images: Tensor,
        padding_masks: Tensor | None = None,
        *,
        return_aux: bool = True,
    ) -> dict[str, Any]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must be float [B,3,H,W]")
        batch_size, _, height, width = images.shape
        if padding_masks is None:
            padding_masks = torch.zeros(
                (batch_size, height, width),
                dtype=torch.bool,
                device=images.device,
            )
        if padding_masks.shape != (batch_size, height, width):
            raise ValueError("padding_masks shape must match images spatial dimensions")

        if self.config.uhr_enabled:
            local_features, global_features = self.backbone.forward_multiscale(images)
        else:
            local_features = None
            global_features = self.backbone(images)
        projected = self.input_projection(global_features)
        feature_mask = F.interpolate(
            padding_masks[:, None].float(),
            size=projected.shape[-2:],
            mode="nearest",
        ).to(torch.bool)[:, 0]
        position_map = self.position_embedding(feature_mask).to(projected.dtype)
        source = projected.flatten(2).transpose(1, 2)
        position = position_map.flatten(2).transpose(1, 2)
        flat_mask = feature_mask.flatten(1)
        memory = self.encoder(source, position, flat_mask)

        local_memory: Tensor | None = None
        local_position: Tensor | None = None
        local_flat_mask: Tensor | None = None
        gain_logits: Tensor | None = None
        gain_map: Tensor | None = None
        gain_valid_mask: Tensor | None = None
        routing: Tensor | None = None
        gain_patch_fraction: tuple[float, float] | None = None
        if self.config.uhr_enabled:
            if (
                local_features is None
                or self.local_projection is None
                or self.gain_map_head is None
            ):  # pragma: no cover - initialization invariant
                raise RuntimeError("UHR small-object modules are incompletely initialized")
            gain_logits = self.gain_map_head(projected)
            # Squared-bin expectation is kept in float32 under AMP so close
            # small-object gains do not collapse through fp16 quantization.
            gain_map = gain_map_expectation(gain_logits.float())
            gain_valid_mask = ~feature_mask
            local_projected = self.local_projection(local_features)
            local_feature_mask = F.interpolate(
                padding_masks[:, None].float(),
                size=local_projected.shape[-2:],
                mode="nearest",
            ).to(torch.bool)[:, 0]
            local_position_map = self.position_embedding(local_feature_mask).to(
                local_projected.dtype
            )
            gain_patch_fraction = (
                min(1.0, self.config.uhr_patch_size / float(height)),
                min(1.0, self.config.uhr_patch_size / float(width)),
            )
            (
                local_memory,
                local_position,
                local_flat_mask,
                routing,
            ) = select_sparse_local_tokens(
                local_projected,
                local_position_map,
                local_feature_mask,
                gain_map,
                gain_valid_mask=gain_valid_mask,
                patch_fraction=gain_patch_fraction,
                patch_budget=self.config.uhr_patch_budget,
                max_tokens=self.config.uhr_max_local_tokens,
            )

        classification_queries = self.classification_query_embedding.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        localization_queries = self.localization_query_embedding.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        classification_layers, localization_layers = self.decoder(
            classification_queries,
            localization_queries,
            memory,
            position,
            flat_mask,
            local_memory,
            local_position,
            local_flat_mask,
        )
        if not return_aux:
            result: dict[str, Any] = {
                "pred_logits": self.classifier(classification_layers[-1]),
                "pred_boxes": self.box_regressor(localization_layers[-1]).sigmoid(),
            }
            if gain_logits is not None:
                result.update(
                    {
                        "gain_logits": gain_logits,
                        "gain_map": gain_map,
                        "gain_valid_mask": gain_valid_mask,
                        "gain_patch_fraction": gain_patch_fraction,
                        "routing": routing,
                    }
                )
            return result

        logits = self.classifier(classification_layers)
        boxes = self.box_regressor(localization_layers).sigmoid()
        projected_queries = F.normalize(
            self.projection_head(classification_layers),
            dim=-1,
        )
        auxiliary = [
            {
                "pred_logits": logits[index],
                "pred_boxes": boxes[index],
                "projected_queries": projected_queries[index],
            }
            for index in range(logits.shape[0] - 1)
        ]
        result = {
            "pred_logits": logits[-1],
            "pred_boxes": boxes[-1],
            "projected_queries": projected_queries[-1],
            "aux_outputs": auxiliary,
        }
        if gain_logits is not None:
            result.update(
                {
                    "gain_logits": gain_logits,
                    "gain_map": gain_map,
                    "gain_valid_mask": gain_valid_mask,
                    "gain_patch_fraction": gain_patch_fraction,
                    "routing": routing,
                }
            )
        return result


def build_bhcdetr(config: Mapping[str, Any] | BHCDetrConfig | None = None) -> BHCDetr:
    return BHCDetr(config)


__all__ = [
    "BHCDetr",
    "BHCDetrConfig",
    "DecoupledDecoderLayer",
    "MLP",
    "build_bhcdetr",
]
