"""Proposal-Aligned Verifier (PAV) used by HERA-Guard.

One shared ImageNet trunk processes a tight proposal crop and a context crop.
The verifier exposes foreground, coarse/fine identity, localization quality,
and TP-protection heads.  It never changes boxes or drops candidates itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rsdet.models.crop_classifier import CONVNEXT_TINY_WEIGHT_SHA256, sha256_file

try:  # Keep audit-only environments importable when model extras are absent.
    from torch import nn as _nn
except ImportError:  # pragma: no cover - exercised by environment gates

    class _ModuleBase:  # type: ignore[no-redef]
        pass
else:
    _ModuleBase = _nn.Module

FREEZE_BACKBONE = "freeze_backbone"
FREEZE_FIRST_STAGES = "freeze_first_stages"
FULL_FINETUNE = "full"
ALLOWED_FREEZE = frozenset({FREEZE_BACKBONE, FREEZE_FIRST_STAGES, FULL_FINETUNE})


@dataclass(frozen=True)
class ProposalVerifierOutput:
    foreground_logit: Any
    coarse_logits: Any
    fine_logits: Any
    quality_logit: Any
    protect_logit: Any
    active_fp_logit: Any
    embedding: Any


def _set_backbone_trainability(backbone: Any, freeze: str) -> None:
    if freeze not in ALLOWED_FREEZE:
        raise ValueError(f"freeze must be one of {sorted(ALLOWED_FREEZE)}")
    for parameter in backbone.parameters():
        parameter.requires_grad = freeze == FULL_FINETUNE
    if freeze == FREEZE_FIRST_STAGES:
        children = list(backbone.features.children())
        if not children:
            raise ValueError("backbone.features is empty")
        for parameter in children[-1].parameters():
            parameter.requires_grad = True


def build_convnext_tiny_backbone(
    *,
    weight_path: str | Path,
    freeze: str,
    verify_weight_sha256: bool = True,
) -> Any:
    """Load the frozen project ConvNeXt-Tiny asset without network access."""

    try:
        import torch
        from torchvision.models import convnext_tiny
    except ImportError as error:  # pragma: no cover - environment gate
        raise RuntimeError("PAV requires PyTorch and torchvision") from error

    path = Path(weight_path).expanduser().resolve()
    if verify_weight_sha256:
        actual = sha256_file(path)
        if actual != CONVNEXT_TINY_WEIGHT_SHA256:
            raise ValueError(
                "ConvNeXt-Tiny weight SHA256 mismatch: "
                f"expected={CONVNEXT_TINY_WEIGHT_SHA256}, actual={actual}"
            )
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - old torch compatibility
        state = torch.load(path, map_location="cpu")
    backbone = convnext_tiny(weights=None)
    backbone.load_state_dict(state, strict=True)
    _set_backbone_trainability(backbone, freeze)
    return backbone


def _infer_feature_dim(backbone: Any) -> int:
    classifier = getattr(backbone, "classifier", None)
    if classifier is None or not len(classifier):
        raise ValueError("backbone must expose a ConvNeXt-like classifier")
    for layer in reversed(list(classifier)):
        if hasattr(layer, "in_features"):
            return int(layer.in_features)
        if hasattr(layer, "normalized_shape"):
            shape = layer.normalized_shape
            return int(shape[-1] if isinstance(shape, (tuple, list)) else shape)
    raise ValueError("cannot infer backbone feature dimension")


class ProposalAlignedVerifier(_ModuleBase):
    """Two-view shared-trunk multi-task proposal verifier."""

    def __init__(
        self,
        backbone: Any,
        *,
        feature_dim: int | None = None,
        metadata_dim: int = 12,
        hidden_dim: int = 512,
        num_coarse_classes: int = 3,
        num_fine_classes: int = 25,
        dropout: float = 0.10,
    ) -> None:
        from torch import nn

        super().__init__()
        if metadata_dim < 0 or hidden_dim <= 0:
            raise ValueError("invalid PAV dimensions")
        self.backbone = backbone
        self.feature_dim = int(feature_dim or _infer_feature_dim(backbone))
        self.metadata_dim = int(metadata_dim)
        pair_dim = self.feature_dim * 4
        self.pair_projection = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.metadata_dim:
            metadata_hidden = max(hidden_dim // 4, 32)
            self.metadata_projection = nn.Sequential(
                nn.Linear(self.metadata_dim, metadata_hidden),
                nn.LayerNorm(metadata_hidden),
                nn.GELU(),
            )
        else:
            metadata_hidden = 0
            self.metadata_projection = None
        fused_dim = hidden_dim + metadata_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.foreground_head = nn.Linear(hidden_dim, 1)
        self.coarse_head = nn.Linear(hidden_dim, num_coarse_classes)
        self.fine_head = nn.Linear(hidden_dim, num_fine_classes)
        self.quality_head = nn.Linear(hidden_dim, 1)
        self.protect_head = nn.Linear(hidden_dim, 1)
        self.active_fp_head = nn.Linear(hidden_dim, 1)

    def _encode(self, images: Any) -> Any:
        feature_map = self.backbone.features(images)
        pooled = self.backbone.avgpool(feature_map)
        normalized = self.backbone.classifier[0](pooled)
        return normalized.flatten(1)

    def forward(
        self, tight: Any, context: Any, metadata: Any | None = None
    ) -> ProposalVerifierOutput:
        import torch

        if tight.shape != context.shape or tight.ndim != 4:
            raise ValueError("tight/context tensors must have identical BCHW shapes")
        combined = torch.cat([tight, context], dim=0)
        encoded = self._encode(combined)
        tight_feature, context_feature = encoded.chunk(2, dim=0)
        pair = torch.cat(
            [
                tight_feature,
                context_feature,
                tight_feature - context_feature,
                torch.abs(tight_feature - context_feature),
            ],
            dim=1,
        )
        pair_embedding = self.pair_projection(pair)
        if self.metadata_projection is not None:
            if metadata is None or metadata.shape != (tight.shape[0], self.metadata_dim):
                raise ValueError("metadata shape does not match PAV contract")
            pair_embedding = torch.cat([pair_embedding, self.metadata_projection(metadata)], dim=1)
        elif metadata is not None and metadata.numel():
            raise ValueError("metadata supplied to a metadata-free PAV")
        embedding = self.fusion(pair_embedding)
        return ProposalVerifierOutput(
            foreground_logit=self.foreground_head(embedding).squeeze(1),
            coarse_logits=self.coarse_head(embedding),
            fine_logits=self.fine_head(embedding),
            quality_logit=self.quality_head(embedding).squeeze(1),
            protect_logit=self.protect_head(embedding).squeeze(1),
            active_fp_logit=self.active_fp_head(embedding).squeeze(1),
            embedding=embedding,
        )


def build_proposal_aligned_verifier(
    *,
    convnext_weight_path: str | Path,
    freeze: str,
    metadata_dim: int = 12,
    hidden_dim: int = 512,
    checkpoint_path: str | Path | None = None,
    verify_weight_sha256: bool = True,
    device: Any = None,
) -> ProposalAlignedVerifier:
    """Build PAV and optionally restore a strict HERA checkpoint."""

    import torch

    backbone = build_convnext_tiny_backbone(
        weight_path=convnext_weight_path,
        freeze=freeze,
        verify_weight_sha256=verify_weight_sha256,
    )
    model = ProposalAlignedVerifier(
        backbone,
        metadata_dim=metadata_dim,
        hidden_dim=hidden_dim,
    )
    if checkpoint_path is not None:
        try:
            checkpoint: Mapping[str, Any] = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        except TypeError:  # pragma: no cover
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state, strict=True)
    if device is not None:
        model = model.to(device)
    return model


__all__ = [
    "ALLOWED_FREEZE",
    "FREEZE_BACKBONE",
    "FREEZE_FIRST_STAGES",
    "FULL_FINETUNE",
    "ProposalAlignedVerifier",
    "ProposalVerifierOutput",
    "build_convnext_tiny_backbone",
    "build_proposal_aligned_verifier",
]
