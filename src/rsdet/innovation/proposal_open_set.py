"""Proposal-domain open-set evidence for HERA-Guard V5.

The verifier distinguishes a real foreground proposal from two materially
different negative modes.  Keeping structured and ordinary background
separate is the registered difference from the earlier binary F1 experiment.
The output is evidence for the official-match quality head; it is never a hard
drop/relabel decision by itself.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from rsdet.analysis.proposal_open_set import (
    OPEN_FOREGROUND,
    OPEN_IGNORE,
    OPEN_LABEL_NAMES,
    OPEN_ORDINARY_BACKGROUND,
    OPEN_STRUCTURED_BACKGROUND,
    STRUCTURED_CONFUSER_CLASSES,
    proposal_open_set_label,
)


class ProposalOpenSetHead(nn.Module):
    """Low-capacity head over frozen tight/context crop embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or hidden_dim < 16:
            raise ValueError("embedding_dim must be positive and hidden_dim >= 16")
        self.embedding_dim = int(embedding_dim)
        self.input_dim = 2 * int(embedding_dim) + 3
        self.network = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(OPEN_LABEL_NAMES)),
        )

    def forward(
        self,
        tight_embedding: torch.Tensor,
        context_embedding: torch.Tensor,
        coarse_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        tensors: Sequence[torch.Tensor] = (
            tight_embedding,
            context_embedding,
            coarse_one_hot,
        )
        if any(value.ndim != 2 for value in tensors):
            raise ValueError("all open-set inputs must be matrices")
        if len({int(value.shape[0]) for value in tensors}) != 1:
            raise ValueError("open-set inputs must have identical row counts")
        if (
            int(tight_embedding.shape[1]) != self.embedding_dim
            or context_embedding.shape != tight_embedding.shape
            or int(coarse_one_hot.shape[1]) != 3
        ):
            raise ValueError("open-set input dimensions do not match the contract")
        return self.network(torch.cat(tensors, dim=1))


__all__ = [
    "OPEN_FOREGROUND",
    "OPEN_IGNORE",
    "OPEN_LABEL_NAMES",
    "OPEN_ORDINARY_BACKGROUND",
    "OPEN_STRUCTURED_BACKGROUND",
    "ProposalOpenSetHead",
    "STRUCTURED_CONFUSER_CLASSES",
    "proposal_open_set_label",
]
