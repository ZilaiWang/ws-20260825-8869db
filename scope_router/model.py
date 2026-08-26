from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class RelationalAttentionBlock(nn.Module):
    """Self-attention with learned pairwise relation bias.

    rel_x[b, i, j] can contain IoU, normalized center distance, log-scale ratio,
    same-class/coarse-family flags, score/rank deltas, and head agreement.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        relation_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.relation_bias = nn.Sequential(
            nn.Linear(relation_dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_heads),
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout=dropout)

    def forward(self, x: Tensor, rel_x: Tensor, valid_mask: Tensor) -> Tensor:
        # x: [B, N, D], rel_x: [B, N, N, R], valid_mask: [B, N]
        bsz, n, _ = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(
            bsz, n, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B,H,N,d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        rel_bias = self.relation_bias(rel_x).permute(0, 3, 1, 2)
        logits = logits + rel_bias

        key_valid = valid_mask[:, None, None, :]
        logits = logits.masked_fill(~key_valid, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
        weights = self.attn_dropout(weights)

        y = torch.matmul(weights, v).transpose(1, 2).reshape(bsz, n, self.dim)
        x = x + self.resid_dropout(self.out(y))
        x = x + self.ffn(self.norm2(x))
        return x * valid_mask.unsqueeze(-1)


@dataclass(frozen=True)
class ControllerOutput:
    utility_quantiles: Tensor  # [B,N,A,3], ordered q10/q50/q90 after sorting
    action_logits: Tensor  # [B,N,A]
    score_delta: Tensor  # [B,N,A]
    set_embedding: Tensor  # [B,D]


class RelationalSetController(nn.Module):
    """Fixed-candidate set controller for keep/drop/relabel/rescore actions."""

    def __init__(
        self,
        *,
        node_dim: int,
        relation_dim: int,
        scene_dim: int,
        num_actions: int,
        dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.node_proj = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.scene_proj = nn.Sequential(
            nn.LayerNorm(scene_dim),
            nn.Linear(scene_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.scene_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.scene_token, std=0.02)
        self.blocks = nn.ModuleList(
            [
                RelationalAttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    relation_dim=relation_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.utility_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_actions * 3),
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_actions),
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_actions),
            nn.Tanh(),
        )

    @staticmethod
    def _prepend_scene_relations(rel_x: Tensor) -> Tensor:
        bsz, n, _, rdim = rel_x.shape
        out = rel_x.new_zeros((bsz, n + 1, n + 1, rdim))
        out[:, 1:, 1:, :] = rel_x
        return out

    def forward(
        self,
        node_x: Tensor,
        relation_x: Tensor,
        scene_x: Tensor,
        valid_mask: Tensor,
    ) -> ControllerOutput:
        if node_x.ndim != 3 or relation_x.ndim != 4:
            raise ValueError("node_x must be [B,N,F], relation_x [B,N,N,R]")
        if valid_mask.dtype is not torch.bool:
            valid_mask = valid_mask.bool()

        h = self.node_proj(node_x)
        scene = self.scene_token.expand(node_x.shape[0], -1, -1)
        scene = scene + self.scene_proj(scene_x).unsqueeze(1)
        h = torch.cat([scene, h], dim=1)
        mask = torch.cat(
            [torch.ones_like(valid_mask[:, :1], dtype=torch.bool), valid_mask], dim=1
        )
        rel = self._prepend_scene_relations(relation_x)

        for block in self.blocks:
            h = block(h, rel, mask)

        set_embedding = h[:, 0]
        candidate_h = h[:, 1:]
        raw_q = self.utility_head(candidate_h).reshape(
            node_x.shape[0], node_x.shape[1], self.num_actions, 3
        )
        # Enforce non-crossing quantiles without constraining their absolute values.
        utility_quantiles, _ = torch.sort(raw_q, dim=-1)
        action_logits = self.action_head(candidate_h)
        score_delta = self.score_head(candidate_h)
        return ControllerOutput(
            utility_quantiles=utility_quantiles,
            action_logits=action_logits,
            score_delta=score_delta,
            set_embedding=set_embedding,
        )
