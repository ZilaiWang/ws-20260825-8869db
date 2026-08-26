"""集中聚合函数（方案6 §十一 工程不变量）。

family / coarse / vehicle 等组级辅助损失的 logits 聚合必须统一走本模块，
业务代码禁止再复制一份局部求和/最大值聚合实现。

为什么只支持 max 聚合：
    - 求和聚合 ``logits @ 归属矩阵`` 会让梯度均匀推高同组所有类，
      导致候选爆炸（F2/F3/V2/D4 都踩过此雷，frontier 崩盘）。
    - max 聚合只把梯度分给组内最高类，天然有界、不扩候选。
"""
from __future__ import annotations

from typing import Literal, Sequence

import torch
from torch import Tensor


def aggregate_group_scores(
    logits: Tensor,
    group_indices: Sequence[Sequence[int]],
    reduction: Literal["max"] = "max",
) -> Tensor:
    """把细类 logits 按组 max 聚合为组级 logits。

    Args:
        logits: 形状 ``(..., nc)`` 的细类 logits，最后一维为细类。
        group_indices: 每个组的细类索引序列，例如 ``[[0, 1, 2, 3], list(range(4, 24)), [24]]``。
        reduction: 仅支持 ``"max"``（求和聚合是被禁止的候选爆炸源）。

    Returns:
        形状 ``(..., n_group)`` 的组级 logits。
    """
    if reduction != "max":
        raise ValueError(
            "aggregate_group_scores 仅支持 max 聚合；求和聚合会推高同组所有类导致候选爆炸"
        )
    if not isinstance(logits, Tensor):
        raise TypeError(f"logits 必须是 torch.Tensor，收到 {type(logits)!r}")
    nc = int(logits.shape[-1])
    seen: set[int] = set()
    for idx in group_indices:
        for c in idx:
            if c < 0 or c >= nc:
                raise ValueError(f"细类索引 {c} 越界（nc={nc}）")
            if c in seen:
                raise ValueError(f"细类索引 {c} 出现在多个组中")
            seen.add(c)
    return torch.stack(
        [logits[..., list(idx)].max(dim=-1).values for idx in group_indices],
        dim=-1,
    )
