"""F2 family 层级辅助损失(方案5 §10.1 属性组合的 family 近似)。

保留 25 细类主损失 + 3 粗类辅助(Y3), 额外增加 9 家族中间层辅助分类损失。
三层层次: fine(25) -> family(9) -> coarse(3)。

目的: 兄弟机型(如 TU-22/TU-160 同属 tupolev 家族)先共享家族表征, 再在家族内
细判, 缓解平坦 25 类 softmax 对尾类(HM/LQS/TU-160)过拟合。类权重分解近似:
    w_c = w_family(c) + w_coarse(c) + δ_c

实现: 在正样本上 family_logit = fine_logit @ family_mat, 与 family one-hot 做 BCE,
并入 cls 分量(与 Y3 coarse 辅助同一机制, 保持 3 分量契约)。

训练期模块, 顶层 import ultralytics; 不参与推理。
"""
from __future__ import annotations

from typing import Any, Sequence

import torch

from rsdet.innovation.aggregation import aggregate_group_scores
from rsdet.innovation.coarse import (
    COARSE_MAPPING,
    FAMILY_MAPPING,
    build_coarse_matrix,
)
from rsdet.innovation.hierarchical_loss import HierarchicalCoarseLoss


class FamilyHierarchicalLoss(HierarchicalCoarseLoss):
    """带 family + coarse 三层辅助损失的检测损失(继承 HierarchicalCoarseLoss)。

    loss 保持 3 分量 (box, cls, dfl); family 辅助并入 cls 分量。
    """

    def __init__(
        self,
        model: Any,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        coarse_mapping: tuple[int, ...] = COARSE_MAPPING,
        family_mapping: tuple[int, ...] = FAMILY_MAPPING,
        coarse_gain: float = 0.5,
        family_gain: float = 0.5,
    ) -> None:
        super().__init__(
            model,
            tal_topk=tal_topk,
            tal_topk2=tal_topk2,
            coarse_mapping=coarse_mapping,
            coarse_gain=coarse_gain,
        )
        if len(family_mapping) != self.nc:
            raise ValueError(f"family_mapping 长度 {len(family_mapping)} != model.nc {self.nc}")
        self.family_mapping = tuple(family_mapping)
        self.n_family = max(family_mapping) + 1
        self.family_gain = float(family_gain)
        self.family_mat = build_coarse_matrix(self.nc, self.family_mapping, device=self.device)
        # 每个 family 的细类索引(用于 max 聚合)
        self._family_cls_idx = [
            [c for c in range(self.nc) if self.family_mapping[c] == k]
            for k in range(self.n_family)
        ]

    def get_assigned_targets_and_loss(self, preds: dict[str, Any], batch: dict[str, Any]) -> tuple:
        assignment, loss, loss_det = super().get_assigned_targets_and_loss(preds, batch)
        fg_mask, _, _, _, _ = assignment

        if fg_mask.sum() > 0 and self.family_gain > 0.0:
            pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
            target_scores = self._last_target_scores
            # family 辅助: 统一走 aggregate_group_scores 的 max 聚合（梯度只分给
            # family 内最高类，避免求和聚合推高同 family 所有类导致候选爆炸）。
            family_logits = aggregate_group_scores(pred_scores, self._family_cls_idx)
            family_targets = (target_scores @ self.family_mat).clamp(max=1.0)
            family_bce = self.bce(family_logits, family_targets.to(pred_scores.dtype))
            loss[1] = loss[1] + self.family_gain * (
                family_bce[fg_mask].sum() / max(target_scores.sum(), 1)
            )

        return assignment, loss, loss.detach()


def family_trainer(
    coarse_gain: float = 0.5,
    family_gain: float = 0.5,
    coarse_mapping: Sequence[int] | None = None,
    family_mapping: Sequence[int] | None = None,
) -> type:
    """返回带 family 三层辅助损失的 ``DetectionTrainer`` 子类(F2)。

    Args:
        coarse_gain: 粗类辅助损失权重。
        family_gain: family 中间层辅助损失权重。
        coarse_mapping: 细类 -> 粗类索引(默认冻结 25 类映射)。
        family_mapping: 细类 -> family 索引(默认 9 家族)。

    Returns:
        可传入 ``model.train(trainer=...)`` 的 trainer 类。
    """
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    cmap = tuple(coarse_mapping) if coarse_mapping is not None else COARSE_MAPPING
    fmap = tuple(family_mapping) if family_mapping is not None else FAMILY_MAPPING

    class _FamilyTrainer(DetectionTrainer):
        def _setup_train(self) -> None:
            super()._setup_train()
            model = unwrap_model(self.model)
            criterion = getattr(model, "criterion", None)
            if criterion is None and hasattr(model, "init_criterion"):
                criterion = model.init_criterion()
                model.criterion = criterion
            criterion_name = type(criterion).__name__ if criterion is not None else ""
            if criterion_name in ("E2ELoss", "E2EDetectLoss"):
                from rsdet.innovation.family_loss import FamilyE2ELoss

                model.criterion = FamilyE2ELoss(
                    model,
                    coarse_gain=coarse_gain,
                    family_gain=family_gain,
                    coarse_mapping=cmap,
                    family_mapping=fmap,
                )
            else:
                model.criterion = FamilyHierarchicalLoss(
                    model,
                    coarse_gain=coarse_gain,
                    family_gain=family_gain,
                    coarse_mapping=cmap,
                    family_mapping=fmap,
                )

    _FamilyTrainer.__name__ = "FamilyTrainer"
    return _FamilyTrainer


class FamilyE2ELoss:
    """E2E(one2many/one2one 双分支)的三层辅助损失包装。"""

    def __init__(
        self,
        model: Any,
        coarse_mapping: tuple[int, ...] = COARSE_MAPPING,
        family_mapping: tuple[int, ...] = FAMILY_MAPPING,
        coarse_gain: float = 0.5,
        family_gain: float = 0.5,
    ) -> None:
        from ultralytics.utils.loss import E2ELoss

        cmap = tuple(coarse_mapping)
        fmap = tuple(family_mapping)
        cg = float(coarse_gain)
        fg = float(family_gain)

        def loss_fn(m: Any, tal_topk: int = 10, tal_topk2: int | None = None) -> FamilyHierarchicalLoss:
            return FamilyHierarchicalLoss(
                m,
                tal_topk=tal_topk,
                tal_topk2=tal_topk2,
                coarse_mapping=cmap,
                family_mapping=fmap,
                coarse_gain=cg,
                family_gain=fg,
            )

        self._inner = E2ELoss(model, loss_fn=loss_fn)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __call__(self, preds: Any, batch: Any, *args: Any, **kwargs: Any):
        return self._inner(preds, batch, *args, **kwargs)
