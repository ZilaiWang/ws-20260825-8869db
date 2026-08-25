"""F6 尾类原型残差(方案5 §10.1 批次2)——尾类 cls 重加权。

尾类(HM 17 / LQS 30 / TU-160 361 / FSC 402 / E-8 432 / F-22 493)样本量远小于
头部 FA-18(2147), 平坦分类对尾类欠拟合。F6 在 F2(family 共享表征)基础上, 对
尾类的 cls 损失额外加权(默认 2x), 让尾类在家族共享表征之外学到更强的判别残差
(近似 w_c = w_family + w_coarse + δ_c 中 δ_c 的增强)。

实现: 继承 FamilyHierarchicalLoss(fine+family+coarse 三层), 额外对尾类的正样本
cls 损失乘 tail_weight。保持 3 分量契约。训练期模块, 推理零成本。
"""
from __future__ import annotations

from typing import Any, Sequence

import torch

from rsdet.innovation.coarse import COARSE_MAPPING, FAMILY_MAPPING
from rsdet.innovation.family_loss import FamilyHierarchicalLoss, FamilyE2ELoss

# 尾类(样本量 < 500 的关键尾类, 2026-08-20 GT 分布)
TAIL_CLASSES: tuple[int, ...] = (0, 1, 9, 14, 18, 24)  # HM, LQS, TU-160, E-8, F-22, FSC


class TailWeightLoss(FamilyHierarchicalLoss):
    """fine+family+coarse 三层 + 尾类重加权(F6)。"""

    def __init__(
        self,
        model: Any,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        coarse_mapping: tuple[int, ...] = COARSE_MAPPING,
        family_mapping: tuple[int, ...] = FAMILY_MAPPING,
        coarse_gain: float = 0.5,
        family_gain: float = 0.5,
        tail_weight: float = 2.0,
        tail_classes: tuple[int, ...] = TAIL_CLASSES,
    ) -> None:
        super().__init__(
            model,
            tal_topk=tal_topk,
            tal_topk2=tal_topk2,
            coarse_mapping=coarse_mapping,
            family_mapping=family_mapping,
            coarse_gain=coarse_gain,
            family_gain=family_gain,
        )
        self.tail_weight = float(tail_weight)
        # 每类权重(尾类=tail_weight, 否则 1)
        w = torch.ones(self.nc, device=self.device)
        for c in tail_classes:
            if 0 <= c < self.nc:
                w[c] = self.tail_weight
        self._tail_w = w.view(1, 1, -1)

    def get_assigned_targets_and_loss(self, preds, batch):
        assignment, loss, loss_det = super().get_assigned_targets_and_loss(preds, batch)
        fg_mask, _, _, _, _ = assignment
        if fg_mask.sum() == 0 or self.tail_weight <= 1.0:
            return assignment, loss, loss_det

        # 尾类额外加权: 在 cls 分量上对尾类正样本再乘 (tail_weight - 1)
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        target_scores = self._last_target_scores
        dtype = pred_scores.dtype

        extra_w = (self._tail_w - 1.0).to(dtype)  # 尾类额外增量
        bce = self.bce(pred_scores, target_scores.to(dtype))  # (bs, na, nc)
        tail_bce = bce * extra_w * (target_scores.to(dtype) > 0.5)  # 只尾类正样本
        tail_sum = (target_scores * (self._tail_w - 1.0)).sum().clamp(min=1.0)
        loss[1] = loss[1] + tail_bce[fg_mask].sum() / tail_sum

        return assignment, loss, loss.detach()


def tail_weight_trainer(
    coarse_gain: float = 0.5,
    family_gain: float = 0.5,
    tail_weight: float = 2.0,
    coarse_mapping: Sequence[int] | None = None,
    family_mapping: Sequence[int] | None = None,
    tail_classes: Sequence[int] | None = None,
) -> type:
    """返回带 F6 尾类重加权的 ``DetectionTrainer`` 子类。"""
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    cmap = tuple(coarse_mapping) if coarse_mapping is not None else COARSE_MAPPING
    fmap = tuple(family_mapping) if family_mapping is not None else FAMILY_MAPPING
    tcls = tuple(tail_classes) if tail_classes is not None else TAIL_CLASSES

    class _TailTrainer(DetectionTrainer):
        def _setup_train(self) -> None:
            super()._setup_train()
            model = unwrap_model(self.model)
            criterion = getattr(model, "criterion", None)
            if criterion is None and hasattr(model, "init_criterion"):
                criterion = model.init_criterion()
                model.criterion = criterion
            criterion_name = type(criterion).__name__ if criterion is not None else ""
            if criterion_name in ("E2ELoss", "E2EDetectLoss"):
                from rsdet.innovation.tail_weight import TailWeightE2ELoss

                model.criterion = TailWeightE2ELoss(
                    model,
                    coarse_gain=coarse_gain,
                    family_gain=family_gain,
                    tail_weight=tail_weight,
                    coarse_mapping=cmap,
                    family_mapping=fmap,
                    tail_classes=tcls,
                )
            else:
                model.criterion = TailWeightLoss(
                    model,
                    coarse_gain=coarse_gain,
                    family_gain=family_gain,
                    tail_weight=tail_weight,
                    coarse_mapping=cmap,
                    family_mapping=fmap,
                    tail_classes=tcls,
                )

    _TailTrainer.__name__ = "TailWeightTrainer"
    return _TailTrainer


class TailWeightE2ELoss:
    """E2E 双分支的 F6 尾类重加权包装。"""

    def __init__(
        self,
        model: Any,
        coarse_mapping: tuple[int, ...] = COARSE_MAPPING,
        family_mapping: tuple[int, ...] = FAMILY_MAPPING,
        coarse_gain: float = 0.5,
        family_gain: float = 0.5,
        tail_weight: float = 2.0,
        tail_classes: tuple[int, ...] = TAIL_CLASSES,
    ) -> None:
        from ultralytics.utils.loss import E2ELoss

        cmap = tuple(coarse_mapping)
        fmap = tuple(family_mapping)
        tcls = tuple(tail_classes)

        def loss_fn(m, tal_topk=10, tal_topk2=None):
            return TailWeightLoss(
                m,
                tal_topk=tal_topk,
                tal_topk2=tal_topk2,
                coarse_mapping=cmap,
                family_mapping=fmap,
                coarse_gain=coarse_gain,
                family_gain=family_gain,
                tail_weight=tail_weight,
                tail_classes=tcls,
            )

        self._inner = E2ELoss(model, loss_fn=loss_fn)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __call__(self, preds, batch, *args, **kwargs):
        return self._inner(preds, batch, *args, **kwargs)
