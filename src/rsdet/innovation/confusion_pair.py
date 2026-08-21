"""F3 反事实兄弟类硬负样本(方案5 §10.3 批次2)——混淆组内判别辅助。

F2(family)让兄弟类共享家族表征(存在性); F3 更进一步对五组最混淆的兄弟类做
"组内 softmax 判别"辅助——在正样本上强制组内正确类的 logit 显著高于兄弟类,
等价于"反事实硬负样本"(把兄弟类当 hard negative), 治疗五组核心混淆:
  TU-160↔TU-22(347) / MS↔QHS(288) / SU-35/34/24(277) / E-8↔KC-135(173) / F-15/16/22(174)。

实现: 继承 FamilyHierarchicalLoss(fine+family+coarse 三层), 额外对每个混淆组 G
取 pred_scores 的 G 列, 在"GT 属于 G"的正样本上施加组内 softmax cross-entropy
(组内标签 = GT 在 G 内的 one-hot)。保持 3 分量契约。

训练期模块, 推理零成本。
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from rsdet.innovation.coarse import COARSE_MAPPING, FAMILY_MAPPING
from rsdet.innovation.family_loss import FamilyHierarchicalLoss, FamilyE2ELoss

# 五组核心兄弟混淆(来自混淆矩阵 top, 2026-08-20 全量 2228 对)
CONFUSION_GROUPS: tuple[tuple[int, ...], ...] = (
    (9, 15),              # TU-160 ↔ TU-22
    (3, 2),               # MS ↔ QHS
    (4, 22, 23),          # SU-35 / SU-34 / SU-24
    (14, 17),             # E-8 ↔ KC-135
    (8, 16, 18),          # F-15 / F-16 / F-22
)


class ConfusionPairLoss(FamilyHierarchicalLoss):
    """fine+family+coarse 三层 + 混淆组内判别辅助(F3)。"""

    def __init__(
        self,
        model: Any,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        coarse_mapping: tuple[int, ...] = COARSE_MAPPING,
        family_mapping: tuple[int, ...] = FAMILY_MAPPING,
        coarse_gain: float = 0.5,
        family_gain: float = 0.5,
        confusion_gain: float = 0.5,
        confusion_groups: tuple[tuple[int, ...], ...] = CONFUSION_GROUPS,
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
        self.confusion_gain = float(confusion_gain)
        self.confusion_groups = tuple(tuple(g) for g in confusion_groups)

    def get_assigned_targets_and_loss(self, preds, batch):
        assignment, loss, loss_det = super().get_assigned_targets_and_loss(preds, batch)
        fg_mask, _, _, _, _ = assignment
        if fg_mask.sum() == 0 or self.confusion_gain <= 0.0:
            return assignment, loss, loss_det

        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        target_scores = self._last_target_scores
        dtype = pred_scores.dtype

        # 每个混淆组: 组内 softmax cross-entropy(只在 GT 属于该组的正样本上)
        n_cf = 0
        cf_loss = torch.zeros((), device=self.device, dtype=dtype)
        for g in self.confusion_groups:
            idx = list(g)
            logits_g = pred_scores[:, :, idx]          # (bs, na, |G|)
            target_g = target_scores[:, :, idx]        # (bs, na, |G|), GT在组内时 one-hot
            # GT 属于该组的正样本: target_g 行和为 1
            in_group = target_g.sum(dim=-1) > 0.5      # (bs, na)
            mask = fg_mask & in_group
            if mask.sum() == 0:
                continue
            # 组内 softmax + cross entropy(负类即"反事实硬负样本")
            l = F.cross_entropy(
                logits_g[mask].float(),
                target_g[mask].argmax(dim=-1).long(),
                reduction="mean",
            )
            cf_loss = cf_loss + l
            n_cf += 1
        if n_cf > 0:
            cf_loss = cf_loss / n_cf
            loss[1] = loss[1] + self.confusion_gain * cf_loss

        return assignment, loss, loss.detach()


def confusion_pair_trainer(
    coarse_gain: float = 0.5,
    family_gain: float = 0.5,
    confusion_gain: float = 0.5,
    coarse_mapping: Sequence[int] | None = None,
    family_mapping: Sequence[int] | None = None,
    confusion_groups: Sequence[Sequence[int]] | None = None,
) -> type:
    """返回带 F3 混淆组内判别辅助的 ``DetectionTrainer`` 子类。"""
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    cmap = tuple(coarse_mapping) if coarse_mapping is not None else COARSE_MAPPING
    fmap = tuple(family_mapping) if family_mapping is not None else FAMILY_MAPPING
    cgrp = tuple(tuple(g) for g in confusion_groups) if confusion_groups is not None else CONFUSION_GROUPS

    class _ConfusionTrainer(DetectionTrainer):
        def _setup_train(self) -> None:
            super()._setup_train()
            model = unwrap_model(self.model)
            criterion = getattr(model, "criterion", None)
            if criterion is None and hasattr(model, "init_criterion"):
                criterion = model.init_criterion()
                model.criterion = criterion
            criterion_name = type(criterion).__name__ if criterion is not None else ""
            if criterion_name in ("E2ELoss", "E2EDetectLoss"):
                from rsdet.innovation.confusion_pair import ConfusionPairE2ELoss

                model.criterion = ConfusionPairE2ELoss(
                    model,
                    coarse_gain=coarse_gain,
                    family_gain=family_gain,
                    confusion_gain=confusion_gain,
                    coarse_mapping=cmap,
                    family_mapping=fmap,
                    confusion_groups=cgrp,
                )
            else:
                model.criterion = ConfusionPairLoss(
                    model,
                    coarse_gain=coarse_gain,
                    family_gain=family_gain,
                    confusion_gain=confusion_gain,
                    coarse_mapping=cmap,
                    family_mapping=fmap,
                    confusion_groups=cgrp,
                )

    _ConfusionTrainer.__name__ = "ConfusionPairTrainer"
    return _ConfusionTrainer


class ConfusionPairE2ELoss:
    """E2E 双分支的 F3 混淆组辅助包装。"""

    def __init__(
        self,
        model: Any,
        coarse_mapping: tuple[int, ...] = COARSE_MAPPING,
        family_mapping: tuple[int, ...] = FAMILY_MAPPING,
        coarse_gain: float = 0.5,
        family_gain: float = 0.5,
        confusion_gain: float = 0.5,
        confusion_groups: tuple[tuple[int, ...], ...] = CONFUSION_GROUPS,
    ) -> None:
        from ultralytics.utils.loss import E2ELoss

        cmap = tuple(coarse_mapping)
        fmap = tuple(family_mapping)
        cgrp = tuple(tuple(g) for g in confusion_groups)

        def loss_fn(m, tal_topk=10, tal_topk2=None):
            return ConfusionPairLoss(
                m,
                tal_topk=tal_topk,
                tal_topk2=tal_topk2,
                coarse_mapping=cmap,
                family_mapping=fmap,
                coarse_gain=coarse_gain,
                family_gain=family_gain,
                confusion_gain=confusion_gain,
                confusion_groups=cgrp,
            )

        self._inner = E2ELoss(model, loss_fn=loss_fn)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __call__(self, preds, batch, *args, **kwargs):
        return self._inner(preds, batch, *args, **kwargs)
