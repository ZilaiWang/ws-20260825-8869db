"""F4 可观测性掩码(方案5 §10.2 批次2)——小目标不强学不可见属性。

一个 30 像素飞机无法可靠观察发动机布局/翼型, 强制监督兄弟类判别会制造噪声。
F4 在 F3(混淆组判别)基础上, 按目标短边像素计算可观测性权重, 对短边小的目标的
混淆组辅助损失降权——小目标只学粗类/家族(可观测), 不强行学兄弟类细判(不可观测)。

实现: 继承 ConfusionPairLoss, 在混淆组辅助里按 GT 短边像素计算 obs 权重
(短边 < obs_min 时权重线性降到 0), 对混淆组 cross-entropy 加权。
保持 3 分量契约。训练期模块, 推理零成本。
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from rsdet.innovation.coarse import COARSE_MAPPING, FAMILY_MAPPING
from rsdet.innovation.confusion_pair import CONFUSION_GROUPS
from rsdet.innovation.family_loss import FamilyHierarchicalLoss, FamilyE2ELoss


class ObservabilityLoss(FamilyHierarchicalLoss):
    """fine+family+coarse 三层 + 混淆组判别(可观测性加权)(F4)。

    注意: 继承 FamilyHierarchicalLoss(而非 ConfusionPairLoss), 只加一次
    可观测性加权的混淆组辅助(替代 F3 的无加权版本), 避免重复计算。
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
        confusion_gain: float = 0.5,
        confusion_groups: tuple[tuple[int, ...], ...] = CONFUSION_GROUPS,
        obs_min: float = 24.0,
        obs_max: float = 48.0,
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
        self.obs_min = float(obs_min)
        self.obs_max = float(obs_max)

    def get_assigned_targets_and_loss(self, preds, batch):
        # 先复用 F3 的混淆组辅助(在 super 里已做), 但 F4 需要按可观测性重加权,
        # 因此这里完全重写混淆组辅助部分。为简单, 调 super 拿基础 loss, 再额外
        # 减掉/重加可观测性加权的混淆组损失是不易的; 改为直接复制 F3 逻辑加 obs 权重。
        assignment, loss, loss_det = super().get_assigned_targets_and_loss(preds, batch)
        fg_mask, target_gt_idx, target_bboxes, _, _ = assignment
        if fg_mask.sum() == 0 or self.confusion_gain <= 0.0:
            return assignment, loss, loss_det

        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        target_scores = self._last_target_scores
        dtype = pred_scores.dtype

        # 每个正样本 anchor 对应 GT 的短边(像素)——向量化(避免双重 Python 循环 + GPU→CPU 同步)
        # 注意: bbox 在 AMP 下是 float32, 短边须用 float32(不能用 pred_scores 的 Half), 否则索引赋值类型不匹配
        bs, na = pred_scores.shape[:2]
        short_edge = torch.zeros(bs, na, device=self.device, dtype=torch.float32)
        if bool(fg_mask.any()):
            fg_rows, fg_cols = fg_mask.nonzero(as_tuple=True)
            gi = target_gt_idx[fg_rows, fg_cols].long()
            # 防御: 无效 GT 索引(负值)兜底为 0
            gi = gi.clamp(min=0)
            bb = target_bboxes[fg_rows, gi]
            w = (bb[:, 2] - bb[:, 0]).abs().clamp(min=1.0)
            h = (bb[:, 3] - bb[:, 1]).abs().clamp(min=1.0)
            short_edge[fg_rows, fg_cols] = w.minimum(h)

        # 可观测性权重: 短边 < obs_min → 0, > obs_max → 1, 中间线性
        obs_w = ((short_edge - self.obs_min) / (self.obs_max - self.obs_min)).clamp(0.0, 1.0)

        # 混淆组内 softmax cross-entropy, 按 obs_w 加权
        n_cf = 0
        cf_loss = torch.zeros((), device=self.device, dtype=dtype)
        for g in self.confusion_groups:
            idx = list(g)
            logits_g = pred_scores[:, :, idx]
            target_g = target_scores[:, :, idx]
            in_group = target_g.sum(dim=-1) > 0.5
            mask = fg_mask & in_group
            if mask.sum() == 0:
                continue
            ce = F.cross_entropy(
                logits_g[mask].float(),
                target_g[mask].argmax(dim=-1).long(),
                reduction="none",
            )
            # 可观测性加权(短边小的目标混淆组辅助降权)
            wgt = obs_w[mask]
            wsum = wgt.sum().clamp(min=1.0)
            cf_loss = cf_loss + (ce * wgt).sum() / wsum
            n_cf += 1
        if n_cf > 0:
            cf_loss = cf_loss / n_cf
            loss[1] = loss[1] + self.confusion_gain * cf_loss

        return assignment, loss, loss.detach()


def observability_trainer(
    coarse_gain: float = 0.5,
    family_gain: float = 0.5,
    confusion_gain: float = 0.5,
    obs_min: float = 24.0,
    obs_max: float = 48.0,
    coarse_mapping: Sequence[int] | None = None,
    family_mapping: Sequence[int] | None = None,
    confusion_groups: Sequence[Sequence[int]] | None = None,
) -> type:
    """返回带 F4 可观测性掩码的 ``DetectionTrainer`` 子类。"""
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    cmap = tuple(coarse_mapping) if coarse_mapping is not None else COARSE_MAPPING
    fmap = tuple(family_mapping) if family_mapping is not None else FAMILY_MAPPING
    cgrp = tuple(tuple(g) for g in confusion_groups) if confusion_groups is not None else CONFUSION_GROUPS

    class _ObsTrainer(DetectionTrainer):
        def _setup_train(self) -> None:
            super()._setup_train()
            model = unwrap_model(self.model)
            criterion = getattr(model, "criterion", None)
            if criterion is None and hasattr(model, "init_criterion"):
                criterion = model.init_criterion()
                model.criterion = criterion
            criterion_name = type(criterion).__name__ if criterion is not None else ""
            if criterion_name in ("E2ELoss", "E2EDetectLoss"):
                from rsdet.innovation.observability import ObservabilityE2ELoss

                model.criterion = ObservabilityE2ELoss(
                    model,
                    coarse_gain=coarse_gain,
                    family_gain=family_gain,
                    confusion_gain=confusion_gain,
                    obs_min=obs_min,
                    obs_max=obs_max,
                    coarse_mapping=cmap,
                    family_mapping=fmap,
                    confusion_groups=cgrp,
                )
            else:
                model.criterion = ObservabilityLoss(
                    model,
                    coarse_gain=coarse_gain,
                    family_gain=family_gain,
                    confusion_gain=confusion_gain,
                    obs_min=obs_min,
                    obs_max=obs_max,
                    coarse_mapping=cmap,
                    family_mapping=fmap,
                    confusion_groups=cgrp,
                )

    _ObsTrainer.__name__ = "ObservabilityTrainer"
    return _ObsTrainer


class ObservabilityE2ELoss:
    """E2E 双分支的 F4 可观测性掩码包装。"""

    def __init__(
        self,
        model: Any,
        coarse_mapping: tuple[int, ...] = COARSE_MAPPING,
        family_mapping: tuple[int, ...] = FAMILY_MAPPING,
        coarse_gain: float = 0.5,
        family_gain: float = 0.5,
        confusion_gain: float = 0.5,
        confusion_groups: tuple[tuple[int, ...], ...] = CONFUSION_GROUPS,
        obs_min: float = 24.0,
        obs_max: float = 48.0,
    ) -> None:
        from ultralytics.utils.loss import E2ELoss

        cmap = tuple(coarse_mapping)
        fmap = tuple(family_mapping)
        cgrp = tuple(tuple(g) for g in confusion_groups)

        def loss_fn(m, tal_topk=10, tal_topk2=None):
            return ObservabilityLoss(
                m,
                tal_topk=tal_topk,
                tal_topk2=tal_topk2,
                coarse_mapping=cmap,
                family_mapping=fmap,
                coarse_gain=coarse_gain,
                family_gain=family_gain,
                confusion_gain=confusion_gain,
                confusion_groups=cgrp,
                obs_min=obs_min,
                obs_max=obs_max,
            )

        self._inner = E2ELoss(model, loss_fn=loss_fn)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __call__(self, preds, batch, *args, **kwargs):
        return self._inner(preds, batch, *args, **kwargs)
