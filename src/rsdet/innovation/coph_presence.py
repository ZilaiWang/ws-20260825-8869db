"""COPH 存在性正则损失(E8)——在检测损失上附加类别无关存在性监督。

DCR²-YOLO 总纲 §4(类别无关候选头 COPH)的训练期近似:
- 对每个分配到正样本的锚点, 要求 ``max_c p(c)`` 尽量接近 1——
  即"该位置存在任一目标"时, 至少一个细类必须报高置信;
- 直接治疗 Y5 的失败模式: 兄弟机型/稀有类细类不确定 → 所有类 logit
  都被压低 → 真实候选消失;
- 结构零改动: 复用检测头的 scores 输出取 max, 不新增头;
- 保留 HierarchicalCoarseLoss 的粗类辅助(Y3 表征约束, 但不参与最终分数)。
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from rsdet.innovation.coarse import COARSE_MAPPING
from rsdet.innovation.hierarchical_loss import HierarchicalCoarseLoss


class CophPresenceLoss(HierarchicalCoarseLoss):
    """带存在性正则的检测损失(继承 HierarchicalCoarseLoss 的完整分配逻辑)。

    loss 保持 3 分量 (box, cls, dfl); presence 项并入 cls 分量。
    """

    def __init__(
        self,
        model: Any,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        coarse_mapping: tuple[int, ...] = COARSE_MAPPING,
        coarse_gain: float = 0.5,
        presence_gain: float = 1.0,
    ) -> None:
        super().__init__(
            model,
            tal_topk=tal_topk,
            tal_topk2=tal_topk2,
            coarse_mapping=coarse_mapping,
            coarse_gain=coarse_gain,
        )
        self.presence_gain = float(presence_gain)

    def get_assigned_targets_and_loss(self, preds: dict[str, Any], batch: dict[str, Any]) -> tuple:
        assignment, loss, loss_det = super().get_assigned_targets_and_loss(preds, batch)
        fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor = assignment

        if fg_mask.sum() > 0 and self.presence_gain > 0.0:
            pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
            # 类别无关存在性: 每个锚点的 max 类概率
            max_prob = pred_scores.sigmoid().amax(dim=-1)  # (bs, na)
            presence_targets = torch.ones_like(max_prob[fg_mask])
            presence_loss = F.binary_cross_entropy(
                max_prob[fg_mask].to(loss.dtype), presence_targets.to(loss.dtype)
            )
            loss[1] = loss[1] + self.presence_gain * presence_loss

        return assignment, loss, loss.detach()


def coph_trainer(
    coarse_gain: float = 0.5,
    presence_gain: float = 1.0,
    coarse_mapping: Sequence[int] | None = None,
) -> type:
    """返回带 COPH 存在性正则的 ``DetectionTrainer`` 子类(E8)。

    Args:
        coarse_gain: 粗类辅助损失权重(继承 Y3, 可选 0 关闭)。
        presence_gain: 存在性正则权重。
        coarse_mapping: 细类 -> 粗类映射(默认冻结 25 类映射)。

    Returns:
        可传入 ``model.train(trainer=...)`` 的 trainer 类。
    """
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    mapping = tuple(coarse_mapping) if coarse_mapping is not None else COARSE_MAPPING

    class _CophTrainer(DetectionTrainer):
        def _setup_train(self) -> None:
            super()._setup_train()
            model = unwrap_model(self.model)
            criterion = getattr(model, "criterion", None)
            if criterion is None and hasattr(model, "init_criterion"):
                criterion = model.init_criterion()
                model.criterion = criterion
            criterion_name = type(criterion).__name__ if criterion is not None else ""
            if criterion_name in ("E2ELoss", "E2EDetectLoss"):
                from rsdet.innovation.coph_presence import E2EPresenceLoss

                # E2E 双分支(one2many/one2one)包存在性正则。
                model.criterion = E2EPresenceLoss(
                    model,
                    coarse_gain=coarse_gain,
                    coarse_mapping=mapping,
                    presence_gain=presence_gain,
                )
            else:
                model.criterion = CophPresenceLoss(
                    model,
                    coarse_gain=coarse_gain,
                    coarse_mapping=mapping,
                    presence_gain=presence_gain,
                )

    _CophTrainer.__name__ = "CophTrainer"
    return _CophTrainer


class E2EPresenceLoss:
    """E2E(one2many/one2one 双分支)存在性正则损失。

    两个分支的 loss_fn 都换成 CophPresenceLoss(presence 并入 cls 分量),
    加权求和/验证兼容与 E2ELoss 原生一致。
    """

    def __init__(
        self,
        model: Any,
        coarse_mapping: tuple[int, ...] = COARSE_MAPPING,
        coarse_gain: float = 0.5,
        presence_gain: float = 1.0,
    ) -> None:
        from ultralytics.utils.loss import E2ELoss

        mapping = tuple(coarse_mapping)
        gain = float(coarse_gain)
        pres = float(presence_gain)

        def loss_fn(m: Any, tal_topk: int = 10, tal_topk2: int | None = None) -> CophPresenceLoss:
            return CophPresenceLoss(
                m,
                tal_topk=tal_topk,
                tal_topk2=tal_topk2,
                coarse_mapping=mapping,
                coarse_gain=gain,
                presence_gain=pres,
            )

        self._inner = E2ELoss(model, loss_fn=loss_fn)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __call__(self, preds: Any, batch: Any, *args: Any, **kwargs: Any):
        return self._inner(preds, batch, *args, **kwargs)
