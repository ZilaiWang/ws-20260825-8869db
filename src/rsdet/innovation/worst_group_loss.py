"""D4 worst-group loss(方案5 §十二 批次4)—— 在 loss 层对 worst-group 样本重加权。

D3 通过 hard-curriculum(重复 2x)在采样层过采样 worst-group; D4 更进一步在
loss 层对 worst-group 样本的 cls 损失乘 wg_gain(默认 1.5), 让模型在"高误检域/
低召回域"上获得更强的梯度。

实现: trainer 读 split_view 的 group_id + hard_image_ids 集合, 构建
worst-group 的 im_file 路径集合; WorstGroupLoss 继承 HierarchicalCoarseLoss,
在 get_assigned_targets_and_loss 里按 batch["im_file"] 判断 hard 样本, 对 cls
损失按 (bs, na) 加权。保持 3 分量契约。

训练期模块, 顶层 import ultralytics。
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
from ultralytics.utils.loss import E2ELoss
from ultralytics.utils.tal import make_anchors

from rsdet.innovation.coarse import COARSE_MAPPING, build_coarse_matrix
from rsdet.innovation.hierarchical_loss import HierarchicalCoarseLoss


class WorstGroupLoss(HierarchicalCoarseLoss):
    """带 worst-group 加权的检测损失(继承 HierarchicalCoarseLoss, 复用其 assigner 逻辑)。"""

    def __init__(
        self,
        model: Any,
        hard_paths: set[str] | None = None,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        coarse_mapping: tuple[int, ...] = COARSE_MAPPING,
        coarse_gain: float = 0.5,
        wg_gain: float = 1.5,
    ) -> None:
        super().__init__(
            model,
            tal_topk=tal_topk,
            tal_topk2=tal_topk2,
            coarse_mapping=coarse_mapping,
            coarse_gain=coarse_gain,
        )
        self.hard_paths = set(hard_paths) if hard_paths else set()
        self.wg_gain = float(wg_gain)

    def get_assigned_targets_and_loss(self, preds: dict[str, Any], batch: dict[str, Any]) -> tuple:
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        self._last_target_scores = target_scores

        target_scores_sum = max(target_scores.sum(), 1)

        # 主 cls 损失 + worst-group 加权(按 batch 样本)
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights

        # worst-group 权重: (bs, 1, 1) broadcast
        if self.hard_paths:
            im_files = batch.get("im_file")
            w = torch.ones(batch_size, device=self.device, dtype=dtype)
            for bi, f in enumerate(im_files):
                if str(f) in self.hard_paths:
                    w[bi] = self.wg_gain
            bce_loss = bce_loss * w.view(batch_size, 1, 1)

        loss[1] = bce_loss.sum() / target_scores_sum

        # 粗类辅助(继承 Y3)
        coarse_logits = pred_scores @ self.coarse_mat
        coarse_targets = target_scores @ self.coarse_mat
        coarse_bce = self.bce(coarse_logits, coarse_targets.to(dtype))
        if fg_mask.sum():
            loss[1] = loss[1] + self.coarse_gain * (coarse_bce[fg_mask].sum() / target_scores_sum)

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )


def worst_group_trainer(
    hard_paths: set[str] | None = None,
    coarse_gain: float = 0.5,
    wg_gain: float = 1.5,
    coarse_mapping: Sequence[int] | None = None,
) -> type:
    """返回带 worst-group 加权的 ``DetectionTrainer`` 子类(D4)。

    Args:
        hard_paths: worst-group 的 im_file 绝对路径集合(在 trainer 内按 batch 匹配)。
        coarse_gain: 粗类辅助损失权重。
        wg_gain: worst-group 样本的 cls 损失放大倍数。
        coarse_mapping: 细类 -> 粗类索引。

    Returns:
        可传入 ``model.train(trainer=...)`` 的 trainer 类。
    """
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    cmap = tuple(coarse_mapping) if coarse_mapping is not None else COARSE_MAPPING
    hpaths = set(hard_paths) if hard_paths else set()

    class _WorstGroupTrainer(DetectionTrainer):
        def _setup_train(self) -> None:
            super()._setup_train()
            model = unwrap_model(self.model)
            criterion = getattr(model, "criterion", None)
            if criterion is None and hasattr(model, "init_criterion"):
                criterion = model.init_criterion()
                model.criterion = criterion
            criterion_name = type(criterion).__name__ if criterion is not None else ""
            if criterion_name in ("E2ELoss", "E2EDetectLoss"):
                from rsdet.innovation.worst_group_loss import WorstGroupE2ELoss

                model.criterion = WorstGroupE2ELoss(
                    model,
                    hard_paths=hpaths,
                    coarse_gain=coarse_gain,
                    wg_gain=wg_gain,
                    coarse_mapping=cmap,
                )
            else:
                model.criterion = WorstGroupLoss(
                    model,
                    hard_paths=hpaths,
                    coarse_gain=coarse_gain,
                    wg_gain=wg_gain,
                    coarse_mapping=cmap,
                )

    _WorstGroupTrainer.__name__ = "WorstGroupTrainer"
    return _WorstGroupTrainer


class WorstGroupE2ELoss:
    """E2E 双分支的 worst-group 加权包装。"""

    def __init__(
        self,
        model: Any,
        hard_paths: set[str] | None = None,
        coarse_mapping: tuple[int, ...] = COARSE_MAPPING,
        coarse_gain: float = 0.5,
        wg_gain: float = 1.5,
    ) -> None:
        hpaths = set(hard_paths) if hard_paths else set()
        cmap = tuple(coarse_mapping)
        cg = float(coarse_gain)
        wg = float(wg_gain)

        def loss_fn(m: Any, tal_topk: int = 10, tal_topk2: int | None = None) -> WorstGroupLoss:
            return WorstGroupLoss(
                m,
                hard_paths=hpaths,
                tal_topk=tal_topk,
                tal_topk2=tal_topk2,
                coarse_mapping=cmap,
                coarse_gain=cg,
                wg_gain=wg,
            )

        self._inner = E2ELoss(model, loss_fn=loss_fn)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __call__(self, preds: Any, batch: Any, *args: Any, **kwargs: Any):
        return self._inner(preds, batch, *args, **kwargs)
