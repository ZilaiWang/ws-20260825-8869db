"""Y3 层次粗细类辅助损失（材料 19 第三优先）。

保留 25 类主损失（box/cls/dfl），额外增加 ship/aircraft/vehicle 粗类辅助分类
损失，只在训练期存在，不改变输出接口与推理成本。针对 TU-160/F-22、LQS/HM
等类间混淆（当前最大结构错误之一：``FP_CLS``）。

实现方式：在正样本（``fg_mask``）上，把 25 细类 logits 按粗类归属矩阵聚合为
3 粗类 logits（``coarse_logit = fine_logit @ M``），与粗类 one-hot 标签做 BCE，
把粗类辅助损失**并入 cls 分量**（``loss[1]``）。这样 ``loss`` 仍为 3 分量
``(box, cls, dfl)``，与 ultralytics validator/EMA 的 3 分量契约完全兼容——
训练时粗类损失参与总损失，验证时退化为标准 v8DetectionLoss 不引入额外分量。

准入信号（材料 19）：``FP_CLS`` 净下降、目标细类 macro Recall/FDR 改善、
飞机总体与 pooled Recall 不退化。

注意：本模块需 ultralytics 环境（继承 ``v8DetectionLoss``），属于训练期模块，
故顶层 import ultralytics；``rsdet.innovation.__init__`` 不 import 本模块以保持
包在无深度学习环境下可导入。
"""

from __future__ import annotations

from typing import Any

import torch
from ultralytics.utils.loss import E2EDetectLoss, v8DetectionLoss
from ultralytics.utils.tal import make_anchors

from rsdet.innovation.coarse import COARSE_MAPPING, build_coarse_matrix


class HierarchicalCoarseLoss(v8DetectionLoss):
    """带粗类辅助损失的检测损失（继承 ultralytics ``v8DetectionLoss``）。

    粗类辅助损失并入 ``cls`` 分量，``loss`` 保持 3 分量 ``(box, cls, dfl)``，
    trainer/validator 无需任何改动。
    """

    def __init__(
        self,
        model: Any,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        coarse_mapping: tuple[int, ...] = COARSE_MAPPING,
        coarse_gain: float = 0.5,
    ) -> None:
        """初始化。

        Args:
            model: ultralytics ``DetectionModel``（须 de-paralleled）。
            coarse_mapping: 细类 -> 粗类索引（长度 = model.nc）。
            coarse_gain: 粗类辅助损失相对主 cls 损失的权重（默认 0.5）。
        """
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        if len(coarse_mapping) != self.nc:
            raise ValueError(f"coarse_mapping 长度 {len(coarse_mapping)} != model.nc {self.nc}")
        self.coarse_mapping = tuple(coarse_mapping)
        self.n_coarse = max(coarse_mapping) + 1
        self.coarse_gain = float(coarse_gain)
        # (nc, n_coarse) 归属矩阵，随模型设备
        self.coarse_mat = build_coarse_matrix(self.nc, self.coarse_mapping, device=self.device)

    def get_assigned_targets_and_loss(self, preds: dict[str, Any], batch: dict[str, Any]) -> tuple:
        """计算 box/cls(+coarse)/dfl 三分量损失。

        复刻 ultralytics 8.4.103 ``v8DetectionLoss.get_assigned_targets_and_loss``
        的前向逻辑，唯一差异是把粗类辅助损失并入 ``cls`` 分量。
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss（主 25 细类）
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        # 粗类辅助 cls（Y3 增量）：并入 cls 分量，只在正样本上。
        coarse_logits = pred_scores @ self.coarse_mat  # (bs, na, n_coarse)
        coarse_targets = target_scores @ self.coarse_mat  # (bs, na, n_coarse) one-hot
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
        loss[1] *= self.hyp.cls  # cls 分量（含粗类辅助）统一乘 cls gain
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )


class HierarchicalE2ECoarseLoss(E2EDetectLoss):
    """YOLO26 系列（end2end one2many/one2one 双分支）的层次粗类辅助损失。

    ultralytics 8.4.103 的 yolo26 系 Detect head 在训练时输出
    ``{"one2many": ..., "one2one": ...}``，必须用 ``E2EDetectLoss`` 包装
    （one2many 用 ``tal_topk=10``、one2one 用 ``tal_topk=1``），两个分支各自
    使用 ``HierarchicalCoarseLoss``（粗类辅助并入 cls 分量，loss 仍 3 分量）。
    """

    def __init__(
        self,
        model: Any,
        coarse_mapping: tuple[int, ...] = COARSE_MAPPING,
        coarse_gain: float = 0.5,
    ) -> None:
        """初始化 E2E 双分支层次损失。

        Args:
            model: ultralytics ``DetectionModel``（须 de-paralleled）。
            coarse_mapping: 细类 -> 粗类索引（长度 = model.nc）。
            coarse_gain: 粗类辅助损失相对主 cls 损失的权重。
        """
        self.one2many = HierarchicalCoarseLoss(
            model, tal_topk=10, coarse_mapping=coarse_mapping, coarse_gain=coarse_gain
        )
        self.one2one = HierarchicalCoarseLoss(
            model, tal_topk=1, coarse_mapping=coarse_mapping, coarse_gain=coarse_gain
        )
