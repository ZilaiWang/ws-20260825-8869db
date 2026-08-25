"""V2 车辆中心种子图 + 频域/中心-周围对比增强(方案5 §九 批次3)。

V1(vehicle-only DFD)失败根因: 密集高斯把前景响应"扩散"到中心邻域, 候选爆炸
(+79%)且新增 FP 拖累前沿。V2 的关键改进是"中心-周围对比"——让车辆中心 anchor 的
前景响应显著高于周围环带(等价于频域增强 F_res = F - AvgPool(F) 的 loss 层近似),
使恢复的车辆候选更"锐利"、更少扩散, 受控地救回低对比车辆。

实现: 继承 DfdPresenceLoss(vehicle-only 高斯中心热力图), 额外加:
- 中心-周围对比损失: 对每个车辆 GT, 中心峰值 anchor 的 max_c logit 应比
  环带 anchor 的 max_c logit 高 margin(soft margin ranking);
- 更锐利的 sigma(默认 0.10, 比 V1 的 0.15 更聚焦中心)。

训练期模块, 推理零成本。准入: vehicle NO_CAND 下降且候选不爆炸(≤1.15x Y5)。
"""
from __future__ import annotations

from typing import Any, Sequence

import torch

from rsdet.innovation.coarse import COARSE_MAPPING
from rsdet.innovation.dfd_presence import DfdPresenceLoss, E2EDfdLoss


class VehicleCenterContrastLoss(DfdPresenceLoss):
    """车辆中心种子 + 中心-周围对比损失(V2)。"""

    def __init__(
        self,
        model: Any,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        coarse_mapping: tuple[int, ...] = COARSE_MAPPING,
        coarse_gain: float = 0.5,
        presence_gain: float = 1.0,
        dfd_gain: float = 1.0,
        dfd_sigma_scale: float = 0.10,
        dfd_focal_alpha: float = 2.0,
        dfd_neg_gamma: float = 4.0,
        only_classes: tuple[int, ...] | None = (24,),
        center_gain: float = 0.5,
        center_margin: float = 0.5,
    ) -> None:
        super().__init__(
            model,
            tal_topk=tal_topk,
            tal_topk2=tal_topk2,
            coarse_mapping=coarse_mapping,
            coarse_gain=coarse_gain,
            presence_gain=presence_gain,
            dfd_gain=dfd_gain,
            dfd_sigma_scale=dfd_sigma_scale,
            dfd_focal_alpha=dfd_focal_alpha,
            dfd_neg_gamma=dfd_neg_gamma,
            only_classes=only_classes,
        )
        self.center_gain = float(center_gain)
        self.center_margin = float(center_margin)

    def get_assigned_targets_and_loss(self, preds, batch):
        assignment, loss, loss_det = super().get_assigned_targets_and_loss(preds, batch)
        if self.center_gain <= 0.0 or self._is_one2one:
            return assignment, loss, loss_det

        fg_mask, _, _, anchor_points, _ = assignment
        if fg_mask.sum() == 0:
            return assignment, loss, loss_det

        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        # 用 vehicle 类(24)的 logit 而非 max_c——max_c 会让 center 对比推高"任意类",
        # 导致候选全局爆炸(frontier -63pp)
        presence_logit = pred_scores[:, :, 24]  # (bs, na) vehicle 列

        # 重建车辆 GT 中心/bbox
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        if self.only_classes is not None:
            gt_cls = gt_labels.squeeze(-1).long()
            keep = torch.zeros_like(gt_cls, dtype=torch.bool)
            for c in self.only_classes:
                keep = keep | (gt_cls == c)
            mask_gt = mask_gt * keep.unsqueeze(-1).to(mask_gt.dtype)

        # 中心-周围对比: 中心峰值 anchor 的 logit 应高于环带
        ax = anchor_points[:, 0]
        ay = anchor_points[:, 1]
        na = anchor_points.shape[0]
        center_loss = torch.zeros((), device=self.device, dtype=dtype)
        n_pairs = 0
        for b in range(batch_size):
            for g in range(gt_bboxes.shape[1]):
                if not bool(mask_gt[b, g, 0]):
                    continue
                cx = (gt_bboxes[b, g, 0] + gt_bboxes[b, g, 2]) * 0.5
                cy = (gt_bboxes[b, g, 1] + gt_bboxes[b, g, 3]) * 0.5
                w = (gt_bboxes[b, g, 2] - gt_bboxes[b, g, 0]).clamp(min=1.0)
                h = (gt_bboxes[b, g, 3] - gt_bboxes[b, g, 1]).clamp(min=1.0)
                # 中心邻域(2*stride) vs 环带(0.8~1.6 box)
                d = torch.sqrt((ax - cx) ** 2 + (ay - cy) ** 2)
                center_mask = d < 2.0 * float(self.stride[0])
                ring_mask = (d >= 0.6 * float(w.maximum(h))) & (d < 1.6 * float(w.maximum(h)))
                if center_mask.sum() == 0 or ring_mask.sum() == 0:
                    continue
                center_logit = presence_logit[b][center_mask].max()
                ring_logit = presence_logit[b][ring_mask].mean()
                # soft margin ranking: 中心应比环带高 margin
                center_loss = center_loss + torch.clamp(
                    self.center_margin - (center_logit - ring_logit), min=0.0
                )
                n_pairs += 1
        if n_pairs > 0:
            center_loss = center_loss / n_pairs
            loss[1] = loss[1] + self.center_gain * center_loss

        return assignment, loss, loss.detach()


def v2_vehicle_seed_trainer(
    coarse_gain: float = 0.5,
    presence_gain: float = 1.0,
    dfd_gain: float = 1.0,
    dfd_sigma_scale: float = 0.10,
    center_gain: float = 0.5,
    center_margin: float = 0.5,
    coarse_mapping: Sequence[int] | None = None,
    only_classes: Sequence[int] | None = (24,),
) -> type:
    """返回带 V2 车辆中心-周围对比的 ``DetectionTrainer`` 子类。"""
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    mapping = tuple(coarse_mapping) if coarse_mapping is not None else COARSE_MAPPING
    only_cls = tuple(only_classes) if only_classes is not None else None

    class _V2Trainer(DetectionTrainer):
        def _setup_train(self) -> None:
            super()._setup_train()
            model = unwrap_model(self.model)
            criterion = getattr(model, "criterion", None)
            if criterion is None and hasattr(model, "init_criterion"):
                criterion = model.init_criterion()
                model.criterion = criterion
            criterion_name = type(criterion).__name__ if criterion is not None else ""
            if criterion_name in ("E2ELoss", "E2EDetectLoss"):
                from rsdet.innovation.v2_vehicle_seed import VehicleCenterContrastE2ELoss

                model.criterion = VehicleCenterContrastE2ELoss(
                    model,
                    coarse_gain=coarse_gain,
                    presence_gain=presence_gain,
                    dfd_gain=dfd_gain,
                    dfd_sigma_scale=dfd_sigma_scale,
                    center_gain=center_gain,
                    center_margin=center_margin,
                    coarse_mapping=mapping,
                    only_classes=only_cls,
                )
            else:
                model.criterion = VehicleCenterContrastLoss(
                    model,
                    coarse_gain=coarse_gain,
                    presence_gain=presence_gain,
                    dfd_gain=dfd_gain,
                    dfd_sigma_scale=dfd_sigma_scale,
                    center_gain=center_gain,
                    center_margin=center_margin,
                    coarse_mapping=mapping,
                    only_classes=only_cls,
                )

    _V2Trainer.__name__ = "V2VehicleSeedTrainer"
    return _V2Trainer


class VehicleCenterContrastE2ELoss:
    """E2E 双分支的 V2 车辆中心-周围对比包装。"""

    def __init__(
        self,
        model: Any,
        coarse_mapping: tuple[int, ...] = COARSE_MAPPING,
        coarse_gain: float = 0.5,
        presence_gain: float = 1.0,
        dfd_gain: float = 1.0,
        dfd_sigma_scale: float = 0.10,
        center_gain: float = 0.5,
        center_margin: float = 0.5,
        only_classes: tuple[int, ...] | None = (24,),
    ) -> None:
        from ultralytics.utils.loss import E2ELoss

        mapping = tuple(coarse_mapping)
        only_cls = tuple(only_classes) if only_classes is not None else None

        def loss_fn(m, tal_topk=10, tal_topk2=None):
            return VehicleCenterContrastLoss(
                m,
                tal_topk=tal_topk,
                tal_topk2=tal_topk2,
                coarse_mapping=mapping,
                coarse_gain=coarse_gain,
                presence_gain=presence_gain,
                dfd_gain=dfd_gain,
                dfd_sigma_scale=dfd_sigma_scale,
                center_gain=center_gain,
                center_margin=center_margin,
                only_classes=only_cls,
            )

        self._inner = E2ELoss(model, loss_fn=loss_fn)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __call__(self, preds, batch, *args, **kwargs):
        return self._inner(preds, batch, *args, **kwargs)
