"""DFD 密集前景监督(E9 / 方案4 B4 训练期密集前景监督)——治疗候选缺失漏检。

诊断结论(2026-08-20): 当前 HERA-Core(OER+改类+has_oto)已到 Recall@FDR=0.12 = 0.9607,
剩余瓶颈是**候选缺失**——vehicle(FSC) 17.9% / ship 4.3% 的 GT 完全无候选,
而细类已饱和(aircraft 98.2% 正确)。

COPH 存在性正则的局限: 它只对 TAL 分配的正样本 anchor 做 ``max_c p(c)≈1`` 监督。
但"漏检"目标恰恰是 anchor 响应太低、TAL 根本分配不到正样本的目标,
因此 COPH 对它们没有梯度。这解释了为何 COPH 候选 +44~63% 却救不回 hard-blind 目标。

DFD(密集前景监督)机制(方案4 §四.3 一级确定性 box supervision):
- **独立于 TAL 分配**, 直接用每个 GT box 生成自适应高斯中心热力图
  (``sigma ~ box/6``, 最小 ``stride[0]`` 保证小目标覆盖 3x3 邻域);
- 用现有 head 的 ``max_c logit`` 作为前景响应(与 COPH 一致, 结构零改动);
- penalty-reduced focal loss 对"所有靠近 GT 中心的 anchor"施加软标签监督,
  使 backbone/head 在低对比/小目标中心也学出前景响应;
- 训练期启用, 推理零成本(不新增 head, 不改变输出接口)。

与 COPH 的差异(核心价值):
  COPH = 稀疏(仅 TAL 正样本)二值(≈1)监督, 漏检目标无梯度;
  DFD  = 密集(中心邻域)软标签(高斯)监督, 漏检目标也有梯度。

准入门槛(方案4 B 系列):
  candidate-floor Recall +0.5pp 以上;
  Recall@FDR=0.12 +0.3~0.5pp;
  vehicle/ship NO_CAND 明显下降;
  最坏折不反向下降。
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from rsdet.innovation.coarse import COARSE_MAPPING
from rsdet.innovation.coph_presence import CophPresenceLoss


class DfdPresenceLoss(CophPresenceLoss):
    """COPH 存在性正则 + DFD 密集前景监督(继承 COPH 完整分配逻辑)。

    注意: DFD 密集监督只在 one2many 分支生效(``tal_topk2 is None``),
    one2one 分支保持 COPH 原语义(精确唯一匹配不应被密集高斯软化)。
    """

    def __init__(
        self,
        model: Any,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        coarse_mapping: tuple[int, ...] = COARSE_MAPPING,
        coarse_gain: float = 0.5,
        presence_gain: float = 1.0,
        dfd_gain: float = 1.0,
        dfd_sigma_scale: float = 0.15,
        dfd_focal_alpha: float = 2.0,
        dfd_neg_gamma: float = 4.0,
    ) -> None:
        """初始化。

        Args:
            model: ultralytics ``DetectionModel``(须 de-paralleled)。
            coarse_gain / presence_gain: 继承 COPH 的粗类/存在性正则权重。
            dfd_gain: 密集前景监督损失权重(0 关闭)。
            dfd_sigma_scale: 高斯 sigma 相对 box 尺寸的比例(sigma = box * scale,
                最小 clamp 到 stride[0])。
            dfd_focal_alpha: focal 调制指数。
            dfd_neg_gamma: 负样本 (1-heatmap)^gamma 调制(靠近中心但非峰值处降权)。
        """
        super().__init__(
            model,
            tal_topk=tal_topk,
            tal_topk2=tal_topk2,
            coarse_mapping=coarse_mapping,
            coarse_gain=coarse_gain,
            presence_gain=presence_gain,
        )
        self.dfd_gain = float(dfd_gain)
        self.dfd_sigma_scale = float(dfd_sigma_scale)
        self.dfd_focal_alpha = float(dfd_focal_alpha)
        self.dfd_neg_gamma = float(dfd_neg_gamma)
        # one2one 分支(tal_topk2 is not None)不做密集监督
        self._is_one2one = tal_topk2 is not None

    def get_assigned_targets_and_loss(
        self, preds: dict[str, Any], batch: dict[str, Any]
    ) -> tuple:
        assignment, loss, loss_det = super().get_assigned_targets_and_loss(preds, batch)

        if self.dfd_gain <= 0.0 or self._is_one2one:
            return assignment, loss, loss_det

        fg_mask, _target_gt_idx, _target_bboxes, anchor_points, _stride_tensor = assignment
        if fg_mask.sum() == 0:
            return assignment, loss, loss_det

        # 前景响应: max_c logit(结构零改动, 与 COPH 一致)
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()  # (bs, na, nc)
        presence_logit = pred_scores.amax(dim=-1)  # (bs, na)

        # 重新提取 GT targets(与 super 内 preprocess 一致)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        _gt_labels, gt_bboxes = targets.split((1, 4), 2)  # (bs, ngt, ...)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)  # (bs, ngt, 1)

        heatmap = self._build_center_heatmap(anchor_points, gt_bboxes, mask_gt, batch_size)

        # penalty-reduced focal loss(软标签, CenterNet 风格)
        p = torch.sigmoid(presence_logit)
        pos = -heatmap * (1 - p).pow(self.dfd_focal_alpha) * F.logsigmoid(presence_logit)
        neg = (
            -(1 - heatmap).pow(self.dfd_neg_gamma)
            * p.pow(self.dfd_focal_alpha)
            * F.logsigmoid(-presence_logit)
        )
        num_pos = heatmap.sum().clamp(min=1.0)
        dfd_loss = (pos.sum() + neg.sum()) / num_pos

        loss[1] = loss[1] + self.dfd_gain * dfd_loss
        return assignment, loss, loss_det

    def _build_center_heatmap(
        self,
        anchor_points: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """对每个 GT 生成自适应高斯中心热力图(anchor 网格上, 逐 GT 取 max)。

        Args:
            anchor_points: (na, 2) 原图像素坐标。
            gt_bboxes: (bs, ngt, 4) xyxy。
            mask_gt: (bs, ngt, 1) 有效 GT 掩码。
            batch_size: 当前 batch 大小。

        Returns:
            (bs, na) 高斯中心热力图, 值域 [0, 1]。
        """
        device = anchor_points.device
        na = anchor_points.shape[0]
        ngt = gt_bboxes.shape[1]
        heatmap = torch.zeros(batch_size, na, device=device, dtype=gt_bboxes.dtype)

        cx = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) * 0.5  # (bs, ngt)
        cy = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) * 0.5
        w = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp(min=1.0)
        h = (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp(min=1.0)
        min_sigma = float(self.stride[0])  # 小目标保证至少覆盖 3x3 邻域
        sx = (w * self.dfd_sigma_scale).clamp(min=min_sigma)
        sy = (h * self.dfd_sigma_scale).clamp(min=min_sigma)

        ax = anchor_points[:, 0]  # (na,)
        ay = anchor_points[:, 1]
        for b in range(batch_size):
            for g in range(ngt):
                if not bool(mask_gt[b, g, 0]):
                    continue
                dx = (ax - cx[b, g]) / sx[b, g]
                dy = (ay - cy[b, g]) / sy[b, g]
                hm = torch.exp(-0.5 * (dx * dx + dy * dy))
                heatmap[b] = torch.maximum(heatmap[b], hm)
        return heatmap


class E2EDfdLoss:
    """E2E(one2many/one2one 双分支)DFD 密集前景监督损失。

    与 ``E2EPresenceLoss`` 同构: 两个分支的 loss_fn 换成 ``DfdPresenceLoss``,
    DFD 密集监督只在 one2many 分支(``tal_topk2 is None``)生效,
    one2one 分支退化为 COPH 语义。加权求和/验证兼容与 ``E2ELoss`` 一致。
    """

    def __init__(
        self,
        model: Any,
        coarse_mapping: tuple[int, ...] = COARSE_MAPPING,
        coarse_gain: float = 0.5,
        presence_gain: float = 1.0,
        dfd_gain: float = 1.0,
        dfd_sigma_scale: float = 0.15,
        dfd_focal_alpha: float = 2.0,
        dfd_neg_gamma: float = 4.0,
    ) -> None:
        from ultralytics.utils.loss import E2ELoss

        mapping = tuple(coarse_mapping)

        def loss_fn(
            m: Any, tal_topk: int = 10, tal_topk2: int | None = None
        ) -> DfdPresenceLoss:
            return DfdPresenceLoss(
                m,
                tal_topk=tal_topk,
                tal_topk2=tal_topk2,
                coarse_mapping=mapping,
                coarse_gain=coarse_gain,
                presence_gain=presence_gain,
                dfd_gain=dfd_gain,
                dfd_sigma_scale=dfd_sigma_scale,
                dfd_focal_alpha=dfd_focal_alpha,
                dfd_neg_gamma=dfd_neg_gamma,
            )

        self._inner = E2ELoss(model, loss_fn=loss_fn)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __call__(self, preds: Any, batch: Any, *args: Any, **kwargs: Any):
        return self._inner(preds, batch, *args, **kwargs)


def dfd_trainer(
    coarse_gain: float = 0.5,
    presence_gain: float = 1.0,
    dfd_gain: float = 1.0,
    dfd_sigma_scale: float = 0.15,
    dfd_focal_alpha: float = 2.0,
    dfd_neg_gamma: float = 4.0,
    coarse_mapping: Sequence[int] | None = None,
) -> type:
    """返回带 DFD 密集前景监督的 ``DetectionTrainer`` 子类(E9/B4)。

    Args:
        coarse_gain / presence_gain: 继承 COPH 的粗类/存在性正则权重。
        dfd_gain: 密集前景监督权重。
        dfd_sigma_scale / dfd_focal_alpha / dfd_neg_gamma: 高斯/focal 超参。
        coarse_mapping: 细类 -> 粗类映射(默认冻结 25 类映射)。

    Returns:
        可传入 ``model.train(trainer=...)`` 的 trainer 类。
    """
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    mapping = tuple(coarse_mapping) if coarse_mapping is not None else COARSE_MAPPING

    class _DfdTrainer(DetectionTrainer):
        def _setup_train(self) -> None:
            super()._setup_train()
            model = unwrap_model(self.model)
            criterion = getattr(model, "criterion", None)
            if criterion is None and hasattr(model, "init_criterion"):
                criterion = model.init_criterion()
                model.criterion = criterion
            criterion_name = type(criterion).__name__ if criterion is not None else ""
            if criterion_name in ("E2ELoss", "E2EDetectLoss"):
                model.criterion = E2EDfdLoss(
                    model,
                    coarse_gain=coarse_gain,
                    coarse_mapping=mapping,
                    presence_gain=presence_gain,
                    dfd_gain=dfd_gain,
                    dfd_sigma_scale=dfd_sigma_scale,
                    dfd_focal_alpha=dfd_focal_alpha,
                    dfd_neg_gamma=dfd_neg_gamma,
                )
            else:
                model.criterion = DfdPresenceLoss(
                    model,
                    coarse_gain=coarse_gain,
                    coarse_mapping=mapping,
                    presence_gain=presence_gain,
                    dfd_gain=dfd_gain,
                    dfd_sigma_scale=dfd_sigma_scale,
                    dfd_focal_alpha=dfd_focal_alpha,
                    dfd_neg_gamma=dfd_neg_gamma,
                )

    _DfdTrainer.__name__ = "DfdTrainer"
    return _DfdTrainer
