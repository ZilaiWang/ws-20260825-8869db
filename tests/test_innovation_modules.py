"""创新训练期模块（Y3/Y4/Y5）的单元测试。

Y3 层次损失的前向验证依赖 ultralytics（构造最小 DetectionModel 跑 loss），
其余（coarse 映射 / AFSS 采样 / 旋转增强）为纯逻辑，不依赖 GPU。
"""

from __future__ import annotations

import torch

from rsdet.innovation.afss_sampler import AFSSSampler, suff_to_weight
from rsdet.innovation.coarse import (
    COARSE_MAPPING,
    COARSE_NAMES,
    N_FINE_CLASSES,
    build_coarse_matrix,
)


def test_coarse_mapping() -> None:
    assert N_FINE_CLASSES == 25
    assert len(COARSE_MAPPING) == 25
    assert COARSE_NAMES == ("ship", "aircraft", "vehicle")
    assert COARSE_MAPPING[:4] == (0, 0, 0, 0)  # 舰船 4 类
    assert COARSE_MAPPING[4:24] == (1,) * 20  # 飞机 20 类
    assert COARSE_MAPPING[24] == 2  # 车辆 1 类


def test_coarse_matrix() -> None:
    m = build_coarse_matrix()
    assert m.shape == (25, 3)
    assert float(m.sum()) == 25.0  # 每细类恰好归属一个粗类
    assert m[0, 0] == 1 and m[0, 1] == 0 and m[0, 2] == 0  # 细类0 -> ship
    assert m[4, 1] == 1  # 细类4 -> aircraft
    assert m[24, 2] == 1  # 细类24 -> vehicle


def test_suff_to_weight() -> None:
    assert suff_to_weight(0.0) == 1.0  # 完全失败 -> 权重最高
    assert suff_to_weight(0.5) == 0.5
    assert suff_to_weight(1.0) == 0.05  # 完美 -> 压到 easy_floor
    assert suff_to_weight(1.0, easy_floor=0.1) == 0.1


def test_afss_sampler() -> None:
    sampler = AFSSSampler([0.0, 0.5, 1.0], num_samples=3)
    assert list(sampler.weights) == [1.0, 0.5, 0.05]
    samples = list(sampler)
    assert len(samples) == 3
    assert all(0 <= s < 3 for s in samples)


def test_rotate90() -> None:
    from rsdet.innovation.rotate90 import build_rotate90_augmentations, resolve_rotate90

    augs = build_rotate90_augmentations(p=1.0)
    assert len(augs) == 1
    assert resolve_rotate90(None) is None
    assert resolve_rotate90(False) is None
    assert resolve_rotate90("none") is None
    assert len(resolve_rotate90("rotate90")) == 1


def test_hierarchical_loss_forward() -> None:
    """Y3 层次损失：保持 3 分量，粗类辅助并入 cls（box/dfl 不变、cls 增大）。"""
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import DEFAULT_CFG, IterableSimpleNamespace
    from ultralytics.utils.loss import v8DetectionLoss

    from rsdet.innovation.hierarchical_loss import HierarchicalCoarseLoss

    model = DetectionModel("yolo11n.yaml", nc=25, verbose=False)
    model.train()
    model.args = IterableSimpleNamespace(**vars(DEFAULT_CFG))  # 模拟 trainer.set_model_attributes
    torch.manual_seed(0)
    batch = {
        "img": torch.randn(2, 3, 640, 640),
        "batch_idx": torch.tensor([0, 0, 1]),  # targets 须按 batch_idx 排序
        "cls": torch.tensor([0, 24, 5]),
        "bboxes": torch.tensor([[100.0, 100.0, 200.0, 200.0], [50.0, 50.0, 150.0, 150.0], [300.0, 300.0, 400.0, 400.0]]),
    }

    # 同一模型先跑基线再跑层次，避免随机初始化差异
    model.criterion = v8DetectionLoss(model)
    loss_base, _ = model.loss(batch)
    model.criterion = HierarchicalCoarseLoss(model, coarse_gain=0.5)
    loss_hier, loss_items = model.loss(batch)

    assert tuple(loss_base.shape) == (3,)
    assert tuple(loss_hier.shape) == (3,)  # 保持 3 分量，兼容 validator/EMA
    assert tuple(loss_items.shape) == (3,)
    assert torch.allclose(loss_hier[0], loss_base[0], atol=1e-5)  # box 不变
    assert torch.allclose(loss_hier[2], loss_base[2], atol=1e-5)  # dfl 不变
    assert float(loss_hier[1].detach()) >= float(loss_base[1].detach())  # cls 含粗类辅助
    loss_hier.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
