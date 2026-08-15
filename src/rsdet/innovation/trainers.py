"""自定义 Trainer 工厂（Y3/Y4/Y5 训练期模块的注入入口）。

三个工厂返回 ``DetectionTrainer`` 子类，通过 ``model.train(trainer=...)`` 注入：

- :func:`hierarchical_trainer` —— Y3 层次粗细类辅助损失；
- :func:`afss_trainer` —— Y4 AFSS 反遗忘采样器；
- :func:`rotate90_augmentations` —— Y5 旋转增强（无需自定义 trainer，直接作
  ``train(augmentations=...)`` 参数传入）。

闭包捕获超参，避免依赖 ultralytics 的参数校验系统。本模块顶层 import
ultralytics（属训练期模块），``rsdet.innovation.__init__`` 不 import 本模块。
"""

from __future__ import annotations

from typing import Any, Sequence


def hierarchical_trainer(coarse_gain: float = 0.5, coarse_mapping: Sequence[int] | None = None) -> type:
    """返回带层次粗类辅助损失的 ``DetectionTrainer`` 子类（Y3）。

    Args:
        coarse_gain: 粗类辅助损失相对主 cls 损失的权重。
        coarse_mapping: 细类 -> 粗类索引（默认用 25 类冻结映射，长度须 = model.nc）。

    Returns:
        可传入 ``model.train(trainer=...)`` 的 trainer 类。
    """
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    from rsdet.innovation.coarse import COARSE_MAPPING
    from rsdet.innovation.hierarchical_loss import HierarchicalCoarseLoss

    mapping = tuple(coarse_mapping) if coarse_mapping is not None else COARSE_MAPPING

    class _HierarchicalTrainer(DetectionTrainer):
        def _setup_train(self) -> None:
            super()._setup_train()
            model = unwrap_model(self.model)
            model.criterion = HierarchicalCoarseLoss(model, coarse_gain=coarse_gain, coarse_mapping=mapping)

    _HierarchicalTrainer.__name__ = "HierarchicalTrainer"
    return _HierarchicalTrainer


def afss_trainer(suff_list: Sequence[float], easy_floor: float = 0.05) -> type:
    """返回带 AFSS 反遗忘采样器的 ``DetectionTrainer`` 子类（Y4）。

    Args:
        suff_list: 每图充分度，顺序须与训练 dataset 索引一致（= split_view 中
            ``split=="train"`` 的样本顺序）。
        easy_floor: 容易图最低回看权重。

    Returns:
        可传入 ``model.train(trainer=...)`` 的 trainer 类。
    """
    from ultralytics.data.build import InfiniteDataLoader, seed_worker
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import torch_distributed_zero_first

    from rsdet.innovation.afss_sampler import AFSSSampler

    class _AFSSTrainer(DetectionTrainer):
        def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
            if mode != "train":
                return super().get_dataloader(dataset_path, batch_size, rank, mode)
            with torch_distributed_zero_first(rank):
                dataset = self.build_dataset(dataset_path, mode, batch_size)
            if len(suff_list) != len(dataset):
                raise ValueError(
                    f"suff_list 长度 {len(suff_list)} != dataset 大小 {len(dataset)}；"
                    "须按 split_view 中 split=='train' 的样本顺序对齐"
                )
            generator = __import__("torch").Generator()
            generator.manual_seed(int(getattr(self.args, "seed", 0)))
            sampler = AFSSSampler(suff_list, num_samples=len(dataset), easy_floor=easy_floor, generator=generator)
            nw = self.args.workers
            return InfiniteDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=nw,
                sampler=sampler,
                pin_memory=False,
                collate_fn=getattr(dataset, "collate_fn", None),
                worker_init_fn=seed_worker,
                drop_last=False,
            )

    _AFSSTrainer.__name__ = "AFSSTrainer"
    return _AFSSTrainer


def rotate90_augmentations(p: float = 1.0) -> list[Any]:
    """返回 Y5 旋转增强的 albumentations 列表，作 ``train(augmentations=...)`` 参数。

    Args:
        p: 应用 90° 旋转的概率。

    Returns:
        ``[albumentations.RandomRotate90(p=p)]``。
    """
    from rsdet.innovation.rotate90 import build_rotate90_augmentations

    return build_rotate90_augmentations(p=p)
