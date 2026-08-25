"""Y4 AFSS 反遗忘采样器（材料 19 第四优先）。

用每图 min(precision, recall) 作为学习充分度（由 ``afss_diagnose.py`` 离线
计算），按充分度加权采样：困难图（低充分度）持续参与、中等图短期覆盖、
容易图（充分度=1）低频回看防止遗忘。

充分度 -> 权重映射（可审计）::

    weight = max(1 - suff, easy_floor)

即充分度 0（完全失败）权重 1.0、充分度 0.5 权重 0.5、充分度 1.0（完美）权重
被压到 ``easy_floor``（默认 0.05，即容易图仅 5% 概率被回看）。

材料 19 要求先做日志回放诊断（已完成 ``afss_diagnose.py``），相关性成立后才
接入训练期采样器——本模块是训练期采样器的实现。
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import WeightedRandomSampler


def suff_to_weight(suff: float, easy_floor: float = 0.05) -> float:
    """充分度 -> 采样权重。充分度越低权重越高，容易图压到 easy_floor。

    Args:
        suff: 每图 min(precision, recall)，范围 [0, 1]。
        easy_floor: 容易图的最低回看权重（默认 0.05）。

    Returns:
        采样权重，范围 [easy_floor, 1.0]。
    """
    return max(float(1.0 - suff), float(easy_floor))


class AFSSSampler(WeightedRandomSampler):
    """按学习充分度加权的无限采样器（继承 ``WeightedRandomSampler``）。

    ``replacement=True`` 保证每个 epoch 可采样 ``num_samples`` 次（默认等于
    数据集大小），配合 ultralytics 的 ``InfiniteDataLoader`` 无限循环。
    """

    def __init__(
        self,
        suff_list: Sequence[float],
        num_samples: int | None = None,
        easy_floor: float = 0.05,
        generator: torch.Generator | None = None,
    ) -> None:
        """初始化。

        Args:
            suff_list: 每图充分度，顺序须与 dataset 索引一致（长度 = len(dataset)）。
            num_samples: 每 epoch 采样数，默认 = len(suff_list)。
            easy_floor: 容易图最低回看权重。
            generator: 随机数生成器（可复现）。
        """
        if len(suff_list) == 0:
            raise ValueError("suff_list 不能为空")
        weights = [suff_to_weight(float(s), easy_floor) for s in suff_list]
        if num_samples is None:
            num_samples = len(suff_list)
        super().__init__(
            weights=torch.as_tensor(weights, dtype=torch.float64),
            num_samples=int(num_samples),
            replacement=True,
            generator=generator,
        )
        self.suff_list = [float(s) for s in suff_list]
        self.easy_floor = float(easy_floor)
