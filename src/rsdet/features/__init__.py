"""P04 训练期教师特征工具。

该包保持核心导入不强依赖 PyTorch/diffusers，便于本地先完成
manifest、缓存和统计代码的单元测试。重模型依赖只在服务器实际构建
teacher adapter 时延迟导入。
"""

from rsdet.features.p04_cache import FeatureCache, FeatureCacheWriter
from rsdet.features.p04_inputs import D4_VIEW_IDS, apply_d4_view, load_all_crop_records

__all__ = [
    "D4_VIEW_IDS",
    "FeatureCache",
    "FeatureCacheWriter",
    "apply_d4_view",
    "load_all_crop_records",
]
