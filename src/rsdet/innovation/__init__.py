"""创新实验训练期模块（材料 19 Y1~Y5）。

本包存放"改 YOLO 训练"的复用模块，仅在训练期存在，不改变正式推理接口：

- :mod:`rsdet.innovation.coarse` —— 25 细类 -> 3 粗类映射（Y3 层次损失用）；
- :mod:`rsdet.innovation.hierarchical_loss` —— Y3 层次粗细类辅助损失（需 ultralytics）；
- :mod:`rsdet.innovation.afss_sampler` —— Y4 AFSS 反遗忘采样器；
- :mod:`rsdet.innovation.rotate90` —— Y5 90° 离散旋转增强/一致性；
- :mod:`rsdet.innovation.trainers` —— 自定义 Trainer 注入（trainer 工厂）。

仅 ``coarse`` 无 ultralytics 依赖，随包顶层导入；其余模块在需要时显式 import，
以保持包在无深度学习环境下可导入。
"""

from rsdet.innovation.coarse import (
    COARSE_MAPPING,
    COARSE_NAMES,
    N_FINE_CLASSES,
    build_coarse_matrix,
)

__all__ = ["COARSE_MAPPING", "COARSE_NAMES", "N_FINE_CLASSES", "build_coarse_matrix"]
