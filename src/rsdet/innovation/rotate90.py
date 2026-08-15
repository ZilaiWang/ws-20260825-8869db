"""Y5 90° 离散旋转增强（材料 19 第五优先）。

通过 ultralytics 内置的 Albumentations 扩展点（``hyp.augmentations``）注入
``albumentations.RandomRotate90``：在 0/90/180/270° 中随机旋转，HBB 精确变换、
bbox 由 ultralytics 的 ``Albumentations`` 包装类同步，适合飞机/舰船任意朝向。

材料 19 的两种选择：①0/90/180/270° 离散旋转增强；②旋转前后一致性损失。
本模块实现低风险的 ①（增强），②作为后续进阶（需两次前向，成本更高）。

注意：仅在来源隔离 CV3 上评估，并单独检查小目标插值损失；若旋转一致性改善
尾类但伤害车辆 Recall，则只用于飞机/舰船训练，不进共享主干。
"""

from __future__ import annotations

from typing import Any


def build_rotate90_augmentations(p: float = 1.0) -> list[Any]:
    """构建 90° 离散旋转的 albumentations transform 列表（供 ``hyp.augmentations``）。

    Args:
        p: 应用旋转的概率（默认 1.0，每图都做一次 0/90/180/270 旋转）。

    Returns:
        ``[albumentations.RandomRotate90(p=p)]``。
    """
    import albumentations as A

    return [A.RandomRotate90(p=p)]


def resolve_rotate90(rotation: str | float | None, p: float = 1.0) -> list[Any] | None:
    """按配置解析旋转增强。rotation 为真值（``true``/``1``/``"rotate90"``）时启用。

    Args:
        rotation: 配置开关。None/False/"none" 表示禁用。
        p: 旋转概率。

    Returns:
        albumentations transform 列表，或 None（禁用）。
    """
    if rotation is None or rotation is False:
        return None
    if isinstance(rotation, str) and rotation.strip().lower() in {"none", "off", "false", "0"}:
        return None
    return build_rotate90_augmentations(p=p)
