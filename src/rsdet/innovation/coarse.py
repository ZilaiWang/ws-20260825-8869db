"""25 细类 -> 3 粗类映射（材料 19 Y3 层次损失的依据）。

官方 V1.6 口径下 25 个细类归属 3 个粗类（ship / aircraft / vehicle）：
细类 0~3 为舰船，4~23 为飞机（20 类），24 为车辆（1 类）。

映射关系与 ``configs/project.yaml`` 的 ``category_mapping`` 一致，冻结：
    ship(0-3) / aircraft(4-23) / vehicle(24)
"""

from __future__ import annotations

import torch

# 粗类名（顺序与 protocol.class_names 一致）
COARSE_NAMES: tuple[str, ...] = ("ship", "aircraft", "vehicle")

# 25 个细类 -> 粗类索引（4 舰船 + 20 飞机 + 1 车辆）
COARSE_MAPPING: tuple[int, ...] = (0, 0, 0, 0) + (1,) * 20 + (2,)

# 细类总数（冻结 25）
N_FINE_CLASSES: int = 25

# ---- F2 属性组合的 family 中间层（9 家族, 方案 §10.1）----
# 按机型家族 + 兄弟混淆分组: 让兄弟类同家族(家族内细判), 跨家族先粗分。
# F0 ship-large / F1 ship-combat / F2 su-jet / F3 fighter / F4 tupolev /
# F5 us-bomber / F6 transport / F7 awacs-refuel / F8 vehicle
FAMILY_NAMES: tuple[str, ...] = (
    "ship-large", "ship-combat", "su-jet", "fighter", "tupolev",
    "us-bomber", "transport", "awacs-refuel", "vehicle",
)

# 25 细类 -> family 索引(0~8)
FAMILY_MAPPING: tuple[int, ...] = (
    0, 0, 1, 1,          # HM, LQS, QHS, MS
    2,                   # A1 SU-35
    6, 6, 6,             # A2 C-130, A3 C-17, A4 C-5
    3,                   # A5 F-16
    4,                   # A6 TU-160
    7,                   # A7 E-3
    5,                   # A8 B-52
    7,                   # A9 P-3C
    5,                   # A10 B-1B
    7,                   # A11 E-8
    4,                   # A12 TU-22
    3,                   # A13 F-15
    7,                   # A14 KC-135
    3,                   # A15 F-22
    3,                   # A16 FA-18
    4,                   # A17 TU-95
    7,                   # A18 KC-10
    2,                   # A19 SU-34
    2,                   # A20 SU-24
    8,                   # FSC
)


def build_coarse_matrix(
    nc: int = N_FINE_CLASSES,
    mapping: tuple[int, ...] = COARSE_MAPPING,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """构建 (nc, n_coarse) 的 0/1 归属矩阵 ``M``，满足 ``coarse_logit = fine_logit @ M``。

    Args:
        nc: 细类数，默认 25。
        mapping: 细类 -> 粗类索引（长度须等于 nc）。
        device: 返回张量的设备。

    Returns:
        ``torch.Tensor``，形状 ``(nc, n_coarse)``，元素为 0/1。
    """
    if len(mapping) != nc:
        raise ValueError(f"mapping 长度 {len(mapping)} != nc {nc}")
    n_coarse = max(mapping) + 1
    mat = torch.zeros(nc, n_coarse, dtype=torch.float32, device=device)
    for fine, coarse in enumerate(mapping):
        mat[fine, coarse] = 1.0
    return mat
