"""SCOPE: Set-wise Counterfactual Optimization with Predictive-risk Experts."""

from .actions import Action, ActionKind, apply_action
from .calibration import GroupwiseConformalLCB
from .counterfactual import CounterfactualLabelBuilder, CounterfactualRecord
from .decode import safe_greedy_decode
from .provenance import OOFRecord, validate_cross_fit

# model.py 依赖 torch；标签生成/校准/解码阶段不需要 torch，
# 因此惰性导入，避免无 torch 环境（仅 CPU 离线标签）无法使用其余模块。
try:  # pragma: no cover - torch 可用性由环境决定
    from .model import RelationalSetController

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    RelationalSetController = None  # type: ignore[assignment]
    _HAS_TORCH = False

__all__ = [
    "Action",
    "ActionKind",
    "apply_action",
    "GroupwiseConformalLCB",
    "CounterfactualLabelBuilder",
    "CounterfactualRecord",
    "safe_greedy_decode",
    "RelationalSetController",
    "OOFRecord",
    "validate_cross_fit",
]
