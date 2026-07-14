"""简单模型注册表。

用法：
    @register_model("my_detector")
    class MyDetector(BaseDetector):
        ...

    detector = build_model("my_detector", config)
"""

from typing import Any, Dict, Type

from rsdet.models.base import BaseDetector

_MODEL_REGISTRY: Dict[str, Type[BaseDetector]] = {}


def register_model(name: str):
    """模型注册装饰器。

    Args:
        name: 模型名称，用于 --config model=name 引用。
    """
    def decorator(cls: Type[BaseDetector]) -> Type[BaseDetector]:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"模型 '{name}' 已注册，请使用不同名称")
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def build_model(name: str, config: Dict[str, Any]) -> BaseDetector:
    """根据名称和配置构建检测器。

    Args:
        name: 注册的模型名称。
        config: 模型配置字典。

    Returns:
        BaseDetector 实例。

    Raises:
        KeyError: 模型名称未注册。
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys()) or "（无）"
        raise KeyError(f"未找到模型 '{name}'。已注册: {available}")
    cls = _MODEL_REGISTRY[name]
    return cls(**config.get("init_args", {}))


def list_models() -> Dict[str, Type[BaseDetector]]:
    """返回已注册模型字典（只读副本）。"""
    return dict(_MODEL_REGISTRY)


# -------------------- DummyDetector (仅用于测试) --------------------

@register_model("dummy")
class DummyDetector(BaseDetector):
    """占位检测器，仅用于单元测试和接口验证，非实际基线。"""

    def __init__(self, **kwargs):
        self._device = "cpu"
        self._loaded = False

    def load(self, checkpoint_path: str) -> None:
        self._loaded = True

    def predict(self, batch: list) -> list:
        from rsdet.contracts import Prediction

        return [
            Prediction(image_id=i, boxes_xyxy=[], scores=[], labels=[])
            for i in range(len(batch))
        ]

    def train_step(self) -> None:
        raise NotImplementedError("DummyDetector 不支持训练")

    def to(self, device: str) -> None:
        self._device = device

    def eval(self) -> None:
        pass
