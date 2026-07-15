"""简单模型注册表。

用法：
    @register_model("my_detector")
    class MyDetector(BaseDetector):
        ...

    detector = build_model("my_detector", config)
"""

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Type

from rsdet.contracts import InferenceSample, Prediction
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


def build_model_from_config(model_config: str | Mapping[str, Any]) -> BaseDetector:
    """从最小模型配置构建检测器。

    为降低接入成本，配置既可以直接写注册名，也可以使用映射：

    ``{"name": "my_detector", "module": "pkg.adapter", "init_args": {...}}``

    ``module`` 可省略；提供时会先导入该模块，使第三方适配器完成注册。
    本函数只构建模型，不强制统一训练、权重加载或预处理流程。
    """
    if isinstance(model_config, str):
        return build_model(model_config, {})
    if not isinstance(model_config, Mapping):
        raise TypeError("model 配置必须是注册名字符串或映射")

    module_name = model_config.get("module")
    if module_name:
        importlib.import_module(str(module_name))

    name = model_config.get("name")
    if not name:
        raise ValueError("model 配置缺少 name")
    init_args = model_config.get("init_args", {})
    if not isinstance(init_args, Mapping):
        raise TypeError("model.init_args 必须是映射")
    return build_model(str(name), {"init_args": dict(init_args)})


# -------------------- DummyDetector (仅用于测试) --------------------


@register_model("dummy")
class DummyDetector(BaseDetector):
    """占位检测器，仅用于单元测试和接口验证，非实际基线。"""

    def __init__(self, **kwargs):
        self._device = "cpu"
        self._loaded = False

    def load(self, checkpoint_path: str) -> None:
        self._loaded = True

    def predict(self, batch: Sequence[InferenceSample]) -> list[Prediction]:
        return [
            Prediction(image_id=sample.image_id, boxes_xyxy=[], scores=[], labels=[])
            for sample in batch
        ]

    def to(self, device: str) -> None:
        self._device = device

    def eval(self) -> None:
        pass
