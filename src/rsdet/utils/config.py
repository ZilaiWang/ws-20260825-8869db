"""配置加载工具。

支持 YAML 加载和 local.yaml 覆盖。
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """加载 YAML 配置文件。

    Args:
        config_path: 配置文件路径。

    Returns:
        配置字典。

    Raises:
        FileNotFoundError: 文件不存在。
        yaml.YAMLError: YAML 解析错误。
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    logger.info(f"已加载配置: {config_path}")
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并配置，override 中的值覆盖 base。

    Args:
        base: 基础配置。
        override: 覆盖配置。

    Returns:
        合并后的新字典。
    """
    import copy

    result = copy.deepcopy(base)

    def _merge(dst: dict, src: dict) -> None:
        for key, value in src.items():
            if key in dst and isinstance(dst[key], dict) and isinstance(value, dict):
                _merge(dst[key], value)
            else:
                dst[key] = value

    _merge(result, override)
    return result
