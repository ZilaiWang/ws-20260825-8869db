"""日志配置工具。"""

import logging
import sys


def setup_logging(level: int = logging.INFO, name: str = "rsdet") -> logging.Logger:
    """配置控制台日志输出。

    Args:
        level: 日志级别。
        name: logger 名称。

    Returns:
        配置好的 Logger 实例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
