"""随机种子设置工具。"""

import logging
import random

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """设置 Python random 和 numpy 的随机种子。

    Args:
        seed: 随机种子值。
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass  # numpy 未安装时跳过

    logger.info(f"随机种子已设为 {seed}")
