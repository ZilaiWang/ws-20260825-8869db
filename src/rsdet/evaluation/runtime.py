"""推理阶段计时工具。

统计完整推理流程各阶段耗时，不把 model forward 时间冒充 total。
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict


@dataclass
class RuntimeBreakdown:
    """各阶段耗时（秒）。"""

    preprocess_after_read: float = 0.0
    tiling: float = 0.0
    model: float = 0.0
    postprocess: float = 0.0
    coordinate_restore: float = 0.0
    serialization: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.preprocess_after_read
            + self.tiling
            + self.model
            + self.postprocess
            + self.coordinate_restore
            + self.serialization
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "preprocess_after_read": self.preprocess_after_read,
            "tiling": self.tiling,
            "model": self.model,
            "postprocess": self.postprocess,
            "coordinate_restore": self.coordinate_restore,
            "serialization": self.serialization,
            "total": self.total,
        }


@contextmanager
def timed_block(runtime: RuntimeBreakdown, attr: str):
    """记录某阶段耗时的上下文管理器。

    Args:
        runtime: RuntimeBreakdown 实例。
        attr: 要更新的属性名。

    Example:
        with timed_block(rt, "tiling"):
            tiles = generate_tiles(...)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        setattr(runtime, attr, getattr(runtime, attr) + elapsed)
