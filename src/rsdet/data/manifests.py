"""数据清单管理。

TODO: 数据集审计完成后实现 manifest 和 checksum 功能。
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class DataManifest:
    """数据集清单。

    Attributes:
        version: 数据集版本标识。
        num_images: 图像总数。
        class_distribution: 各类别实例数统计。
        checksums: 文件路径 → checksum 映射。
    """
    version: str = "TBD"
    num_images: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    checksums: Dict[str, str] = field(default_factory=dict)
