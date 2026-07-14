"""抽象检测器基类。

所有检测器实现必须继承 BaseDetector，不绑定具体模型。
"""

from abc import ABC, abstractmethod
from typing import List

from rsdet.contracts import Prediction


class BaseDetector(ABC):
    """模型无关的检测器抽象基类。

    predict() 返回 Prediction 列表以保证下游模块接口统一。
    """

    @abstractmethod
    def load(self, checkpoint_path: str) -> None:
        """加载模型权重。

        Args:
            checkpoint_path: 权重文件路径。
        """
        ...

    @abstractmethod
    def predict(self, batch: list) -> List[Prediction]:
        """对一批图像执行推理。

        Args:
            batch: 图像数据列表，格式取决于具体实现。

        Returns:
            Prediction 列表，长度等于 batch 长度。
        """
        ...

    @abstractmethod
    def train_step(self) -> None:  # pragma: no cover
        """单步训练接口（预留）。

        TODO: 基线模型确定后定义具体签名。
        """
        raise NotImplementedError("train_step 尚未实现，待基线模型确定")

    @abstractmethod
    def to(self, device: str) -> None:
        """将模型移至指定设备。

        Args:
            device: 设备字符串，如 "cuda" 或 "cpu"。
        """
        ...

    @abstractmethod
    def eval(self) -> None:
        """将模型设置为评估模式。"""
        ...
