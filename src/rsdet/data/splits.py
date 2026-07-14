"""训练/验证/测试划分管理。

TODO: 数据划分策略确定后实现。
"""

from pathlib import Path
from typing import Dict, List


def save_split_manifest(
    train_ids: List[int],
    val_ids: List[int],
    test_ids: List[int],
    output_path: Path,
) -> None:
    """保存划分清单为 JSON。

    Args:
        train_ids: 训练集 image_id 列表。
        val_ids: 验证集 image_id 列表。
        test_ids: 测试集 image_id 列表。
        output_path: 输出 JSON 路径。
    """
    import json

    manifest: Dict[str, List[int]] = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
