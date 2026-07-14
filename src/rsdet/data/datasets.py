"""数据集辅助工具。

TODO: 数据集结构审计完成后实现具体加载逻辑。
"""


def check_data_root(data_root: str) -> bool:
    """检查数据根目录是否存在。"""
    from pathlib import Path

    p = Path(data_root)
    if not p.exists():
        raise FileNotFoundError(f"数据根目录不存在: {data_root}")
    return True
