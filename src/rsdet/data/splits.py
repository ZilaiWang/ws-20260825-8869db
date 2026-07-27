"""训练/验证/交叉验证划分管理。

产出格式遵循 docs/INTEGRATION_CONTRACT.md 第 4 节（contract_v1）。
分组依据见 reports/data/ 下的划分说明。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

DATA_VERSION = "official_raw_v1"


def save_split_manifest(
    samples: Iterable[Mapping[str, Any]],
    output_path: Path,
    *,
    version: str,
    data_version: str = DATA_VERSION,
    **extra: Any,
) -> None:
    """按 contract_v1 保存 split manifest。

    Args:
        samples: 每项至少含 image_id、relative_path、group_id，
            并含 split（train/val）或 fold（折号）之一。
        output_path: 输出 JSON 路径。
        version: 划分版本号，如 ``dev_v1``、``cv3_v1``。
        data_version: 数据版本号。
        extra: 写入 manifest 顶层的附加字段（seed、fold_count 等）。
    """
    manifest: dict[str, Any] = {"version": version, "data_version": data_version}
    manifest.update(extra)
    manifest["samples"] = list(samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
