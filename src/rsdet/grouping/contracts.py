"""MAR20 分组协议的稳定合同和通用哈希函数。"""

from __future__ import annotations

import hashlib
import json
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageOps

PROTOCOL_VERSION = "mar20-source-grouping-v1.1"
MASKED_PATCH_PROTOCOL_VERSION = "mar20-source-grouping-v1.2"
REGISTRY_SCHEMA_VERSION = "mar20-image-registry-v1"
ANNOTATION_SCHEMA_VERSION = "mar20-image-annotations-v1"
FEATURE_CACHE_SCHEMA_VERSION = "mar20-place-feature-cache-v1"

_NODE_PATTERN = re.compile(r"^mar20:([1-9][0-9]*)$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class Scope(str, Enum):
    """分组图节点范围。"""

    TARGET_ONLY = "target_only"
    FULL_BRIDGE = "full_bridge"


class EvidenceLabel(str, Enum):
    """人工/机器共用的冻结证据语义。"""

    SAME_FRAME = "same_frame"
    GEOMETRIC_OVERLAP = "geometric_overlap"
    SAME_LOCAL_SITE = "same_local_site"
    LIKELY_SAME_AIRPORT = "likely_same_airport"
    NOT_SAME_LOCAL_SITE = "not_same_local_site"
    DIFFERENT_AIRPORT = "different_airport"
    UNCERTAIN = "uncertain"


STRICT_LABELS = frozenset(
    {
        EvidenceLabel.SAME_FRAME,
        EvidenceLabel.GEOMETRIC_OVERLAP,
        EvidenceLabel.SAME_LOCAL_SITE,
    }
)


def sha256_file(path: str | Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Mapping[str, Any] | list[Any]) -> str:
    """对 JSON 可序列化值做稳定哈希。"""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def canonical_pixel_sha256(image: Image.Image) -> str:
    """按 EXIF 朝向纠正后的完整 RGB 像素和尺寸生成 H0 哈希。"""

    rgb = ImageOps.exif_transpose(image).convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"RGB|{rgb.width}|{rgb.height}|".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def canonical_node_uid(mar20_number: int) -> str:
    if mar20_number <= 0:
        raise ValueError("MAR20 编号必须为正整数")
    return f"mar20:{mar20_number}"


def parse_node_uid(node_uid: str) -> int:
    matched = _NODE_PATTERN.fullmatch(node_uid)
    if not matched:
        raise ValueError(f"非法 MAR20 node_uid: {node_uid!r}")
    return int(matched.group(1))


def canonical_pair_uid(node_u: str, node_v: str) -> str:
    """生成无向 pair 的稳定 UID。"""

    parse_node_uid(node_u)
    parse_node_uid(node_v)
    if node_u == node_v:
        raise ValueError("pair 两端不得相同")
    left, right = sorted((node_u, node_v), key=parse_node_uid)
    return f"{left}--{right}"


def validate_sha256(value: str, *, field: str = "sha256") -> str:
    normalized = value.strip().lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field} 不是 64 位十六进制 SHA-256")
    return normalized


def resolve_within(root: str | Path, path: str | Path) -> Path:
    """解析路径并禁止逃逸只读数据根目录。"""

    resolved_root = Path(root).expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = resolved_root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"路径逃逸根目录: root={resolved_root}, path={candidate}") from error
    return candidate


def atomic_write_json(path: str | Path, value: Mapping[str, Any] | list[Any]) -> None:
    """以临时文件+replace 写出 JSON。"""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
