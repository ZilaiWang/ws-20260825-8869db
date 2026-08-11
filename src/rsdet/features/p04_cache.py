"""P04 特征分片缓存、断点续跑和完整性审计。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CACHE_SCHEMA_VERSION = "p04_feature_cache_v1"
REQUIRED_ROW_KEYS = (
    "annotation_uid",
    "crop_id",
    "canonical_input_sha256",
    "view_id",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass(frozen=True)
class CacheShard:
    path: str
    row_count: int
    sha256: str
    feature_shapes: dict[str, list[int]]
    storage_dtypes: dict[str, str]


class FeatureCacheWriter:
    """以固定行顺序写入 NumPy shard，并为 resume 保存单片 sidecar。"""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        metadata: Mapping[str, Any],
        feature_names: Sequence[str],
        storage_dtype: str = "float16",
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if storage_dtype not in {"float16", "float32"}:
            raise ValueError("storage_dtype 只能是 float16 或 float32")
        names = tuple(feature_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("feature_names 必须非空且唯一")
        if any(not name or not name.replace("_", "").isalnum() for name in names):
            raise ValueError("feature name 只允许字母、数字和下划线")
        self.metadata = dict(metadata)
        self.feature_names = names
        self.storage_dtype = storage_dtype
        self.config_fingerprint = stable_json_sha256(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "metadata": self.metadata,
                "feature_names": list(self.feature_names),
                "storage_dtype": storage_dtype,
            }
        )
        root_meta = self.output_dir / "cache_meta.json"
        if root_meta.exists():
            existing = json.loads(root_meta.read_text(encoding="utf-8"))
            if existing.get("config_fingerprint") != self.config_fingerprint:
                raise ValueError("现有 cache_meta 与本次配置不一致，禁止混写")
        else:
            _atomic_json(
                root_meta,
                {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "config_fingerprint": self.config_fingerprint,
                    "metadata": self.metadata,
                    "feature_names": list(self.feature_names),
                    "storage_dtype": self.storage_dtype,
                },
            )

    def _paths(self, shard_index: int) -> tuple[Path, Path]:
        stem = f"shard-{shard_index:05d}"
        return self.output_dir / f"{stem}.npz", self.output_dir / f"{stem}.json"

    def valid_existing_shard(self, shard_index: int, expected_rows: int) -> bool:
        data_path, sidecar_path = self._paths(shard_index)
        if not data_path.is_file() or not sidecar_path.is_file():
            return False
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return bool(
            sidecar.get("config_fingerprint") == self.config_fingerprint
            and sidecar.get("row_count") == expected_rows
            and sidecar.get("sha256") == sha256_file(data_path)
        )

    def write_shard(
        self,
        shard_index: int,
        rows: Mapping[str, Sequence[Any] | np.ndarray],
        features: Mapping[str, np.ndarray],
    ) -> CacheShard:
        if set(features) != set(self.feature_names):
            raise ValueError(
                f"feature keys 不匹配: expected={self.feature_names}, actual={sorted(features)}"
            )
        missing = set(REQUIRED_ROW_KEYS) - set(rows)
        if missing:
            raise ValueError(f"shard 行缺少字段: {sorted(missing)}")
        row_count = len(rows["annotation_uid"])
        if row_count <= 0:
            raise ValueError("禁止写入空 shard")
        arrays: dict[str, np.ndarray] = {}
        for key, values in rows.items():
            array = np.asarray(values)
            if array.shape[0] != row_count:
                raise ValueError(f"{key} 行数不一致")
            arrays[f"row__{key}"] = array
        feature_shapes: dict[str, list[int]] = {}
        storage_dtypes: dict[str, str] = {}
        dtype = np.float16 if self.storage_dtype == "float16" else np.float32
        for name in self.feature_names:
            array = np.asarray(features[name])
            if array.ndim != 2 or array.shape[0] != row_count:
                raise ValueError(f"feature {name} 必须是 [N,D]")
            if not np.isfinite(array).all():
                raise ValueError(f"feature {name} 含 NaN/Inf")
            stored = array.astype(dtype, copy=False)
            arrays[f"feature__{name}"] = stored
            feature_shapes[name] = [int(value) for value in stored.shape]
            storage_dtypes[name] = str(stored.dtype)

        data_path, sidecar_path = self._paths(shard_index)
        temporary = data_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as file:
            np.savez(file, **arrays)
        os.replace(temporary, data_path)
        checksum = sha256_file(data_path)
        sidecar = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "config_fingerprint": self.config_fingerprint,
            "shard_index": shard_index,
            "path": data_path.name,
            "row_count": row_count,
            "sha256": checksum,
            "feature_shapes": feature_shapes,
            "storage_dtypes": storage_dtypes,
        }
        _atomic_json(sidecar_path, sidecar)
        return CacheShard(
            path=data_path.name,
            row_count=row_count,
            sha256=checksum,
            feature_shapes=feature_shapes,
            storage_dtypes=storage_dtypes,
        )

    def finalize(self, expected_shards: int, expected_rows: int) -> dict[str, Any]:
        shards: list[dict[str, Any]] = []
        total_rows = 0
        for shard_index in range(expected_shards):
            data_path, sidecar_path = self._paths(shard_index)
            if not data_path.is_file() or not sidecar_path.is_file():
                raise FileNotFoundError(f"缺少 shard {shard_index}")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if sidecar.get("config_fingerprint") != self.config_fingerprint:
                raise ValueError(f"shard {shard_index} fingerprint 不一致")
            if sidecar.get("sha256") != sha256_file(data_path):
                raise ValueError(f"shard {shard_index} SHA-256 不一致")
            total_rows += int(sidecar["row_count"])
            shards.append(sidecar)
        if total_rows != expected_rows:
            raise ValueError(f"cache 总行数错误: expected={expected_rows}, actual={total_rows}")
        index = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "config_fingerprint": self.config_fingerprint,
            "row_count": total_rows,
            "shard_count": len(shards),
            "feature_names": list(self.feature_names),
            "storage_dtype": self.storage_dtype,
            "shards": shards,
        }
        _atomic_json(self.output_dir / "index.json", index)
        return index


class FeatureCache:
    """严格读取并审计 P04 特征缓存。"""

    def __init__(self, cache_dir: str | Path, *, verify_sha256: bool = True) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.index_path = self.cache_dir / "index.json"
        self.meta_path = self.cache_dir / "cache_meta.json"
        if not self.index_path.is_file() or not self.meta_path.is_file():
            raise FileNotFoundError(f"{self.cache_dir} 缺少 index.json/cache_meta.json")
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if self.index.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError("cache schema_version 不支持")
        if self.index.get("config_fingerprint") != self.meta.get("config_fingerprint"):
            raise ValueError("cache index/meta fingerprint 不一致")
        if verify_sha256:
            for shard in self.index["shards"]:
                path = self.cache_dir / shard["path"]
                if not path.is_file() or sha256_file(path) != shard["sha256"]:
                    raise ValueError(f"cache shard 缺失或 SHA 不一致: {path}")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.index["feature_names"])

    def load_feature(self, feature_name: str) -> dict[str, np.ndarray]:
        if feature_name not in self.feature_names:
            raise ValueError(f"未知 feature_name={feature_name!r}")
        collected: dict[str, list[np.ndarray]] = {}
        for shard in self.index["shards"]:
            with np.load(self.cache_dir / shard["path"], allow_pickle=False) as payload:
                keys = [key for key in payload.files if key.startswith("row__")]
                keys.append(f"feature__{feature_name}")
                for key in keys:
                    output_key = key.removeprefix("row__").removeprefix("feature__")
                    collected.setdefault(output_key, []).append(np.asarray(payload[key]))
        result = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
        expected = int(self.index["row_count"])
        if any(value.shape[0] != expected for value in result.values()):
            raise ValueError("cache 读取后行数不一致")
        return result

    def audit(self) -> dict[str, Any]:
        seen: set[tuple[str, str, str, str]] = set()
        seen_annotation_views: set[tuple[str, str]] = set()
        contract_by_annotation: dict[str, tuple[str, str]] = {}
        views_by_annotation: dict[str, set[str]] = {}
        dimensions: dict[str, set[int]] = {name: set() for name in self.feature_names}
        row_count = 0
        nonfinite = 0
        for shard in self.index["shards"]:
            with np.load(self.cache_dir / shard["path"], allow_pickle=False) as payload:
                arrays = {key: np.asarray(payload[key]) for key in payload.files}
            n = arrays["row__annotation_uid"].shape[0]
            row_count += n
            for offset in range(n):
                uid, crop_id, input_sha, view_id = (
                    str(arrays[f"row__{name}"][offset]) for name in REQUIRED_ROW_KEYS
                )
                key = (uid, crop_id, input_sha, view_id)
                if key in seen:
                    raise ValueError(f"cache 重复行 key: {key}")
                seen.add(key)
                annotation_view = (uid, view_id)
                if annotation_view in seen_annotation_views:
                    raise ValueError(f"cache 重复 annotation/view: {annotation_view}")
                seen_annotation_views.add(annotation_view)
                previous = contract_by_annotation.setdefault(uid, (crop_id, input_sha))
                if previous != (crop_id, input_sha):
                    raise ValueError(f"cache 对象 {uid} 跨视图 crop/input 合同不一致")
                views_by_annotation.setdefault(uid, set()).add(view_id)
            for name in self.feature_names:
                feature = arrays[f"feature__{name}"]
                if feature.ndim != 2 or feature.shape[0] != n:
                    raise ValueError(f"feature {name} shape 非法")
                dimensions[name].add(int(feature.shape[1]))
                nonfinite += int((~np.isfinite(feature)).sum())
        if row_count != int(self.index["row_count"]):
            raise ValueError("cache audit 行数与 index 不一致")
        if nonfinite:
            raise ValueError(f"cache 含 {nonfinite} 个 NaN/Inf")
        if any(len(values) != 1 for values in dimensions.values()):
            raise ValueError(f"feature 维度跨 shard 不一致: {dimensions}")
        metadata = self.meta.get("metadata", {})
        expected_object_count = metadata.get("object_count")
        if expected_object_count is not None and len(views_by_annotation) != int(
            expected_object_count
        ):
            raise ValueError("cache 唯一对象数与 metadata.object_count 不一致")
        expected_views = set(metadata.get("view_ids", ()))
        if expected_views:
            invalid = [uid for uid, views in views_by_annotation.items() if views != expected_views]
            if invalid:
                raise ValueError(f"cache 有 {len(invalid)} 个对象视图集合不完整")
        view_counts: dict[str, int] = {}
        for _, view_id in seen_annotation_views:
            view_counts[view_id] = view_counts.get(view_id, 0) + 1
        return {
            "status": "pass",
            "schema_version": CACHE_SCHEMA_VERSION,
            "row_count": row_count,
            "unique_row_keys": len(seen),
            "unique_annotation_views": len(seen_annotation_views),
            "object_count": len(views_by_annotation),
            "view_counts": dict(sorted(view_counts.items())),
            "feature_dimensions": {name: next(iter(values)) for name, values in dimensions.items()},
            "nonfinite_count": nonfinite,
            "config_fingerprint": self.index["config_fingerprint"],
        }
