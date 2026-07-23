"""MAR20 地点描述子的分片缓存、resume 与完整性审计。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rsdet.grouping.contracts import (
    FEATURE_CACHE_SCHEMA_VERSION,
    atomic_write_json,
    sha256_file,
    stable_json_sha256,
)

REQUIRED_ROW_KEYS = (
    "node_uid",
    "view_type",
    "rotation",
    "item_index",
    "input_sha256",
)


class PlaceFeatureCacheWriter:
    """固定输入指纹下写出可断点续跑的 NumPy shards。"""

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
            raise ValueError("storage_dtype 只能为 float16/float32")
        names = tuple(feature_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("feature_names 必须非空且唯一")
        self.metadata = dict(metadata)
        self.feature_names = names
        self.storage_dtype = storage_dtype
        self.fingerprint = stable_json_sha256(
            {
                "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
                "metadata": self.metadata,
                "feature_names": list(names),
                "storage_dtype": storage_dtype,
            }
        )
        meta_path = self.output_dir / "cache_meta.json"
        if meta_path.exists():
            current = json.loads(meta_path.read_text(encoding="utf-8"))
            if current.get("fingerprint") != self.fingerprint:
                raise ValueError("现有 place cache 与本次合同不一致，禁止混写")
        else:
            atomic_write_json(
                meta_path,
                {
                    "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
                    "fingerprint": self.fingerprint,
                    "metadata": self.metadata,
                    "feature_names": list(names),
                    "storage_dtype": storage_dtype,
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
            sidecar.get("fingerprint") == self.fingerprint
            and int(sidecar.get("row_count", -1)) == expected_rows
            and sidecar.get("sha256") == sha256_file(data_path)
        )

    def write_shard(
        self,
        shard_index: int,
        *,
        rows: Mapping[str, Sequence[Any] | np.ndarray],
        features: Mapping[str, np.ndarray],
    ) -> dict[str, Any]:
        missing = set(REQUIRED_ROW_KEYS) - set(rows)
        if missing:
            raise ValueError(f"place cache rows 缺少: {sorted(missing)}")
        if set(features) != set(self.feature_names):
            raise ValueError("place cache feature keys 与合同不一致")
        row_count = len(rows["node_uid"])
        if row_count <= 0:
            raise ValueError("禁止写空 shard")
        payload: dict[str, np.ndarray] = {}
        for key, values in rows.items():
            array = np.asarray(values)
            if array.ndim != 1 or array.shape[0] != row_count:
                raise ValueError(f"row {key} 必须为长度 {row_count} 的一维数组")
            payload[f"row__{key}"] = array
        dtype = np.float16 if self.storage_dtype == "float16" else np.float32
        shapes: dict[str, list[int]] = {}
        for name in self.feature_names:
            array = np.asarray(features[name])
            if array.ndim != 2 or array.shape[0] != row_count:
                raise ValueError(f"feature {name} 必须为 [N,D]")
            if not np.isfinite(array).all():
                raise ValueError(f"feature {name} 含 NaN/Inf")
            stored = array.astype(dtype, copy=False)
            payload[f"feature__{name}"] = stored
            shapes[name] = [int(value) for value in stored.shape]
        data_path, sidecar_path = self._paths(shard_index)
        temporary = data_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as file:
            np.savez(file, **payload)
        os.replace(temporary, data_path)
        sidecar = {
            "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "shard_index": shard_index,
            "path": data_path.name,
            "row_count": row_count,
            "sha256": sha256_file(data_path),
            "feature_shapes": shapes,
        }
        atomic_write_json(sidecar_path, sidecar)
        return sidecar

    def finalize(self, *, expected_shards: int, expected_rows: int) -> dict[str, Any]:
        shards = []
        row_count = 0
        for shard_index in range(expected_shards):
            data_path, sidecar_path = self._paths(shard_index)
            if not data_path.is_file() or not sidecar_path.is_file():
                raise FileNotFoundError(f"缺少 place cache shard {shard_index}")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if sidecar.get("fingerprint") != self.fingerprint:
                raise ValueError(f"place cache shard {shard_index} fingerprint 不一致")
            if sidecar.get("sha256") != sha256_file(data_path):
                raise ValueError(f"place cache shard {shard_index} SHA 不一致")
            row_count += int(sidecar["row_count"])
            shards.append(sidecar)
        if row_count != expected_rows:
            raise ValueError(f"place cache 行数 expected={expected_rows}, actual={row_count}")
        index = {
            "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "row_count": row_count,
            "shard_count": len(shards),
            "feature_names": list(self.feature_names),
            "storage_dtype": self.storage_dtype,
            "shards": shards,
        }
        atomic_write_json(self.output_dir / "index.json", index)
        return index


class PlaceFeatureCache:
    def __init__(self, cache_dir: str | Path, *, verify_sha256: bool = True) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.meta = json.loads((self.cache_dir / "cache_meta.json").read_text(encoding="utf-8"))
        self.index = json.loads((self.cache_dir / "index.json").read_text(encoding="utf-8"))
        if self.meta.get("schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
            raise ValueError("place cache meta schema 不支持")
        if self.index.get("schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
            raise ValueError("place cache index schema 不支持")
        if self.meta.get("fingerprint") != self.index.get("fingerprint"):
            raise ValueError("place cache meta/index fingerprint 不一致")
        if verify_sha256:
            for shard in self.index["shards"]:
                path = self.cache_dir / shard["path"]
                if not path.is_file() or sha256_file(path) != shard["sha256"]:
                    raise ValueError(f"place cache shard 缺失或改变: {path}")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.index["feature_names"])

    def load_all(self) -> dict[str, np.ndarray]:
        collected: dict[str, list[np.ndarray]] = {}
        for shard in self.index["shards"]:
            with np.load(self.cache_dir / shard["path"], allow_pickle=False) as payload:
                for key in payload.files:
                    collected.setdefault(key, []).append(np.asarray(payload[key]))
        result = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
        expected = int(self.index["row_count"])
        if any(value.shape[0] != expected for value in result.values()):
            raise ValueError("place cache 拼接后行数不一致")
        return result

    def audit(self) -> dict[str, Any]:
        payload = self.load_all()
        keys = list(
            zip(
                payload["row__node_uid"].astype(str),
                payload["row__view_type"].astype(str),
                payload["row__rotation"].astype(int),
                payload["row__item_index"].astype(int),
                strict=True,
            )
        )
        if len(keys) != len(set(keys)):
            raise ValueError("place cache 行 key 重复")
        dimensions = {}
        nonfinite = 0
        for name in self.feature_names:
            array = payload[f"feature__{name}"]
            dimensions[name] = int(array.shape[1])
            nonfinite += int((~np.isfinite(array)).sum())
        if nonfinite:
            raise ValueError("place cache 含 NaN/Inf")
        return {
            "status": "pass",
            "row_count": len(keys),
            "unique_row_count": len(set(keys)),
            "feature_dimensions": dimensions,
            "nonfinite_count": nonfinite,
            "fingerprint": self.index["fingerprint"],
        }
