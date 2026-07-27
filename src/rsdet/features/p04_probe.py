"""P04 缓存特征与 formal manifest 的对齐。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from rsdet.features.p04_cache import FeatureCache, sha256_file
from rsdet.features.p04_inputs import D4_VIEW_IDS, load_all_crop_records


@dataclass(frozen=True)
class ProbeData:
    annotation_uids: tuple[str, ...]
    crop_ids: tuple[str, ...]
    source_image_ids: tuple[str, ...]
    leakage_group_ids: tuple[str, ...]
    class_ids: np.ndarray
    folds: np.ndarray
    view_ids: tuple[str, ...]
    features: np.ndarray
    cache_fingerprint: str
    manifest_sha256: str
    ignored_cache_annotations: int


def l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norms, eps)


def deterministic_view_index(
    annotation_uid: str, epoch: int, seed: int, view_count: int
) -> int:
    if view_count <= 0:
        raise ValueError("view_count 必须大于 0")
    digest = hashlib.sha256(f"{seed}|{epoch}|{annotation_uid}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % view_count


def load_probe_data(
    cache_dir: str | Path,
    *,
    feature_name: str,
    manifest_path: str | Path,
    crop_policy: str = "tight",
    allow_cache_subset: bool = False,
    reuse_audit_path: str | Path | None = None,
) -> ProbeData:
    """以 manifest 为唯一标签/fold 真值，对齐 fold-independent cache。"""

    cache = FeatureCache(cache_dir)
    manifest_sha256 = sha256_file(manifest_path)
    extraction_manifest_sha256 = cache.meta.get("metadata", {}).get("manifest_sha256")
    if extraction_manifest_sha256 is not None and extraction_manifest_sha256 != manifest_sha256:
        if reuse_audit_path is None:
            raise ValueError(
                "probe manifest 与 cache 提取 manifest 不同；必须先执行 P04-TASK-05 正式复验 "
                "UID/crop/canonical input SHA 对齐，当前入口禁止未经证明的跨 manifest 复用"
            )
        # 延迟导入，普通 exploratory probe 不需要加载正式复验门禁。
        from rsdet.analysis.formal_replay import validate_cache_reuse_audit

        validate_cache_reuse_audit(
            reuse_audit_path,
            formal_manifest_sha256=manifest_sha256,
            cache_fingerprint=cache.index["config_fingerprint"],
        )
    payload = cache.load_feature(feature_name)
    cache_uids = np.asarray(payload["annotation_uid"]).astype(str)
    cache_views = np.asarray(payload["view_id"]).astype(str)
    cache_crops = np.asarray(payload["crop_id"]).astype(str)
    feature = np.asarray(payload[feature_name], dtype=np.float32)
    if feature.ndim != 2:
        raise ValueError("cache feature 不是 [N,D]")

    records = load_all_crop_records(manifest_path, crop_policy=crop_policy)
    record_map = {item.annotation_uid: item for item in records}
    present_uids = set(cache_uids)
    missing = set(record_map) - present_uids
    if missing and not allow_cache_subset:
        raise ValueError(f"formal manifest 有 {len(missing)} 个对象缺少特征")
    if allow_cache_subset:
        unknown = present_uids - set(record_map)
        if unknown:
            raise ValueError(f"cache 有 {len(unknown)} 个对象不在 manifest 中")
        records = [item for item in records if item.annotation_uid in present_uids]
        record_map = {item.annotation_uid: item for item in records}
    view_values = tuple(view for view in D4_VIEW_IDS if view in set(cache_views))
    unexpected_views = set(cache_views) - set(D4_VIEW_IDS)
    if unexpected_views:
        raise ValueError(f"cache 含未知视图: {sorted(unexpected_views)}")
    if not view_values or view_values[0] != "r0":
        raise ValueError("cache 必须包含 identity r0")

    row_lookup: dict[tuple[str, str], int] = {}
    crop_lookup: dict[str, str] = {}
    for index, (uid, view_id, crop_id) in enumerate(zip(cache_uids, cache_views, cache_crops)):
        key = (uid, view_id)
        if key in row_lookup:
            raise ValueError(f"cache 重复 annotation/view: {key}")
        row_lookup[key] = index
        previous = crop_lookup.setdefault(uid, crop_id)
        if previous != crop_id:
            raise ValueError(f"{uid} 跨视图 crop_id 不一致")

    ordered = sorted(records, key=lambda item: (item.annotation_uid, item.crop_id))
    matrix = np.empty((len(ordered), len(view_values), feature.shape[1]), dtype=np.float32)
    for object_index, record in enumerate(ordered):
        if crop_lookup[record.annotation_uid] != record.crop_id:
            raise ValueError(
                f"{record.annotation_uid} cache crop_id 与 formal manifest 不一致，禁止复用"
            )
        for view_index, view_id in enumerate(view_values):
            key = (record.annotation_uid, view_id)
            if key not in row_lookup:
                raise ValueError(f"cache 缺少对象/视图: {key}")
            matrix[object_index, view_index] = feature[row_lookup[key]]
    ignored = len(set(cache_uids) - set(record_map))
    return ProbeData(
        annotation_uids=tuple(item.annotation_uid for item in ordered),
        crop_ids=tuple(item.crop_id for item in ordered),
        source_image_ids=tuple(item.source_image_id for item in ordered),
        leakage_group_ids=tuple(item.leakage_group_id for item in ordered),
        class_ids=np.asarray([item.class_id for item in ordered], dtype=np.int64),
        folds=np.asarray([item.fold for item in ordered], dtype=np.int64),
        view_ids=view_values,
        features=matrix,
        cache_fingerprint=cache.index["config_fingerprint"],
        manifest_sha256=manifest_sha256,
        ignored_cache_annotations=ignored,
    )


def fit_pca_train_only(
    train_features: np.ndarray,
    *,
    n_components: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """仅用 train fold 拟合 randomized PCA，明确关闭 whitening。"""

    try:
        from sklearn.decomposition import PCA
    except ImportError as error:
        raise RuntimeError("PCA384 需要 scikit-learn") from error
    flat = np.asarray(train_features, dtype=np.float32).reshape(-1, train_features.shape[-1])
    maximum = min(flat.shape[0], flat.shape[1])
    if not 1 <= n_components <= maximum:
        raise ValueError(f"PCA n_components={n_components} 超出上限 {maximum}")
    model = PCA(
        n_components=n_components,
        whiten=False,
        svd_solver="randomized",
        random_state=seed,
    )
    model.fit(flat)
    return {
        "components": np.asarray(model.components_, dtype=np.float32),
        "mean": np.asarray(model.mean_, dtype=np.float32),
        "explained_variance_ratio": np.asarray(
            model.explained_variance_ratio_, dtype=np.float32
        ),
    }


def transform_pca(values: np.ndarray, pca: dict[str, np.ndarray]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    shape = array.shape
    flat = array.reshape(-1, shape[-1])
    transformed = (flat - pca["mean"]) @ pca["components"].T
    return transformed.reshape(*shape[:-1], pca["components"].shape[0])


def training_class_counts(labels: Sequence[int] | np.ndarray) -> dict[int, int]:
    values, counts = np.unique(np.asarray(labels, dtype=np.int64), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}
