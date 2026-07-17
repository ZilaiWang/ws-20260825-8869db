"""P04 raw DIFT ensemble 和 CleanDIFT D4/重复提取稳定性。"""

from __future__ import annotations

from typing import Any

import numpy as np

from rsdet.features.p04_cache import FeatureCache


def row_cosine(first: np.ndarray, second: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("cosine 输入必须是同形 [N,D]")
    numerator = np.sum(a * b, axis=1)
    denominator = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), eps)
    return numerator / denominator


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("稳定性数组为空或含 NaN/Inf")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _keyed_feature(
    cache: FeatureCache,
    feature_name: str,
    *,
    include_input_contract: bool = False,
) -> dict[tuple[str, ...], np.ndarray]:
    payload = cache.load_feature(feature_name)
    uids = np.asarray(payload["annotation_uid"]).astype(str)
    views = np.asarray(payload["view_id"]).astype(str)
    crops = np.asarray(payload["crop_id"]).astype(str)
    inputs = np.asarray(payload["canonical_input_sha256"]).astype(str)
    features = np.asarray(payload[feature_name], dtype=np.float32)
    result: dict[tuple[str, ...], np.ndarray] = {}
    for uid, view, crop, input_sha, feature in zip(
        uids, views, crops, inputs, features
    ):
        key = (uid, crop, input_sha, view) if include_input_contract else (uid, view)
        if key in result:
            raise ValueError(f"cache 重复 key: {key}")
        result[key] = feature
    return result


def analyze_raw_ensemble(cache_dir: str) -> dict[str, Any]:
    cache = FeatureCache(cache_dir)
    required = {
        "raw_map0_t100_e1",
        "raw_map0_t100_e4",
        "raw_map0_t100_e8",
        "raw_map6_t261_e1",
        "raw_map6_t261_e4",
        "raw_map6_t261_e8",
    }
    missing = required - set(cache.feature_names)
    if missing:
        raise ValueError(f"raw cache 缺少特征: {sorted(missing)}")
    comparisons: dict[str, Any] = {}
    for location in ("map0_t100", "map6_t261"):
        reference = cache.load_feature(f"raw_{location}_e8")
        reference_keys = list(
            zip(
                np.asarray(reference["annotation_uid"]).astype(str),
                np.asarray(reference["view_id"]).astype(str),
            )
        )
        if set(view for _, view in reference_keys) != {"r0"}:
            raise ValueError("raw DIFT calibration 应只含 identity 视图")
        reference_feature = np.asarray(reference[f"raw_{location}_e8"], dtype=np.float32)
        for size in (1, 4):
            payload = cache.load_feature(f"raw_{location}_e{size}")
            keys = list(
                zip(
                    np.asarray(payload["annotation_uid"]).astype(str),
                    np.asarray(payload["view_id"]).astype(str),
                )
            )
            if keys != reference_keys:
                raise ValueError("raw ensemble cache 行顺序不一致")
            cosine = row_cosine(
                np.asarray(payload[f"raw_{location}_e{size}"], dtype=np.float32),
                reference_feature,
            )
            comparisons[f"{location}_e{size}_vs_e8"] = distribution_summary(cosine)
    e4_values = [
        comparisons[f"{location}_e4_vs_e8"]
        for location in ("map0_t100", "map6_t261")
    ]
    passed = all(value["median"] >= 0.99 and value["p05"] >= 0.97 for value in e4_values)
    return {
        "cache_fingerprint": cache.index["config_fingerprint"],
        "row_count": cache.index["row_count"],
        "comparisons": comparisons,
        "ensemble4_gate": {
            "status": "pass" if passed else "fail",
            "median_threshold": 0.99,
            "p05_threshold": 0.97,
        },
    }


def analyze_clean_d4(cache_dir: str) -> dict[str, Any]:
    cache = FeatureCache(cache_dir)
    required = {"clean_map0", "clean_map6", "clean_map9"}
    missing = required - set(cache.feature_names)
    if missing:
        raise ValueError(f"CleanDIFT cache 缺少特征: {sorted(missing)}")
    output: dict[str, Any] = {}
    for feature_name in sorted(required):
        keyed = _keyed_feature(cache, feature_name)
        uids = sorted({uid for uid, _ in keyed})
        views_by_uid: dict[str, list[str]] = {uid: [] for uid in uids}
        for uid, view in keyed:
            views_by_uid[uid].append(view)
        if any((uid, "r0") not in keyed for uid in uids):
            raise ValueError("CleanDIFT D4 cache 缺少 r0")
        cosines: list[float] = []
        by_view: dict[str, list[float]] = {}
        for uid in uids:
            reference = keyed[(uid, "r0")][None, :]
            for key in sorted(view for view in views_by_uid[uid] if view != "r0"):
                cosine = float(row_cosine(keyed[(uid, key)][None, :], reference)[0])
                cosines.append(cosine)
                by_view.setdefault(key, []).append(cosine)
        output[feature_name] = {
            "all_non_identity_vs_r0": distribution_summary(np.asarray(cosines)),
            "by_view": {
                view: distribution_summary(np.asarray(values)) for view, values in by_view.items()
            },
        }
    return {
        "cache_fingerprint": cache.index["config_fingerprint"],
        "row_count": cache.index["row_count"],
        "features": output,
        "note": "D4 cosine 是诊断量，不设硬入选阈值",
    }


def compare_repeat_caches(first_dir: str, second_dir: str) -> dict[str, Any]:
    return _compare_caches(first_dir, second_dir, require_same_keys=True)


def compare_cache_overlap(first_dir: str, second_dir: str) -> dict[str, Any]:
    """比较两个 cache 的公共对象/视图，用于 calibration→full 一致性门禁。"""

    return _compare_caches(first_dir, second_dir, require_same_keys=False)


def _compare_caches(
    first_dir: str, second_dir: str, *, require_same_keys: bool
) -> dict[str, Any]:
    first = FeatureCache(first_dir)
    second = FeatureCache(second_dir)
    common = sorted(set(first.feature_names) & set(second.feature_names))
    if not common:
        raise ValueError("两个 repeat cache 无共同特征")
    values: dict[str, Any] = {}
    passed = True
    for name in common:
        # calibration→full 只能在 crop ID 和 canonical 像素 SHA 也完全
        # 相同时比较；仅 annotation/view 相同不足以证明缓存可复用。
        a = _keyed_feature(first, name, include_input_contract=True)
        b = _keyed_feature(second, name, include_input_contract=True)
        if require_same_keys and set(a) != set(b):
            raise ValueError(f"repeat cache {name} 行 key 不一致")
        keys = sorted(set(a) & set(b))
        if not keys:
            raise ValueError(f"cache {name} 无公共行 key")
        cosine = row_cosine(
            np.stack([a[key] for key in keys]), np.stack([b[key] for key in keys])
        )
        absolute = np.abs(
            np.stack([a[key] for key in keys]) - np.stack([b[key] for key in keys])
        )
        summary = distribution_summary(cosine)
        summary["max_absolute_difference"] = float(absolute.max())
        summary["common_row_count"] = len(keys)
        values[name] = summary
        passed = passed and summary["p05"] >= 0.999
    return {
        "status": "pass" if passed else "fail",
        "p05_cosine_threshold": 0.999,
        "require_same_keys": require_same_keys,
        "features": values,
    }
