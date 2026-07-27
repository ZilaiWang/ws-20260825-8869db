"""P03/P04 正式 CV3 复验的输入、缓存与汇总门禁。

本模块不生成 ``formal_crop_manifest_v2``。正式 crop manifest 由公共数据
入口从 P0-2 对象集合和 CV3 v2 重挂 fold；本模块只验证它确实只改变了
划分字段，并为既有 P04 无标签缓存建立逐对象 canonical 像素证明。
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from rsdet.analysis.formal_crop import HISTORICAL_ASSIGNMENT_COLUMNS
from rsdet.evaluation.classification import evaluate_classification
from rsdet.features.p04_cache import FeatureCache
from rsdet.features.p04_inputs import (
    D4_VIEW_IDS,
    canonical_image_sha256,
    load_all_crop_records,
    render_canonical_crop,
)

FORMAL_CROP_VERSION = "formal_crop_manifest_v2"
CV3_VERSION = "cv3_airport_proxy_k60_v2"
CV3_SHA256 = "27b2eef4d757d91c7759d5fde64232718ea423385f0cf63ff823fa338b577331"
EXPECTED_OBJECTS = 20_933
EXPECTED_POLICIES = ("context_1p25", "jitter_light", "tight")
FULL_SUPPORT_TIERS = {
    # 固定使用 20,933 个正式对象的全数据 support 排序，避免 TU-160 压力折
    # 改变某一折的 tier 定义。9/8/8 与 P03 既有三等分规则一致。
    "tail": (0, 1, 21, 9, 24, 14, 18, 7, 10),
    "middle": (15, 2, 11, 23, 13, 12, 22, 6),
    "head": (8, 20, 16, 5, 4, 17, 3, 19),
}

# 这些字段由正式 CV3 重挂。除此之外，P0-2 对象、类别、几何和像素合同不得变。
MUTABLE_FORMAL_FIELDS = frozenset(
    {
        "manifest_version",
        "schema_version",
        "fold",
        "group_id",
        "leakage_group_id",
        "main_split",
        "split",
    }
)


def sha256_file(path: str | Path) -> str:
    """流式计算 SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: str | Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    target = Path(path).expanduser().resolve()
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    if not fields or not rows:
        raise ValueError(f"CSV 为空或无表头: {target}")
    return fields, rows


def _row_key(row: Mapping[str, str]) -> tuple[str, str]:
    return row["annotation_uid"].strip(), row["crop_policy"].strip()


def _index_unique(
    rows: Iterable[Mapping[str, str]],
    *,
    label: str,
) -> dict[tuple[str, str], Mapping[str, str]]:
    result: dict[tuple[str, str], Mapping[str, str]] = {}
    crop_ids: set[str] = set()
    for row in rows:
        key = _row_key(row)
        crop_id = row["crop_id"].strip()
        if not all(key) or not crop_id:
            raise ValueError(f"{label} 含空 annotation_uid/policy/crop_id")
        if key in result:
            raise ValueError(f"{label} 重复对象/policy: {key}")
        if crop_id in crop_ids:
            raise ValueError(f"{label} 重复 crop_id: {crop_id}")
        result[key] = row
        crop_ids.add(crop_id)
    return result


def _cv3_by_path(
    payload: Mapping[str, Any],
    *,
    expected_sample_count: int,
) -> dict[str, Mapping[str, Any]]:
    if payload.get("version") != CV3_VERSION:
        raise ValueError(f"CV3 version 非法: {payload.get('version')!r}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != expected_sample_count:
        raise ValueError(f"CV3 必须含 {expected_sample_count} 个 samples")
    result: dict[str, Mapping[str, Any]] = {}
    names: set[str] = set()
    for sample in samples:
        relative = str(sample["relative_path"]).replace("\\", "/")
        name = Path(relative).name
        if relative in result or name in names:
            raise ValueError(f"CV3 relative_path/文件名重复: {relative}")
        fold = int(sample["fold"])
        if fold not in (0, 1, 2) or not str(sample["group_id"]).strip():
            raise ValueError(f"CV3 sample fold/group 非法: {relative}")
        result[relative] = sample
        names.add(name)
    return result


def _match_cv3_sample(
    source_relative_path: str,
    by_path: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    normalized = source_relative_path.replace("\\", "/")
    if normalized in by_path:
        return by_path[normalized]
    matches = [sample for path, sample in by_path.items() if Path(path).name == Path(normalized).name]
    if len(matches) != 1:
        raise ValueError(f"source_relative_path 无法唯一匹配 CV3: {source_relative_path}")
    return matches[0]


def audit_formal_crop_manifest(
    formal_manifest: str | Path,
    exploratory_manifest: str | Path,
    cv3_manifest: str | Path,
    *,
    expected_cv3_sha256: str = CV3_SHA256,
    expected_object_count: int = EXPECTED_OBJECTS,
    expected_cv3_sample_count: int = 4_481,
) -> dict[str, Any]:
    """证明 formal manifest 只重挂了 CV3 fold/group，没有改变对象或 crop。"""

    cv3_path = Path(cv3_manifest).expanduser().resolve()
    cv3_sha = sha256_file(cv3_path)
    if cv3_sha != expected_cv3_sha256:
        raise ValueError(
            f"CV3 SHA256 不匹配: expected={expected_cv3_sha256}, actual={cv3_sha}"
        )
    cv3 = json.loads(cv3_path.read_text(encoding="utf-8"))
    by_path = _cv3_by_path(cv3, expected_sample_count=expected_cv3_sample_count)

    exploratory_fields, exploratory_rows = _read_csv(exploratory_manifest)
    formal_fields, formal_rows = _read_csv(formal_manifest)
    required_formal_fields = {
        f"historical_p02_{field}" if field in HISTORICAL_ASSIGNMENT_COLUMNS else field
        for field in exploratory_fields
    }
    missing = required_formal_fields - set(formal_fields)
    if missing:
        raise ValueError(f"formal manifest 丢失 P0-2 字段: {sorted(missing)}")
    exploratory = _index_unique(exploratory_rows, label="exploratory")
    formal = _index_unique(formal_rows, label="formal")
    if set(exploratory) != set(formal):
        raise ValueError(
            "formal/exploratory 对象集合不一致: "
            f"formal_only={len(set(formal) - set(exploratory))}, "
            f"exploratory_only={len(set(exploratory) - set(formal))}"
        )

    immutable_mismatches: Counter[str] = Counter()
    historical_mismatches: Counter[str] = Counter()
    versions: set[str] = set()
    fold_objects: Counter[int] = Counter()
    policy_counts: Counter[str] = Counter()
    source_contract: dict[str, tuple[int, str]] = {}
    group_folds: defaultdict[str, set[int]] = defaultdict(set)
    for key in sorted(formal):
        old = exploratory[key]
        new = formal[key]
        versions.add(new["manifest_version"].strip())
        for field in exploratory_fields:
            if field in HISTORICAL_ASSIGNMENT_COLUMNS:
                historical_field = f"historical_p02_{field}"
                if old.get(field, "") != new.get(historical_field, ""):
                    historical_mismatches[field] += 1
            elif field not in MUTABLE_FORMAL_FIELDS and old.get(field, "") != new.get(field, ""):
                immutable_mismatches[field] += 1
        sample = _match_cv3_sample(new["source_relative_path"], by_path)
        fold = int(new["fold"])
        group_id = str(sample["group_id"])
        if fold != int(sample["fold"]):
            raise ValueError(f"{key} fold 与 CV3 不一致")
        if new["leakage_group_id"].strip() != group_id:
            raise ValueError(f"{key} leakage_group_id 与 CV3 group_id 不一致")
        source_id = new["source_image_id"].strip()
        previous = source_contract.setdefault(source_id, (fold, group_id))
        if previous != (fold, group_id):
            raise ValueError(f"同一 source_image_id 的 fold/group 不一致: {source_id}")
        group_folds[group_id].add(fold)
        policy = new["crop_policy"].strip()
        policy_counts[policy] += 1
        if policy == "tight":
            fold_objects[fold] += 1

    if immutable_mismatches:
        raise ValueError(f"formal manifest 改变不可变字段: {dict(immutable_mismatches)}")
    if historical_mismatches:
        raise ValueError(
            "formal manifest 的 historical_p02_* 与来源不一致: "
            f"{dict(historical_mismatches)}"
        )
    if versions != {FORMAL_CROP_VERSION}:
        raise ValueError(f"formal manifest_version 非法: {sorted(versions)}")
    expected_rows = expected_object_count * len(EXPECTED_POLICIES)
    if len(formal_rows) != expected_rows:
        raise ValueError(f"formal manifest 行数错误: {len(formal_rows)} != {expected_rows}")
    expected_policy_counts = {policy: expected_object_count for policy in EXPECTED_POLICIES}
    if dict(sorted(policy_counts.items())) != expected_policy_counts:
        raise ValueError(f"policy 对象数错误: {dict(policy_counts)}")
    crossing = sorted(group for group, folds in group_folds.items() if len(folds) != 1)
    if crossing:
        raise ValueError(f"formal manifest 有跨折 group: {crossing[:5]}")
    if set(fold_objects) != {0, 1, 2} or sum(fold_objects.values()) != expected_object_count:
        raise ValueError(f"tight fold 对象数非法: {dict(fold_objects)}")

    return {
        "status": "formal_crop_manifest_gate_pass",
        "formal_manifest": str(Path(formal_manifest).expanduser().resolve()),
        "formal_manifest_sha256": sha256_file(formal_manifest),
        "exploratory_manifest": str(Path(exploratory_manifest).expanduser().resolve()),
        "exploratory_manifest_sha256": sha256_file(exploratory_manifest),
        "cv3_manifest": str(cv3_path),
        "cv3_manifest_sha256": cv3_sha,
        "cv3_version": CV3_VERSION,
        "formal_manifest_version": FORMAL_CROP_VERSION,
        "row_count": len(formal_rows),
        "object_count": expected_object_count,
        "policy_counts": dict(sorted(policy_counts.items())),
        "tight_fold_object_counts": [fold_objects[fold] for fold in (0, 1, 2)],
        "source_image_count": len(source_contract),
        "source_group_count": len(group_folds),
        "cross_fold_group_count": 0,
        "immutable_mismatch_count": 0,
        "historical_assignment_mismatch_count": 0,
    }


def audit_cache_reuse(
    formal_manifest: str | Path,
    data_root: str | Path,
    cache_dirs: Mapping[str, str | Path],
    *,
    expected_object_count: int = EXPECTED_OBJECTS,
    expected_fingerprint_prefixes: Mapping[str, str] | None = None,
    expected_teacher_ids: Mapping[str, str] | None = None,
    asset_lock_path: str | Path | None = None,
) -> dict[str, Any]:
    """逐对象重渲染 canonical224，并与每个旧 cache 的输入 SHA 对齐。"""

    records = load_all_crop_records(formal_manifest, crop_policy="tight")
    if len(records) != expected_object_count:
        raise ValueError(f"tight 对象数错误: {len(records)} != {expected_object_count}")
    canonical_by_uid: dict[str, str] = {}
    crop_by_uid: dict[str, str] = {}
    for record in records:
        canonical = render_canonical_crop(record, data_root, resolution=224)
        canonical_by_uid[record.annotation_uid] = canonical_image_sha256(canonical)
        crop_by_uid[record.annotation_uid] = record.crop_id

    expected_fingerprint_prefixes = dict(expected_fingerprint_prefixes or {})
    expected_teacher_ids = dict(expected_teacher_ids or {})
    if set(expected_fingerprint_prefixes) != set(cache_dirs):
        raise ValueError("每个正式 cache 都必须有唯一预注册 fingerprint prefix")
    if set(expected_teacher_ids) != set(cache_dirs):
        raise ValueError("每个正式 cache 都必须有唯一预注册 teacher_id")
    if asset_lock_path is None:
        raise ValueError("正式 cache 复用必须绑定已验证的 P04 ASSET_LOCK.json")
    asset_lock = Path(asset_lock_path).expanduser().resolve()
    if not asset_lock.is_file():
        raise FileNotFoundError(f"P04 asset lock 不存在: {asset_lock}")
    asset_lock_sha = sha256_file(asset_lock)

    results: dict[str, Any] = {}
    for name, cache_dir in sorted(cache_dirs.items()):
        cache = FeatureCache(cache_dir)
        cache_audit = cache.audit()
        fingerprint = str(cache.index["config_fingerprint"])
        prefix = expected_fingerprint_prefixes[name].strip().lower()
        if len(prefix) < 8 or not fingerprint.startswith(prefix):
            raise ValueError(
                f"cache {name} fingerprint 不属于预注册缓存: "
                f"expected_prefix={prefix}, actual={fingerprint}"
            )
        metadata = cache.meta.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"cache {name} metadata 非法")
        if metadata.get("asset_lock_sha256") != asset_lock_sha:
            raise ValueError(f"cache {name} 未绑定当前已验证 P04 asset lock")
        teacher = metadata.get("teacher")
        if (
            not isinstance(teacher, Mapping)
            or teacher.get("teacher_id") != expected_teacher_ids[name]
        ):
            raise ValueError(f"cache {name} teacher_id 与预注册合同不一致")
        payload = cache.load_feature(cache.feature_names[0])
        uids = np.asarray(payload["annotation_uid"]).astype(str)
        crops = np.asarray(payload["crop_id"]).astype(str)
        input_hashes = np.asarray(payload["canonical_input_sha256"]).astype(str)
        views = np.asarray(payload["view_id"]).astype(str)
        seen: dict[str, tuple[str, str]] = {}
        view_sets: defaultdict[str, set[str]] = defaultdict(set)
        unknown = 0
        mismatch = 0
        for uid, crop_id, input_hash, view_id in zip(
            uids, crops, input_hashes, views, strict=True
        ):
            if uid not in canonical_by_uid:
                unknown += 1
                continue
            contract = (crop_id, input_hash)
            previous = seen.setdefault(uid, contract)
            if previous != contract:
                mismatch += 1
            if crop_id != crop_by_uid[uid] or input_hash != canonical_by_uid[uid]:
                mismatch += 1
            view_sets[uid].add(view_id)
        missing = set(canonical_by_uid) - set(seen)
        incomplete_views = sum(set(values) != set(D4_VIEW_IDS) for values in view_sets.values())
        if unknown or mismatch or missing or incomplete_views:
            raise ValueError(
                f"cache {name} 不可复用: unknown={unknown}, mismatch={mismatch}, "
                f"missing={len(missing)}, incomplete_views={incomplete_views}"
            )
        results[name] = {
            "status": "pass",
            "cache_dir": str(Path(cache_dir).expanduser().resolve()),
            "cache_fingerprint": fingerprint,
            "expected_fingerprint_prefix": prefix,
            "cache_meta_sha256": sha256_file(cache.meta_path),
            "cache_index_sha256": sha256_file(cache.index_path),
            "asset_lock": str(asset_lock),
            "asset_lock_sha256": asset_lock_sha,
            "teacher": dict(teacher),
            "feature_names": list(cache.feature_names),
            "row_count": int(cache.index["row_count"]),
            "object_count": len(seen),
            "canonical_mismatch_count": 0,
            "crop_id_mismatch_count": 0,
            "missing_annotation_count": 0,
            "unknown_annotation_count": 0,
            "incomplete_view_object_count": 0,
            "cache_audit": cache_audit,
        }
    return results


def validate_cache_reuse_audit(
    audit_path: str | Path,
    *,
    formal_manifest_sha256: str,
    cache_fingerprint: str,
) -> dict[str, Any]:
    """验证一次 probe 使用的 cache 已进入正式复用白名单。"""

    path = Path(audit_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "formal_replay_inputs_ready":
        raise ValueError("formal input audit 尚未通过")
    if payload.get("formal_manifest_sha256") != formal_manifest_sha256:
        raise ValueError("reuse audit 的 formal manifest SHA 不匹配")
    matches = [
        item
        for item in payload.get("caches", {}).values()
        if item.get("cache_fingerprint") == cache_fingerprint
    ]
    if len(matches) != 1:
        raise ValueError("reuse audit 未唯一准入当前 cache fingerprint")
    admitted = matches[0]
    required_zero = (
        "canonical_mismatch_count",
        "crop_id_mismatch_count",
        "missing_annotation_count",
        "unknown_annotation_count",
        "incomplete_view_object_count",
    )
    if admitted.get("status") != "pass" or any(int(admitted.get(key, -1)) for key in required_zero):
        raise ValueError("reuse audit 的 UID/crop/canonical 门禁未通过")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "cache_fingerprint": cache_fingerprint,
    }


def fit_train_rms(train_features: np.ndarray, eps: float = 1e-12) -> float:
    """只使用训练 fold（含其 D4 视图）拟合一个全局 RMS 标量。"""

    values = np.asarray(train_features, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("train RMS 输入为空或含 NaN/Inf")
    rms = float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))
    if not np.isfinite(rms) or rms <= eps:
        raise ValueError(f"train RMS 非法: {rms}")
    return rms


def aggregate_metric_rows(
    groups: Mapping[tuple[Any, ...], Sequence[Mapping[str, Any]]],
    *,
    metric_names: Sequence[str],
) -> list[dict[str, Any]]:
    """按完整三折汇总 run summary；供正式汇总脚本和测试复用。"""

    rows: list[dict[str, Any]] = []
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        ordered = sorted(values, key=lambda item: int(item["fold"]))
        folds = [int(item["fold"]) for item in ordered]
        if folds != [0, 1, 2]:
            raise ValueError(f"条件 {key} folds 不完整或重复: {folds}")
        row: dict[str, Any] = {"condition": list(key), "folds": folds}
        for metric in metric_names:
            scores = [float(item["metrics"][metric]) for item in ordered]
            row[f"{metric}_fold_values"] = scores
            row[f"{metric}_mean"] = mean(scores)
            row[f"{metric}_std"] = stdev(scores)
        rows.append(row)
    return rows


def pooled_classification_metrics(
    logits: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
) -> dict[str, Any]:
    """把三个互斥验证折合为一次 OOF 对象级指标。"""

    if len(logits) != 3 or len(labels) != 3:
        raise ValueError("pooled OOF 必须恰含三个 fold")
    scores = np.concatenate([np.asarray(value, dtype=np.float64) for value in logits])
    truth = np.concatenate([np.asarray(value, dtype=np.int64) for value in labels])
    metrics, _ = evaluate_classification(truth, scores)
    per_class = metrics["per_class"]

    def tier_metrics(class_ids: Sequence[int]) -> dict[str, Any]:
        rows = [per_class[class_id] for class_id in class_ids]
        return {
            "class_ids": list(class_ids),
            "support": sum(int(row["support"]) for row in rows),
            "macro_recall": float(
                np.mean([float(row["recall"]) for row in rows if row["recall"] is not None])
            ),
            "macro_f1": float(
                np.mean([float(row["f1"]) for row in rows if row["f1"] is not None])
            ),
        }

    return {
        "n_samples": int(len(truth)),
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "aircraft20_macro_recall": metrics["subgroups"]["aircraft20"]["macro_recall"],
        "major_classes": metrics["subgroups"],
        "support_tiers": {
            name: tier_metrics(class_ids) for name, class_ids in FULL_SUPPORT_TIERS.items()
        },
        "per_class": per_class,
    }


def class_recall_support(
    logits: np.ndarray,
    labels: np.ndarray,
    class_id: int,
) -> dict[str, Any]:
    """返回一个 fold 内指定类的 recall/support，零 support 显式为 null。"""

    scores = np.asarray(logits)
    truth = np.asarray(labels, dtype=np.int64)
    mask = truth == class_id
    support = int(mask.sum())
    correct = int((scores[mask].argmax(axis=1) == class_id).sum()) if support else 0
    return {
        "class_id": class_id,
        "support": support,
        "true_positive": correct,
        "recall": correct / support if support else None,
    }
