"""N0-4：FP_BG 人工语义审计抽检包。

目的
----
M1 工作点下有 1,826 个 FP_BG 候选，但它们**不代表已确认是纯背景**——
可能混着漏标目标（``plausible_unlabeled``）、定位差的真目标（``poor_loc``）、
重复/碎片（``duplicate``）、无效裁剪（``invalid_crop``）。不能全当背景训练
样本（会把漏标目标当负样本，污染对象分类器）。

本模块：
1. 从 N0-3 的 hard_negative 视图取 FP_BG 候选；
2. 按 三大类 × 三折 × 分数分位（低/中/高）× 主要预测细类 分层抽检；
3. 每批混入一定比例**盲重复卡**（同一候选重复出现，测人工一致性）；
4. 生成人工标注表（CSV）+ 抽检清单（JSON），供 B（蔡婕）盲审；
5. 只有人工标为 ``clear_background`` 且满足严格几何条件的样本才允许
   进入背景训练集（下游模块据此过滤）。

约束
----
- 分层抽样使用**确定性随机种子**，保证可复现；
- 每批次内重复卡与正卡保持固定比例（默认 20%）；
- 抽取时**不做去重**（重复卡就是故意的），保证一致性统计有效。
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

# 人工标签协议（与总纲 N0-4 一致）。
LABEL_CLEAR_BACKGROUND = "clear_background"
LABEL_PLAUSIBLE_UNLABELED = "plausible_unlabeled_or_ambiguous_target"
LABEL_POOR_LOCALIZATION = "poor_localization_of_known_target"
LABEL_DUPLICATE = "duplicate_or_fragment_not_captured"
LABEL_INVALID_CROP = "invalid_crop_or_render"

HUMAN_LABELS: tuple[str, ...] = (
    LABEL_CLEAR_BACKGROUND,
    LABEL_PLAUSIBLE_UNLABELED,
    LABEL_POOR_LOCALIZATION,
    LABEL_DUPLICATE,
    LABEL_INVALID_CROP,
)

# 背景训练集准入：必须人工标为 clear_background（且由下游进一步筛选）。
ADMISSIBLE_BACKGROUND_LABELS: frozenset[str] = frozenset(
    {LABEL_CLEAR_BACKGROUND}
)


@dataclass(frozen=True)
class AuditSample:
    """一条抽检样本（含盲重复卡标记）。"""

    audit_uid: str
    proposal_uid: str
    image_id: int
    fold: int
    category_id: int
    class_name: str
    score: float
    bbox_xyxy: list[float]
    size_bin: str
    source_group: str
    score_bin: str
    is_repeat_control: bool
    repeat_of: str | None = None
    label: str | None = None
    labeler: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "audit_uid": self.audit_uid,
            "proposal_uid": self.proposal_uid,
            "image_id": self.image_id,
            "fold": self.fold,
            "category_id": self.category_id,
            "class_name": self.class_name,
            "score": self.score,
            "bbox_xyxy": self.bbox_xyxy,
            "size_bin": self.size_bin,
            "source_group": self.source_group,
            "score_bin": self.score_bin,
            "is_repeat_control": self.is_repeat_control,
            "repeat_of": self.repeat_of,
            "label": self.label,
            "labeler": self.labeler,
        }


def score_bin_of(score: float, *, low: float = 0.10, high: float = 0.35) -> str:
    """分数分位：低 < low、中 [low, high)、高 >= high。"""
    if score < low:
        return "low"
    if score < high:
        return "mid"
    return "high"


def build_fp_bg_sample_pool(
    manifest: Mapping[str, Any],
    *,
    include_fp_types: frozenset[str] = frozenset({"FP_BG"}),
) -> list[dict[str, Any]]:
    """从 N0-3 manifest 的 records 中取出指定 FP 类型的候选作为抽检池。"""
    records = manifest.get("records", [])
    pool: list[dict[str, Any]] = []
    for record in records:
        status = record.get("official_status")
        if status in include_fp_types:
            pool.append(record)
    return pool


def _group_key(record: dict[str, Any]) -> tuple[str, int, str]:
    """分层键：大类（由细类映射）+ fold + 分数分位。"""
    category_id = int(record["category_id"])
    return (str(record.get("class_name", "")), int(record["fold"]), score_bin_of(record["score"]))


def _class_name_of(record: dict[str, Any], category_mapping: Mapping[int, str]) -> str:
    category_id = int(record["category_id"])
    return category_mapping.get(category_id, f"category_{category_id:02d}")


def sample_fp_bg_audit(
    manifest: Mapping[str, Any],
    *,
    category_mapping: Mapping[int, str],
    max_per_stratum: int = 10,
    repeat_control_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[list[AuditSample], dict[str, Any]]:
    """分层抽检 FP_BG（确定性种子，含盲重复卡）。

    Args:
        manifest: N0-3 manifest 字典。
        category_mapping: ``{category_id: class_name}``。
        max_per_stratum: 每个分层单元最多抽多少正卡。
        repeat_control_fraction: 盲重复卡占每批正卡的比例。
        seed: 随机种子。

    Returns:
        (samples, summary)。samples 已按批号分组（连续 30 条一批）。
    """
    pool = build_fp_bg_sample_pool(manifest)
    if not pool:
        raise ValueError("FP_BG 抽检池为空")

    # 给每个候选打大类名。
    for record in pool:
        if not record.get("class_name"):
            record["class_name"] = _class_name_of(record, category_mapping)

    # 按 (大类, fold, 分数分位) 分层。
    strata: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in pool:
        strata[_group_key(record)].append(record)

    rng = random.Random(seed)
    positive_samples: list[AuditSample] = []
    stratum_counts: Counter[tuple[str, int, str]] = Counter()

    for stratum_key, records in sorted(strata.items()):
        # 每层内按分数降序取前 max_per_stratum。
        sorted_records = sorted(
            records, key=lambda r: -float(r["score"])
        )
        selected = sorted_records[:max_per_stratum]
        stratum_counts[stratum_key] += len(selected)
        for index, record in enumerate(selected):
            positive_samples.append(
                AuditSample(
                    audit_uid="",
                    proposal_uid=str(record.get("proposal_uid", "")),
                    image_id=int(record.get("image_id", 0)),
                    fold=int(record.get("fold", -1)),
                    category_id=int(record.get("category_id", -1)),
                    class_name=str(record.get("class_name", "")),
                    score=float(record.get("score", 0.0)),
                    bbox_xyxy=[float(v) for v in record.get("bbox_xyxy", [])],
                    size_bin=str(record.get("size_bin", "")),
                    source_group=str(record.get("source_group", "")),
                    score_bin=stratum_key[2],
                    is_repeat_control=False,
                )
            )

    # 生成盲重复卡：从正卡中随机抽取 fraction 比例作为重复。
    n_repeat = int(round(len(positive_samples) * repeat_control_fraction))
    repeat_source = rng.sample(positive_samples, min(n_repeat, len(positive_samples)))
    repeat_of_by_uid: dict[str, str] = {}
    repeat_samples: list[AuditSample] = []
    for source in repeat_source:
        repeat_uid = f"repeat-{source.proposal_uid}-{rng.randint(1000, 9999)}"
        repeat_of_by_uid[repeat_uid] = source.proposal_uid
        repeat_samples.append(
            AuditSample(
                audit_uid="",
                proposal_uid=source.proposal_uid,
                image_id=source.image_id,
                fold=source.fold,
                category_id=source.category_id,
                class_name=source.class_name,
                score=source.score,
                bbox_xyxy=list(source.bbox_xyxy),
                size_bin=source.size_bin,
                source_group=source.source_group,
                score_bin=source.score_bin,
                is_repeat_control=True,
                repeat_of=source.proposal_uid,
            )
        )

    # 混合并打乱，分配 audit_uid 与批号。
    mixed = positive_samples + repeat_samples
    rng.shuffle(mixed)
    samples: list[AuditSample] = []
    for batch_index, sample in enumerate(mixed):
        batch_no = batch_index // 30
        audit_uid = f"batch{batch_no:03d}-{sample.is_repeat_control and 'R' or 'P'}-{batch_index:05d}"
        samples.append(
            AuditSample(
                audit_uid=audit_uid,
                proposal_uid=sample.proposal_uid,
                image_id=sample.image_id,
                fold=sample.fold,
                category_id=sample.category_id,
                class_name=sample.class_name,
                score=sample.score,
                bbox_xyxy=sample.bbox_xyxy,
                size_bin=sample.size_bin,
                source_group=sample.source_group,
                score_bin=sample.score_bin,
                is_repeat_control=sample.is_repeat_control,
                repeat_of=sample.repeat_of,
            )
        )

    summary = {
        "pool_size": len(pool),
        "sampled_positives": len(positive_samples),
        "repeat_controls": len(repeat_samples),
        "total_samples": len(samples),
        "repeat_control_fraction": repeat_control_fraction,
        "strata_sampled": len(strata),
        "stratum_counts": {
            f"{key[0]}|f{key[1]}|{key[2]}": count
            for key, count in sorted(stratum_counts.items())
        },
        "seed": seed,
        "batch_size": 30,
    }
    return samples, summary


def audit_samples_to_csv(
    samples: Iterable[AuditSample],
    path: Path,
) -> None:
    """写人工标注表 CSV。label/labeler 列为空待填。"""
    import csv

    rows = []
    for sample in samples:
        row = sample.to_record()
        row["label"] = ""
        row["labeler"] = ""
        rows.append(row)

    fieldnames = [
        "audit_uid",
        "proposal_uid",
        "image_id",
        "fold",
        "category_id",
        "class_name",
        "score",
        "bbox_xyxy",
        "size_bin",
        "source_group",
        "score_bin",
        "is_repeat_control",
        "repeat_of",
        "label",
        "labeler",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def audit_samples_to_json(
    samples: list[AuditSample],
    path: Path,
    *,
    summary: Mapping[str, Any],
) -> None:
    """写抽检包 JSON（含样本与汇总）。"""
    payload = {
        "manifest_version": "fp_bg_audit_sample_v1",
        "summary": dict(summary),
        "samples": [sample.to_record() for sample in samples],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_audit_labels(path: Path) -> list[dict[str, Any]]:
    """读回人工标注表，校验标签合法性。"""
    import csv

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row.get("label", "").strip()
            if label and label not in HUMAN_LABELS:
                raise ValueError(
                    f"未知人工标签 {label!r}（合法: {HUMAN_LABELS}）"
                )
            rows.append(row)
    return rows


def compute_audit_summary(
    labeled_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """统计人工标签分布与重复卡一致性。

    重复卡一致性：``repeat_of`` 相同的两条样本标签一致的比例（只统计
    双方都标注的）。用于 N0-4 的人工程序质检。
    """
    label_counts: Counter[str] = Counter(
        row["label"] for row in labeled_rows if row.get("label")
    )
    unlabeled = sum(1 for row in labeled_rows if not row.get("label"))

    repeat_pairs: dict[str, list[str]] = defaultdict(list)
    for row in labeled_rows:
        repeat_of = row.get("repeat_of")
        if repeat_of and row.get("label"):
            repeat_pairs[repeat_of].append(row["label"])

    consistent = 0
    total_pairs = 0
    for labels in repeat_pairs.values():
        total_pairs += 1
        if len(set(labels)) == 1:
            consistent += 1

    return {
        "labeled": sum(label_counts.values()),
        "unlabeled": unlabeled,
        "label_counts": dict(label_counts),
        "repeat_pairs_total": total_pairs,
        "repeat_pairs_consistent": consistent,
        "repeat_consistency_rate": (
            consistent / total_pairs if total_pairs else None
        ),
    }
