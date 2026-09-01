"""从项目配置解析冻结的评估协议。"""

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationProtocol:
    """评估入口共用的版本和规则。"""

    contract_version: str
    eval_version: str
    ranking_version: str
    class_names: list[str]
    category_mapping: dict[int, str]
    iou_thresholds: dict[str, float]
    recall_min: float
    fdr_max: float
    latency_max_seconds: float | None = None
    metric_protocol: str = "platform_observed_20260831"


def parse_evaluation_protocol(
    project_config: dict[str, Any],
    *,
    class_names_override: list[str] | None = None,
) -> EvaluationProtocol:
    """解析项目配置；缺少冻结版本时明确拒绝评估。"""
    protocol_versions = project_config["protocol_versions"]
    contract_version = protocol_versions["contract_version"]
    eval_version = protocol_versions["eval_version"]
    ranking_version = protocol_versions.get("ranking_version", "official_ranking_v1_6")
    metric_protocol = protocol_versions.get(
        "metric_protocol", "platform_observed_20260831"
    )
    if not isinstance(contract_version, str) or not contract_version.strip():
        raise ValueError("contract_version 必须是非空字符串")
    if not isinstance(eval_version, str) or not eval_version.strip():
        raise ValueError("eval_version 必须是非空字符串")
    if not isinstance(ranking_version, str) or not ranking_version.strip():
        raise ValueError("ranking_version 必须是非空字符串")
    from rsdet.evaluation.platform_protocol import SUPPORTED_METRIC_PROTOCOLS

    if metric_protocol not in SUPPORTED_METRIC_PROTOCOLS:
        raise ValueError(
            "metric_protocol 必须是明确的冻结协议，当前支持: "
            f"{sorted(SUPPORTED_METRIC_PROTOCOLS)}"
        )

    task_config = project_config["task"]
    official_config = project_config["official_evaluation"]
    class_names = [str(name) for name in (class_names_override or task_config["class_names"])]
    if not class_names or len(class_names) != len(set(class_names)):
        raise ValueError("class_names 必须非空且不能重复")
    recall_min = float(official_config["recall_min"])
    fdr_max = float(official_config["fdr_max"])
    if not math.isfinite(recall_min) or not 0.0 <= recall_min <= 1.0:
        raise ValueError("recall_min 必须在 [0, 1] 内")
    if not math.isfinite(fdr_max) or not 0.0 <= fdr_max <= 1.0:
        raise ValueError("fdr_max 必须在 [0, 1] 内")
    latency_max_seconds: float | None = None
    if official_config.get("latency_max_seconds") is not None:
        latency_max_seconds = float(official_config["latency_max_seconds"])
        if not math.isfinite(latency_max_seconds) or latency_max_seconds <= 0.0:
            raise ValueError("latency_max_seconds 必须是正有限数")
    return EvaluationProtocol(
        contract_version=contract_version,
        eval_version=eval_version,
        ranking_version=ranking_version,
        class_names=class_names,
        category_mapping={
            int(category_id): str(class_name)
            for category_id, class_name in task_config["dataset_category_mapping"].items()
        },
        iou_thresholds={
            str(class_name): float(threshold)
            for class_name, threshold in official_config["iou_thresholds"].items()
        },
        recall_min=recall_min,
        fdr_max=fdr_max,
        latency_max_seconds=latency_max_seconds,
        metric_protocol=metric_protocol,
    )
