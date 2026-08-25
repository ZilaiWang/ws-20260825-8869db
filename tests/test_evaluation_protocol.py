"""评估协议解析测试（含官方 V1.6 时延门槛）。"""

import pytest

from rsdet.evaluation.protocol import parse_evaluation_protocol

_BASE_CONFIG = {
    "protocol_versions": {
        "contract_version": "contract_v1",
        "eval_version": "official_eval_v1",
    },
    "task": {
        "class_names": ["ship", "aircraft", "vehicle"],
        "dataset_category_mapping": {
            "0": "ship",
            "4": "aircraft",
            "24": "vehicle",
        },
    },
    "official_evaluation": {
        "recall_min": 0.85,
        "fdr_max": 0.20,
        "latency_max_seconds": 20.0,
        "iou_thresholds": {
            "ship": 0.50,
            "aircraft": 0.50,
            "vehicle": 0.35,
        },
    },
}


def test_parses_latency_max_seconds():
    protocol = parse_evaluation_protocol(_BASE_CONFIG)
    assert protocol.latency_max_seconds == 20.0
    assert protocol.recall_min == 0.85
    assert protocol.fdr_max == 0.20
    assert protocol.class_names == ["ship", "aircraft", "vehicle"]
    assert protocol.iou_thresholds == {
        "ship": 0.50,
        "aircraft": 0.50,
        "vehicle": 0.35,
    }


def test_latency_optional_when_absent():
    config = {
        **{key: value for key, value in _BASE_CONFIG.items() if key != "protocol_versions"},
        "protocol_versions": dict(_BASE_CONFIG["protocol_versions"]),
    }
    del config["official_evaluation"]["latency_max_seconds"]
    protocol = parse_evaluation_protocol(config)
    assert protocol.latency_max_seconds is None


def test_invalid_latency_rejected():
    config = {
        **{key: value for key, value in _BASE_CONFIG.items() if key != "protocol_versions"},
        "protocol_versions": dict(_BASE_CONFIG["protocol_versions"]),
    }
    config["official_evaluation"]["latency_max_seconds"] = -1.0
    with pytest.raises(ValueError, match="latency_max_seconds"):
        parse_evaluation_protocol(config)


def test_missing_recall_min_rejected():
    config = {
        **{key: value for key, value in _BASE_CONFIG.items() if key != "protocol_versions"},
        "protocol_versions": dict(_BASE_CONFIG["protocol_versions"]),
    }
    del config["official_evaluation"]["recall_min"]
    with pytest.raises(KeyError):
        parse_evaluation_protocol(config)
