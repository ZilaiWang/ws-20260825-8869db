"""Auditable phase-level runtime summaries for the formal 10K pipeline."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from rsdet.experiments.cv3_oof import atomic_write_json, sha256_file

RUNTIME_CONTRACT_VERSION = "runtime_10k_v1"
BENCHMARK_CONTRACT_VERSION = "runtime_10k_benchmark_v1"
IMAGE_SOURCE_TYPES = {
    "real_official",
    "real_project_proxy",
    "synthetic",
    "stitched",
}
# Empty by design until an official 10K release is received, inventoried, and
# reviewed.  Adding an entry changes the scientific claim boundary and must be
# accompanied by a code-lock update and a new server task.
OFFICIAL_IMAGE_MANIFEST_REGISTRY: dict[str, str] = {}
OFFICIAL_TIMING_GPU_NAMES = {"NVIDIA GeForce RTX 3090"}
PHASES = (
    "image_read",
    "tiling",
    "preprocess",
    "model",
    "tile_postprocess",
    "coordinate_restore",
    "fusion",
    "serialization",
)
AFTER_READ_PHASES = tuple(phase for phase in PHASES if phase != "image_read")
PHASE_DEFINITIONS = {
    "image_read": "decode the source image and materialize an RGB uint8 array",
    "tiling": "generate deterministic tile coordinates only",
    "preprocess": "materialize contiguous RGB tile arrays on the host",
    "model": (
        "batched detector adapter calls, including adapter/framework preprocessing, "
        "GPU forward, decode, and adapter-internal tile NMS"
    ),
    "tile_postprocess": "validate/count the adapter's already-decoded predictions",
    "coordinate_restore": "restore tile boxes to clipped parent-image coordinates",
    "fusion": "apply cross-tile fine-class and coarse-group duplicate suppression",
    "serialization": "convert and write the per-run COCO prediction JSON",
}


def expected_tile_count(
    width: int,
    height: int,
    tile_size: int,
    overlap: int,
) -> int:
    """Return the deterministic full-coverage tile count for one image."""

    if width <= 0 or height <= 0 or tile_size <= 0:
        raise ValueError("width/height/tile_size 必须为正数")
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap 必须位于 [0,tile_size)")
    stride = tile_size - overlap

    def count_axis(length: int) -> int:
        if length <= tile_size:
            return 1
        return math.ceil((length - tile_size) / stride) + 1

    return count_axis(width) * count_axis(height)


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} 必须是 JSON 数值")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} 必须是有限非负秒数")
    return result


def _integer(value: Any, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} 必须是整数")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} 必须 >= {minimum}")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} 必须是 SHA256 字符串")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} 不是合法 SHA256")
    return normalized


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Return a deterministic linearly interpolated percentile."""

    if not values:
        raise ValueError("percentile 输入不能为空")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_10k_runtime(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_width: int = 10000,
    expected_height: int = 10000,
    minimum_measured_runs: int = 10,
    maximum_after_read_seconds: float = 20.0,
    benchmark_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate raw runs and report p50/p95 plus the per-image 20 s gate."""

    if expected_width <= 0 or expected_height <= 0:
        raise ValueError("expected image size 必须为正数")
    if minimum_measured_runs <= 0 or maximum_after_read_seconds <= 0.0:
        raise ValueError("runtime gate 参数必须为正数")
    if not records:
        raise ValueError("runtime records 不能为空")

    seen_indices: set[int] = set()
    normalized: list[dict[str, Any]] = []
    contract: dict[str, Any] | None = None
    frozen_tile_count: int | None = None
    if benchmark_contract is not None:
        contract = _validate_benchmark_contract(benchmark_contract)
        frozen_tile_count = expected_tile_count(
            expected_width,
            expected_height,
            int(contract["tile_size"]),
            int(contract["overlap"]),
        )
        if int(contract["expected_tile_count"]) != frozen_tile_count:
            raise ValueError(
                "benchmark contract expected_tile_count 与冻结图像/切片几何不一致: "
                f"declared={contract['expected_tile_count']}, "
                f"computed={frozen_tile_count}"
            )
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"records[{index}] 必须是对象")
        run_index = _integer(
            record.get("run_index"),
            f"records[{index}].run_index",
            minimum=0,
        )
        if run_index in seen_indices:
            raise ValueError(f"run_index 重复: {run_index}")
        seen_indices.add(run_index)
        width = _integer(record.get("width"), f"records[{index}].width", minimum=1)
        height = _integer(record.get("height"), f"records[{index}].height", minimum=1)
        if width != expected_width or height != expected_height:
            raise ValueError(
                f"run {run_index} 不是冻结的 {expected_width}x{expected_height}: "
                f"{width}x{height}"
            )
        if not isinstance(record.get("warmup"), bool):
            raise ValueError(f"records[{index}].warmup 必须是布尔值")
        phase_payload = record.get("phases")
        if not isinstance(phase_payload, Mapping):
            raise ValueError(f"records[{index}].phases 必须是对象")
        missing = set(PHASES) - set(phase_payload)
        extra = set(phase_payload) - set(PHASES)
        if missing or extra:
            raise ValueError(
                f"run {run_index} phase schema 不一致: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        phases = {
            phase: _finite_nonnegative(
                phase_payload[phase], f"records[{index}].phases.{phase}"
            )
            for phase in PHASES
        }
        scientific_fields: dict[str, Any] = {}
        if contract is not None:
            for field in (
                "image_content_sha256",
                "image_source_type",
                "model_key",
                "checkpoint_sha256",
                "config_sha256",
                "timing_method",
                "cuda_synchronized",
                "tile_size",
                "overlap",
                "tile_count",
                "raw_proposal_count",
                "fused_proposal_count",
                "peak_vram_mib",
            ):
                if field not in record:
                    raise ValueError(f"records[{index}] 缺少科学追溯字段 {field}")
            image_sha = _sha256(
                record["image_content_sha256"],
                f"records[{index}].image_content_sha256",
            )
            checkpoint_sha = _sha256(
                record["checkpoint_sha256"],
                f"records[{index}].checkpoint_sha256",
            )
            config_sha = _sha256(
                record["config_sha256"],
                f"records[{index}].config_sha256",
            )
            scientific_fields = {
                "image_content_sha256": image_sha,
                "image_source_type": str(record["image_source_type"]),
                "model_key": str(record["model_key"]),
                "checkpoint_sha256": checkpoint_sha,
                "config_sha256": config_sha,
                "timing_method": str(record["timing_method"]),
                "cuda_synchronized": record["cuda_synchronized"],
                "tile_size": _integer(
                    record["tile_size"], f"records[{index}].tile_size", minimum=1
                ),
                "overlap": _integer(
                    record["overlap"], f"records[{index}].overlap", minimum=0
                ),
                "tile_count": _integer(
                    record["tile_count"], f"records[{index}].tile_count", minimum=1
                ),
                "raw_proposal_count": _integer(
                    record["raw_proposal_count"],
                    f"records[{index}].raw_proposal_count",
                    minimum=0,
                ),
                "fused_proposal_count": _integer(
                    record["fused_proposal_count"],
                    f"records[{index}].fused_proposal_count",
                    minimum=0,
                ),
                "peak_vram_mib": _finite_nonnegative(
                    record["peak_vram_mib"], f"records[{index}].peak_vram_mib"
                ),
            }
            for field in (
                "image_source_type",
                "model_key",
                "checkpoint_sha256",
                "config_sha256",
                "timing_method",
                "cuda_synchronized",
                "tile_size",
                "overlap",
            ):
                if scientific_fields[field] != contract[field]:
                    raise ValueError(
                        f"records[{index}].{field} 与 benchmark contract 不一致"
                    )
            if scientific_fields["cuda_synchronized"] is not True:
                raise ValueError("正式 GPU 分段计时必须执行 CUDA synchronize")
            if scientific_fields["tile_count"] != frozen_tile_count:
                raise ValueError(
                    f"records[{index}].tile_count 必须为冻结值 "
                    f"{frozen_tile_count}，实际 {scientific_fields['tile_count']}"
                )
            if scientific_fields["raw_proposal_count"] < 0:
                raise ValueError("raw_proposal_count 不能为负")
            if not 0 <= scientific_fields["fused_proposal_count"] <= scientific_fields[
                "raw_proposal_count"
            ]:
                raise ValueError("fused_proposal_count 必须位于 [0, raw] 范围")
        after_read = sum(phases[phase] for phase in AFTER_READ_PHASES)
        wall_with_read = phases["image_read"] + after_read
        reported_after_read = record.get("total_after_read")
        if reported_after_read is not None and not math.isclose(
            _finite_nonnegative(
                reported_after_read, f"records[{index}].total_after_read"
            ),
            after_read,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(f"run {run_index} total_after_read 与分项和不一致")
        normalized.append(
            {
                "run_index": run_index,
                "image_id": _integer(
                    record.get("image_id", run_index),
                    f"records[{index}].image_id",
                    minimum=0,
                ),
                "warmup": record["warmup"],
                "width": width,
                "height": height,
                "phases": phases,
                "total_after_read": after_read,
                "wall_with_read": wall_with_read,
                **scientific_fields,
            }
        )

    ordered = sorted(normalized, key=lambda item: item["run_index"])
    expected_indices = list(range(len(ordered)))
    actual_indices = [record["run_index"] for record in ordered]
    if actual_indices != expected_indices:
        raise ValueError(
            "run_index 必须从 0 连续递增，禁止删除或重排运行: "
            f"actual={actual_indices}"
        )
    measured = [record for record in normalized if not record["warmup"]]
    if len(measured) < minimum_measured_runs:
        raise ValueError(
            f"正式测速至少需要 {minimum_measured_runs} 次非预热运行，实际 {len(measured)}"
        )
    if contract is not None:
        warmup_runs = int(contract["warmup_runs"])
        measured_runs = int(contract["minimum_measured_runs"])
        if len(normalized) - len(measured) != warmup_runs:
            raise ValueError("实际 warmup 次数与 benchmark contract 不一致")
        if len(measured) != measured_runs:
            raise ValueError(
                "实际 measured 次数必须与冻结计划完全一致: "
                f"expected={measured_runs}, actual={len(measured)}"
            )
        expected_warmup_flags = [True] * warmup_runs + [False] * measured_runs
        if [record["warmup"] for record in ordered] != expected_warmup_flags:
            raise ValueError("warmup 必须是运行序列的连续前缀，measured 不得交错")
        measured_shas = [
            str(record["image_content_sha256"]) for record in measured
        ]
        if len(set(measured_shas)) != len(measured_shas):
            raise ValueError("measured runs 必须使用不同内容 SHA 的图像")
        measured_image_ids = [record["image_id"] for record in measured]
        if len(set(measured_image_ids)) != len(measured_image_ids):
            raise ValueError("measured runs 必须使用不同 image_id")
    phase_summary = {
        phase: {
            "p50_seconds": _percentile(
                [record["phases"][phase] for record in measured], 0.50
            ),
            "p95_seconds": _percentile(
                [record["phases"][phase] for record in measured], 0.95
            ),
            "max_seconds": max(record["phases"][phase] for record in measured),
        }
        for phase in PHASES
    }
    after_read_values = [record["total_after_read"] for record in measured]
    wall_values = [record["wall_with_read"] for record in measured]
    violating_runs = [
        record["run_index"]
        for record in measured
        if record["total_after_read"] > maximum_after_read_seconds
    ]
    source_type = None if contract is None else contract["image_source_type"]
    official_image_source_eligible = source_type == "real_official"
    # Hardware and final-checkpoint eligibility are only available in
    # ``audit_runtime_file``.  A phase-only summary must never self-upgrade to
    # an official claim merely because a free-text source type says official.
    official_claim_eligible = False
    result = {
        "contract_version": RUNTIME_CONTRACT_VERSION,
        "status": "pass" if not violating_runs else "fail_time_limit",
        "expected_image_size": [expected_width, expected_height],
        "warmup_runs": len(normalized) - len(measured),
        "measured_runs": len(measured),
        "phase_definitions": dict(PHASE_DEFINITIONS),
        "phase_summary": phase_summary,
        "total_after_read": {
            "p50_seconds": _percentile(after_read_values, 0.50),
            "p95_seconds": _percentile(after_read_values, 0.95),
            "max_seconds": max(after_read_values),
        },
        "wall_with_read": {
            "p50_seconds": _percentile(wall_values, 0.50),
            "p95_seconds": _percentile(wall_values, 0.95),
            "max_seconds": max(wall_values),
        },
        "time_gate": {
            "scope": "each_image_after_read",
            "maximum_seconds": maximum_after_read_seconds,
            "passed": not violating_runs,
            "violating_run_indices": violating_runs,
            "official_claim_eligible": official_claim_eligible,
            "official_image_source_eligible": official_image_source_eligible,
        },
        "records": ordered,
    }
    if contract is not None:
        result["benchmark_contract"] = contract
        result["proposal_counts"] = {
            "raw_total": sum(record["raw_proposal_count"] for record in measured),
            "fused_total": sum(record["fused_proposal_count"] for record in measured),
        }
        result["peak_vram_mib"] = max(record["peak_vram_mib"] for record in measured)
        result["claim_note"] = (
            "Eligible for the official-size timing claim."
            if official_claim_eligible
            else (
                "Engineering evidence only: synthetic, stitched, or proxy imagery "
                "must never be reported as official timing evidence."
            )
        )
    return result


def _validate_benchmark_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("contract_version") != BENCHMARK_CONTRACT_VERSION:
        raise ValueError("未知 benchmark contract version")
    source_type_value = payload.get("image_source_type")
    if not isinstance(source_type_value, str):
        raise ValueError("benchmark contract image_source_type 必须是字符串")
    source_type = source_type_value.strip()
    if source_type not in IMAGE_SOURCE_TYPES:
        raise ValueError(f"未知 image_source_type: {source_type!r}")
    result = dict(payload)
    for field in ("model_key", "checkpoint_sha256", "config_sha256", "timing_method"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise ValueError(f"benchmark contract 缺少 {field}")
    model_key = result["model_key"].strip().upper()
    if model_key not in {"M1", "M3"}:
        raise ValueError("benchmark contract model_key 只允许 M1/M3")
    if result["timing_method"] != "perf_counter_with_torch_cuda_synchronize":
        raise ValueError("benchmark contract timing_method 与正式采集器不一致")
    for field in (
        "checkpoint_sha256",
        "checkpoint_provenance_sha256",
        "config_sha256",
        "image_manifest_sha256",
        "hardware_sha256",
    ):
        result[field] = _sha256(
            result.get(field), f"benchmark contract {field}"
        )
    if result.get("cuda_synchronized") is not True:
        raise ValueError("benchmark contract 必须声明 cuda_synchronized=true")
    if not isinstance(result.get("engineering_checkpoint_only"), bool):
        raise ValueError(
            "benchmark contract 必须显式声明 engineering_checkpoint_only 布尔值"
        )
    for field in (
        "tile_size",
        "expected_tile_count",
        "warmup_runs",
        "minimum_measured_runs",
    ):
        result[field] = _integer(
            result.get(field), f"benchmark contract {field}", minimum=1
        )
    overlap = _integer(
        result.get("overlap"), "benchmark contract overlap", minimum=0
    )
    if not 0 <= overlap < result["tile_size"]:
        raise ValueError("benchmark contract overlap 必须位于 [0,tile_size)")
    result["overlap"] = overlap
    result["image_source_type"] = source_type
    result["model_key"] = model_key
    if source_type == "real_official":
        registry_id = str(
            result.get("official_manifest_registry_id", "")
        ).strip()
        if registry_id not in OFFICIAL_IMAGE_MANIFEST_REGISTRY:
            raise ValueError(
                "real_official 输入尚未进入代码内官方 manifest 注册表；"
                "禁止仅改 source_type 获得官方声明资格"
            )
        if (
            result["image_manifest_sha256"]
            != OFFICIAL_IMAGE_MANIFEST_REGISTRY[registry_id]
        ):
            raise ValueError("官方 image manifest SHA 与注册表不一致")
        result["official_manifest_registry_id"] = registry_id
    return result


def load_runtime_jsonl(path: str | Path) -> list[Mapping[str, Any]]:
    """Load non-empty JSON objects from a JSONL runtime ledger."""

    result: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number}: 每行必须是 JSON 对象")
        result.append(value)
    if not result:
        raise ValueError("runtime JSONL 没有记录")
    return result


def audit_runtime_file(
    *,
    input_path: str | Path,
    output_path: str | Path,
    hardware_path: str | Path,
    benchmark_contract_path: str | Path,
    expected_width: int = 10000,
    expected_height: int = 10000,
    minimum_measured_runs: int = 10,
    maximum_after_read_seconds: float = 20.0,
) -> dict[str, Any]:
    """Summarize a ledger and bind it to an immutable hardware record."""

    if (expected_width, expected_height) != (10000, 10000):
        raise ValueError("正式 E-10K 审计只接受冻结的 10000x10000 尺寸")
    if not math.isclose(maximum_after_read_seconds, 20.0, abs_tol=1e-12):
        raise ValueError("正式 E-10K 审计的硬门禁必须固定为 20.0 秒")
    hardware_file = Path(hardware_path)
    if not hardware_file.is_file():
        raise FileNotFoundError(f"hardware record 不存在: {hardware_file}")
    hardware = json.loads(hardware_file.read_text(encoding="utf-8"))
    if not isinstance(hardware, Mapping):
        raise ValueError("hardware record 必须是 JSON 对象")
    for field in ("gpu_name", "driver_version", "torch_version", "cuda_runtime"):
        if not str(hardware.get(field, "")).strip():
            raise ValueError(f"hardware record 缺少 {field}")
    benchmark_file = Path(benchmark_contract_path)
    if not benchmark_file.is_file():
        raise FileNotFoundError(f"benchmark contract 不存在: {benchmark_file}")
    benchmark = json.loads(benchmark_file.read_text(encoding="utf-8"))
    if not isinstance(benchmark, Mapping):
        raise ValueError("benchmark contract 必须是 JSON 对象")
    summary = summarize_10k_runtime(
        load_runtime_jsonl(input_path),
        expected_width=expected_width,
        expected_height=expected_height,
        minimum_measured_runs=minimum_measured_runs,
        maximum_after_read_seconds=maximum_after_read_seconds,
        benchmark_contract=benchmark,
    )
    actual_hardware_sha256 = sha256_file(hardware_file)
    if (
        actual_hardware_sha256
        != summary["benchmark_contract"]["hardware_sha256"]
    ):
        raise ValueError("hardware record SHA 与 benchmark contract 不一致")
    summary["input"] = {
        "runtime_jsonl": str(Path(input_path)),
        "runtime_jsonl_sha256": sha256_file(input_path),
        "hardware": str(hardware_file),
        "hardware_sha256": actual_hardware_sha256,
        "benchmark_contract": str(benchmark_file),
        "benchmark_contract_sha256": sha256_file(benchmark_file),
    }
    summary["hardware"] = dict(hardware)
    hardware_eligible = (
        str(hardware["gpu_name"]).strip() in OFFICIAL_TIMING_GPU_NAMES
        and str(hardware.get("other_gpu_processes", "")).strip().lower()
        == "none"
    )
    official_setup_eligible = (
        summary["time_gate"]["official_image_source_eligible"]
        and hardware_eligible
        and benchmark.get("engineering_checkpoint_only") is False
    )
    official_claim_eligible = (
        summary["time_gate"]["passed"] and official_setup_eligible
    )
    summary["time_gate"]["official_hardware_eligible"] = hardware_eligible
    summary["time_gate"]["official_setup_eligible"] = official_setup_eligible
    summary["time_gate"]["official_claim_eligible"] = official_claim_eligible
    if official_claim_eligible:
        summary["claim_note"] = "Eligible for the official-size timing pass claim."
    elif official_setup_eligible:
        summary["claim_note"] = (
            "Official setup eligibility verified, but the per-image 20 s gate "
            "failed; this may only be reported as an official timing failure."
        )
    else:
        summary["claim_note"] = (
            "Engineering evidence only: official eligibility additionally "
            "requires a code-registered official manifest, the approved RTX "
            "3090 hardware with no competing GPU process, and a non-engineering "
            "checkpoint contract."
        )
    atomic_write_json(output_path, summary)
    return summary


__all__ = [
    "AFTER_READ_PHASES",
    "BENCHMARK_CONTRACT_VERSION",
    "IMAGE_SOURCE_TYPES",
    "OFFICIAL_IMAGE_MANIFEST_REGISTRY",
    "OFFICIAL_TIMING_GPU_NAMES",
    "PHASES",
    "PHASE_DEFINITIONS",
    "RUNTIME_CONTRACT_VERSION",
    "audit_runtime_file",
    "expected_tile_count",
    "load_runtime_jsonl",
    "summarize_10k_runtime",
]
