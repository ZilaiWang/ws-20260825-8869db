from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from copy import deepcopy
from pathlib import Path

import pytest

from rsdet.experiments.runtime_10k import (
    PHASES,
    audit_runtime_file,
    summarize_10k_runtime,
)


def _records(total_model: float = 2.0) -> list[dict]:
    result = []
    for index in range(13):
        phases = {phase: 0.1 for phase in PHASES}
        phases["model"] = total_model + index * 0.01
        result.append(
            {
                "run_index": index,
                "image_id": index + 1,
                "warmup": index < 3,
                "width": 10000,
                "height": 10000,
                "phases": phases,
            }
        )
    return result


def _contract() -> dict:
    return {
        "contract_version": "runtime_10k_benchmark_v1",
        "image_source_type": "synthetic",
        "image_manifest_sha256": "a" * 64,
        "model_key": "M1",
        "checkpoint_sha256": "b" * 64,
        "checkpoint_provenance_sha256": "d" * 64,
        "engineering_checkpoint_only": True,
        "config_sha256": "c" * 64,
        "hardware_sha256": "e" * 64,
        "tile_size": 1280,
        "overlap": 256,
        "expected_tile_count": 100,
        "timing_method": "perf_counter_with_torch_cuda_synchronize",
        "cuda_synchronized": True,
        "warmup_runs": 3,
        "minimum_measured_runs": 10,
    }


def _bind_contract_fields(records: list[dict]) -> list[dict]:
    for index, record in enumerate(records):
        record.update(
            {
                "image_content_sha256": f"{index:064x}",
                "image_source_type": "synthetic",
                "model_key": "M1",
                "checkpoint_sha256": "b" * 64,
                "config_sha256": "c" * 64,
                "timing_method": "perf_counter_with_torch_cuda_synchronize",
                "cuda_synchronized": True,
                "tile_size": 1280,
                "overlap": 256,
                "tile_count": 100,
                "raw_proposal_count": 100,
                "fused_proposal_count": 20,
                "peak_vram_mib": 4096,
            }
        )
    return records


def test_runtime_summary_reports_phases_and_passes() -> None:
    summary = summarize_10k_runtime(_records())
    assert summary["status"] == "pass"
    assert summary["warmup_runs"] == 3
    assert summary["measured_runs"] == 10
    assert summary["time_gate"]["passed"] is True
    assert set(summary["phase_summary"]) == set(PHASES)
    assert summary["wall_with_read"]["p95_seconds"] > summary["total_after_read"][
        "p95_seconds"
    ]


def test_runtime_gate_uses_each_image_max_not_only_p95() -> None:
    records = _records()
    records[-1]["phases"]["model"] = 21.0
    summary = summarize_10k_runtime(records)
    assert summary["status"] == "fail_time_limit"
    assert summary["time_gate"]["violating_run_indices"] == [12]


def test_runtime_contract_binds_provenance_and_prevents_official_claim() -> None:
    summary = summarize_10k_runtime(
        _bind_contract_fields(_records()),
        benchmark_contract=_contract(),
    )
    assert summary["time_gate"]["passed"] is True
    assert summary["time_gate"]["official_claim_eligible"] is False
    assert summary["proposal_counts"] == {"raw_total": 1000, "fused_total": 200}
    assert "Engineering evidence only" in summary["claim_note"]


def test_runtime_cannot_self_declare_unregistered_official_manifest() -> None:
    contract = _contract()
    contract["image_source_type"] = "real_official"
    with pytest.raises(ValueError, match="官方 manifest 注册表"):
        summarize_10k_runtime(
            _bind_contract_fields(_records()),
            benchmark_contract=contract,
        )


def test_runtime_rejects_wrong_size_and_inconsistent_total() -> None:
    wrong_size = _records()
    wrong_size[0]["width"] = 9999
    with pytest.raises(ValueError, match="不是冻结"):
        summarize_10k_runtime(wrong_size)

    wrong_total = deepcopy(_records())
    wrong_total[0]["total_after_read"] = 999.0
    with pytest.raises(ValueError, match="分项和"):
        summarize_10k_runtime(wrong_total)


def test_runtime_rejects_wrong_frozen_tile_count() -> None:
    records = _bind_contract_fields(_records())
    records[-1]["tile_count"] = 99
    with pytest.raises(ValueError, match="tile_count 必须为冻结值 100"):
        summarize_10k_runtime(records, benchmark_contract=_contract())

    contract = _contract()
    contract["expected_tile_count"] = 99
    with pytest.raises(ValueError, match="切片几何不一致"):
        summarize_10k_runtime(
            _bind_contract_fields(_records()),
            benchmark_contract=contract,
        )


def test_runtime_rejects_duplicate_measured_image_content() -> None:
    records = _bind_contract_fields(_records())
    records[-1]["image_content_sha256"] = records[-2]["image_content_sha256"]
    with pytest.raises(ValueError, match="measured runs 必须使用不同内容"):
        summarize_10k_runtime(records, benchmark_contract=_contract())


def test_runtime_rejects_noncanonical_schedule_and_contract_types() -> None:
    records = _bind_contract_fields(_records())
    records[2]["warmup"] = False
    records[3]["warmup"] = True
    with pytest.raises(ValueError, match="连续前缀"):
        summarize_10k_runtime(records, benchmark_contract=_contract())

    records = _bind_contract_fields(_records())
    records[-1]["run_index"] = 14
    with pytest.raises(ValueError, match="从 0 连续递增"):
        summarize_10k_runtime(records, benchmark_contract=_contract())

    records = _bind_contract_fields(_records())
    records[-1]["image_id"] = records[-2]["image_id"]
    with pytest.raises(ValueError, match="不同 image_id"):
        summarize_10k_runtime(records, benchmark_contract=_contract())

    contract = _contract()
    contract["warmup_runs"] = "3"
    with pytest.raises(ValueError, match="必须是整数"):
        summarize_10k_runtime(
            _bind_contract_fields(_records()),
            benchmark_contract=contract,
        )


def test_benchmark_cuda_sync_targets_configured_device_and_rejects_dummy() -> None:
    module = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/benchmark_10k_pipeline.py")
    )

    class FakeCuda:
        def __init__(self) -> None:
            self.devices: list[str] = []

        def is_available(self) -> bool:
            return True

        def synchronize(self, *, device: str) -> None:
            self.devices.append(device)

    class FakeTorch:
        cuda = FakeCuda()

    module["_cuda_sync"](FakeTorch, "cuda:1")
    assert FakeTorch.cuda.devices == ["cuda:1"]

    with pytest.raises(ValueError, match="只允许 adapter=ultralytics"):
        module["_require_real_adapter"]("dummy", object())


def test_benchmark_output_directory_is_published_atomically(tmp_path: Path) -> None:
    module = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/benchmark_10k_pipeline.py")
    )
    run = module["run"]
    output = tmp_path / "capture"
    args = argparse.Namespace(output_dir=output)

    def fail_capture(staged_args: argparse.Namespace, *, final_output_dir: Path) -> dict:
        assert final_output_dir == output.resolve()
        (staged_args.output_dir / "partial.json").write_text("partial")
        raise RuntimeError("injected failure")

    run.__globals__["_run_capture"] = fail_capture
    with pytest.raises(RuntimeError, match="injected failure"):
        run(args)
    assert not output.exists()
    assert list(tmp_path.glob(".capture.staging.*")) == []

    def pass_capture(staged_args: argparse.Namespace, *, final_output_dir: Path) -> dict:
        runtime = staged_args.output_dir / "runtime_samples.jsonl"
        predictions = staged_args.output_dir / "predictions_10k_low.json"
        runtime.write_text("{}\n")
        predictions.write_text("[]")
        return {
            "runtime_jsonl": str(runtime),
            "predictions": str(predictions),
            "runs": 13,
            "measured_runs": 10,
        }

    run.__globals__["_run_capture"] = pass_capture
    result = run(args)
    assert output.is_dir()
    assert (output / "runtime_samples.jsonl").read_text() == "{}\n"
    assert result["runtime_jsonl"] == str(output / "runtime_samples.jsonl")
    assert list(tmp_path.glob(".capture.staging.*")) == []


def test_formal_audit_cannot_relax_size_or_time_gate(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="10000x10000"):
        audit_runtime_file(
            input_path=missing,
            output_path=missing,
            hardware_path=missing,
            benchmark_contract_path=missing,
            expected_width=9999,
        )
    with pytest.raises(ValueError, match="20.0 秒"):
        audit_runtime_file(
            input_path=missing,
            output_path=missing,
            hardware_path=missing,
            benchmark_contract_path=missing,
            maximum_after_read_seconds=21.0,
        )


def test_runtime_audit_binds_hardware_sha256(tmp_path: Path) -> None:
    hardware = tmp_path / "hardware.json"
    hardware.write_text(
        json.dumps(
            {
                "gpu_name": "Example CUDA GPU",
                "driver_version": "test-driver",
                "torch_version": "test-torch",
                "cuda_runtime": "test-cuda",
                "other_gpu_processes": "none",
            }
        ),
        encoding="utf-8",
    )
    contract = _contract()
    contract["hardware_sha256"] = hashlib.sha256(hardware.read_bytes()).hexdigest()
    contract_path = tmp_path / "benchmark_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    runtime = tmp_path / "runtime.jsonl"
    runtime.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in _bind_contract_fields(_records())
        ),
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"
    summary = audit_runtime_file(
        input_path=runtime,
        output_path=output,
        hardware_path=hardware,
        benchmark_contract_path=contract_path,
    )
    assert summary["input"]["hardware_sha256"] == contract["hardware_sha256"]
    assert summary["time_gate"]["official_hardware_eligible"] is True
    assert summary["time_gate"]["official_claim_eligible"] is False

    hardware.write_text(
        hardware.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hardware record SHA"):
        audit_runtime_file(
            input_path=runtime,
            output_path=output,
            hardware_path=hardware,
            benchmark_contract_path=contract_path,
        )


def test_checkpoint_provenance_binds_formal_oof_lineage(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"formal-fold-checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    fold_metadata = tmp_path / "fold_metadata.json"
    fold_metadata.write_text(
        json.dumps(
            {
                "status": "fold_delivery_complete",
                "model_key": "M1",
                "held_out_fold": 0,
                "artifacts": {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": digest,
                },
            }
        ),
        encoding="utf-8",
    )
    fold_sha = hashlib.sha256(fold_metadata.read_bytes()).hexdigest()
    oof_metadata = tmp_path / "oof_metadata.json"
    oof_metadata.write_text(
        json.dumps(
            {
                "status": "complete_downstream_ready",
                "downstream_admission": True,
                "model_key": "M1",
                "folds": [
                    {
                        "fold": 0,
                        "checkpoint_sha256": digest,
                        "metadata_sha256": fold_sha,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    provenance = tmp_path / "checkpoint_provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "contract_version": "checkpoint_provenance_v1",
                "status": "checkpoint_lineage_verified",
                "engineering_checkpoint_only": True,
                "model_key": "M1",
                "source_fold": 0,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": digest,
                "fold_metadata": str(fold_metadata),
                "fold_metadata_sha256": fold_sha,
                "oof_metadata": str(oof_metadata),
                "oof_metadata_sha256": hashlib.sha256(
                    oof_metadata.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    module = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/benchmark_10k_pipeline.py")
    )
    validator = module["_validate_checkpoint_provenance"]
    contract = _contract()
    contract["checkpoint_sha256"] = digest
    contract["checkpoint_provenance_sha256"] = hashlib.sha256(
        provenance.read_bytes()
    ).hexdigest()

    assert validator(provenance, contract=contract) == checkpoint.resolve()

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint 实体/SHA"):
        validator(provenance, contract=contract)

    checkpoint.write_bytes(b"formal-fold-checkpoint")
    fold_payload = json.loads(fold_metadata.read_text(encoding="utf-8"))
    fold_payload["held_out_fold"] = 1
    fold_metadata.write_text(json.dumps(fold_payload), encoding="utf-8")
    changed_fold_sha = hashlib.sha256(fold_metadata.read_bytes()).hexdigest()
    oof_payload = json.loads(oof_metadata.read_text(encoding="utf-8"))
    oof_payload["folds"][0]["metadata_sha256"] = changed_fold_sha
    oof_metadata.write_text(json.dumps(oof_payload), encoding="utf-8")
    provenance_payload = json.loads(provenance.read_text(encoding="utf-8"))
    provenance_payload["fold_metadata_sha256"] = changed_fold_sha
    provenance_payload["oof_metadata_sha256"] = hashlib.sha256(
        oof_metadata.read_bytes()
    ).hexdigest()
    provenance.write_text(json.dumps(provenance_payload), encoding="utf-8")
    contract["checkpoint_provenance_sha256"] = hashlib.sha256(
        provenance.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="fold metadata 未闭环"):
        validator(provenance, contract=contract)
