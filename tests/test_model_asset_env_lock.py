from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsdet.experiments.model_asset_env_lock import (
    LOCK_VERSION,
    atomic_write_new_json,
    build_lock_payload,
    canonical_json_sha256,
    load_and_validate_spec,
    sha256_file,
    verify_existing_lock,
)


def _environment() -> dict:
    return {
        "python_version": "3.10.12",
        "packages": {
            "Pillow": "10.4.0",
            "PyYAML": "6.0.2",
            "numpy": "1.26.4",
            "torch": "2.5.1+cu121",
            "torchvision": "0.20.1+cu121",
            "ultralytics": "8.4.103",
        },
        "installed_distributions": {
            "numpy": "1.26.4",
            "opencv-python": "4.10.0.84",
            "torch": "2.5.1+cu121",
            "torchvision": "0.20.1+cu121",
            "ultralytics": "8.4.103",
        },
        "cuda": {
            "available": True,
            "torch_runtime_version": "12.1",
            "cudnn_version": 90100,
        },
        "torch_gpu": {
            "device_index": 0,
            "name": "NVIDIA GeForce RTX 4080 SUPER",
            "device_count": 1,
            "total_memory_bytes": 34351349760,
            "compute_capability": [8, 9],
        },
        "nvidia_smi": {
            "visible_physical_gpu_count": 1,
            "index": 0,
            "name": "NVIDIA GeForce RTX 4080 SUPER",
            "uuid": "GPU-test",
            "driver_version": "595.58.03",
            "memory_total_mib": 32760,
        },
    }


def _write_spec_and_assets(tmp_path: Path) -> tuple[Path, Path]:
    asset_root = tmp_path / "assets"
    asset_root.mkdir(parents=True)
    first = asset_root / "a.pt"
    second = asset_root / "b.pt"
    first.write_bytes(b"asset-a")
    second.write_bytes(b"asset-b")
    spec = {
        "schema_version": "cv3_model_asset_env_spec_v1",
        "python_version": "3.10.12",
        "packages": _environment()["packages"],
        "cuda": {"required": True, "runtime_version": "12.1"},
        "gpu": {"required": True},
        "assets": {
            "a": {
                "filename": "a.pt",
                "official_url": "https://example.test/a.pt",
                "size_bytes": first.stat().st_size,
                "sha256": sha256_file(first),
            },
            "b": {
                "filename": "b.pt",
                "official_url": "https://example.test/b.pt",
                "size_bytes": second.stat().st_size,
                "sha256": sha256_file(second),
            },
        },
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return spec_path, asset_root


def test_build_write_and_verify_immutable_lock(tmp_path: Path) -> None:
    spec_path, asset_root = _write_spec_and_assets(tmp_path)
    payload = build_lock_payload(
        spec_path=spec_path,
        asset_root=asset_root,
        environment=_environment(),
        expected_gpu="NVIDIA GeForce RTX 4080 SUPER",
    )
    assert payload["schema_version"] == LOCK_VERSION
    body = dict(payload)
    fingerprint = body.pop("lock_fingerprint")
    assert fingerprint == canonical_json_sha256(body)
    assert payload["assets"]["a"]["official_url"] == "https://example.test/a.pt"

    lock_path = tmp_path / "MODEL_ASSET_ENV_LOCK.json"
    atomic_write_new_json(lock_path, payload)
    assert lock_path.stat().st_mode & 0o222 == 0
    report = verify_existing_lock(
        spec_path=spec_path,
        asset_root=asset_root,
        lock_path=lock_path,
        environment=_environment(),
        expected_gpu="NVIDIA GeForce RTX 4080 SUPER",
    )
    assert report["status"] == "pass"
    assert report["asset_count"] == 2
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        atomic_write_new_json(lock_path, payload)
    lock_path.chmod(0o644)
    with pytest.raises(PermissionError, match="存在写权限"):
        verify_existing_lock(
            spec_path=spec_path,
            asset_root=asset_root,
            lock_path=lock_path,
            environment=_environment(),
            expected_gpu="NVIDIA GeForce RTX 4080 SUPER",
        )


def test_asset_tamper_and_environment_drift_are_rejected(tmp_path: Path) -> None:
    spec_path, asset_root = _write_spec_and_assets(tmp_path)
    payload = build_lock_payload(
        spec_path=spec_path,
        asset_root=asset_root,
        environment=_environment(),
        expected_gpu="NVIDIA GeForce RTX 4080 SUPER",
    )
    lock_path = tmp_path / "MODEL_ASSET_ENV_LOCK.json"
    atomic_write_new_json(lock_path, payload)

    (asset_root / "a.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="大小不符|SHA-256 不符"):
        verify_existing_lock(
            spec_path=spec_path,
            asset_root=asset_root,
            lock_path=lock_path,
            environment=_environment(),
            expected_gpu="NVIDIA GeForce RTX 4080 SUPER",
        )

    spec_path, asset_root = _write_spec_and_assets(tmp_path / "fresh")
    bad_environment = _environment()
    bad_environment["packages"]["ultralytics"] = "8.4.52"
    with pytest.raises(ValueError, match="ultralytics 版本不符"):
        build_lock_payload(
            spec_path=spec_path,
            asset_root=asset_root,
            environment=bad_environment,
            expected_gpu="NVIDIA GeForce RTX 4080 SUPER",
        )

    payload = build_lock_payload(
        spec_path=spec_path,
        asset_root=asset_root,
        environment=_environment(),
        expected_gpu="NVIDIA GeForce RTX 4080 SUPER",
    )
    lock_path = tmp_path / "fresh" / "MODEL_ASSET_ENV_LOCK.json"
    atomic_write_new_json(lock_path, payload)
    transitive_drift = _environment()
    transitive_drift["installed_distributions"]["opencv-python"] = "4.11.0.86"
    with pytest.raises(ValueError, match="重算结果不一致"):
        verify_existing_lock(
            spec_path=spec_path,
            asset_root=asset_root,
            lock_path=lock_path,
            environment=transitive_drift,
            expected_gpu="NVIDIA GeForce RTX 4080 SUPER",
        )


def test_spec_rejects_path_escape_and_non_https(tmp_path: Path) -> None:
    spec_path, _ = _write_spec_and_assets(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["assets"]["a"]["filename"] = "../a.pt"
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="不含目录"):
        load_and_validate_spec(spec_path)

    payload["assets"]["a"]["filename"] = "a.pt"
    payload["assets"]["a"]["official_url"] = "http://example.test/a.pt"
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="HTTPS"):
        load_and_validate_spec(spec_path)


def test_lock_fails_when_expected_gpu_differs(tmp_path: Path) -> None:
    spec_path, asset_root = _write_spec_and_assets(tmp_path)
    with pytest.raises(ValueError, match="GPU 不符"):
        build_lock_payload(
            spec_path=spec_path,
            asset_root=asset_root,
            environment=_environment(),
            expected_gpu="NVIDIA GeForce RTX 3090",
        )


def test_lock_requires_explicit_nonempty_gpu_identity(tmp_path: Path) -> None:
    spec_path, asset_root = _write_spec_and_assets(tmp_path)
    with pytest.raises(ValueError, match="非空 expected_gpu"):
        build_lock_payload(
            spec_path=spec_path,
            asset_root=asset_root,
            environment=_environment(),
            expected_gpu="",
        )


def test_project_spec_is_the_reviewed_m1_m3_contract() -> None:
    spec = load_and_validate_spec(
        Path(__file__).parents[1] / "configs" / "experiments" / "cv3_model_asset_env.json"
    )
    assert spec["python_version"] == "3.10.12"
    assert spec["packages"] == {
        "Pillow": "10.4.0",
        "PyYAML": "6.0.2",
        "numpy": "1.26.4",
        "torch": "2.5.1+cu121",
        "torchvision": "0.20.1+cu121",
        "ultralytics": "8.4.103",
    }
    assert spec["cuda"] == {"required": True, "runtime_version": "12.1"}
    assert spec["assets"] == {
        "m1_yolo26s": {
            "filename": "yolo26s.pt",
            "official_url": (
                "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt"
            ),
            "size_bytes": 20422725,
            "sha256": ("646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"),
        },
        "m3_rtdetr_l": {
            "filename": "rtdetr-l.pt",
            "official_url": (
                "https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-l.pt"
            ),
            "size_bytes": 66511432,
            "sha256": ("6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f"),
        },
    }
