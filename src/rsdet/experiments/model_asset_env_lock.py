"""Immutable model-asset and runtime-environment lock for formal CV3 runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SPEC_VERSION = "cv3_model_asset_env_spec_v1"
LOCK_VERSION = "cv3_model_asset_env_lock_v1"
HEX_DIGITS = frozenset("0123456789abcdef")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a mapping using the project's deterministic JSON representation."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(char in HEX_DIGITS for char in text)


def load_and_validate_spec(path: Path) -> dict[str, Any]:
    """Load and strictly validate the human-reviewed lock specification."""

    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("资产环境配置必须是 JSON object")
    expected_top = {
        "schema_version",
        "python_version",
        "packages",
        "cuda",
        "gpu",
        "assets",
    }
    if set(payload) != expected_top:
        raise ValueError(
            "资产环境配置顶层字段不一致: "
            f"missing={sorted(expected_top - set(payload))}, "
            f"extra={sorted(set(payload) - expected_top)}"
        )
    if payload["schema_version"] != SPEC_VERSION:
        raise ValueError(f"schema_version 必须为 {SPEC_VERSION}")
    if not isinstance(payload["python_version"], str) or not payload["python_version"]:
        raise ValueError("python_version 必须为非空字符串")
    packages = payload["packages"]
    if not isinstance(packages, dict) or not packages:
        raise ValueError("packages 必须为非空 object")
    for name, version in packages.items():
        if not isinstance(name, str) or not name:
            raise ValueError("package 名称必须为非空字符串")
        if not isinstance(version, str) or not version:
            raise ValueError(f"package {name} 的版本必须为非空字符串")
    cuda = payload["cuda"]
    if set(cuda) != {"required", "runtime_version"}:
        raise ValueError("cuda 必须且仅包含 required/runtime_version")
    if cuda["required"] is not True:
        raise ValueError("正式模型环境必须要求 CUDA")
    if not isinstance(cuda["runtime_version"], str) or not cuda["runtime_version"]:
        raise ValueError("cuda.runtime_version 必须为非空字符串")
    if payload["gpu"] != {"required": True}:
        raise ValueError("gpu 必须冻结为 required=true")
    assets = payload["assets"]
    if not isinstance(assets, dict) or not assets:
        raise ValueError("assets 必须为非空 object")
    filenames: set[str] = set()
    for key, item in assets.items():
        if not isinstance(key, str) or not key or not isinstance(item, dict):
            raise ValueError("asset key/item 非法")
        expected_asset_fields = {
            "filename",
            "official_url",
            "size_bytes",
            "sha256",
        }
        if set(item) != expected_asset_fields:
            raise ValueError(f"asset {key} 字段不一致")
        filename = item["filename"]
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise ValueError(f"asset {key} filename 必须是不含目录的文件名")
        if filename in filenames:
            raise ValueError(f"asset filename 重复: {filename}")
        filenames.add(filename)
        url = item["official_url"]
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError(f"asset {key} 必须使用 HTTPS 官方 URL")
        if isinstance(item["size_bytes"], bool) or int(item["size_bytes"]) <= 0:
            raise ValueError(f"asset {key} size_bytes 必须为正整数")
        if not _is_sha256(item["sha256"]):
            raise ValueError(f"asset {key} SHA-256 非法")
        item["sha256"] = str(item["sha256"]).lower()
        item["size_bytes"] = int(item["size_bytes"])
    return payload


def _query_nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("nvidia-smi 未返回 GPU")
    values = [part.strip() for part in lines[0].split(",")]
    if len(values) != 5:
        raise RuntimeError(f"无法解析 nvidia-smi 输出: {lines[0]!r}")
    return {
        "visible_physical_gpu_count": len(lines),
        "index": int(values[0]),
        "name": values[1],
        "uuid": values[2],
        "driver_version": values[3],
        "memory_total_mib": int(values[4]),
    }


def collect_environment() -> dict[str, Any]:
    """Collect the exact runtime facts that must accompany M1/M3."""

    import torch

    packages: dict[str, str | None] = {}
    for distribution in (
        "Pillow",
        "PyYAML",
        "numpy",
        "torch",
        "torchvision",
        "ultralytics",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    installed_distributions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name", "")).strip()
        if name:
            installed_distributions[name] = distribution.version
    cuda_available = bool(torch.cuda.is_available())
    torch_gpu: dict[str, Any] | None = None
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        torch_gpu = {
            "device_index": 0,
            "name": torch.cuda.get_device_name(0),
            "device_count": int(torch.cuda.device_count()),
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        }
    try:
        nvidia_smi = _query_nvidia_smi()
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError, ValueError):
        nvidia_smi = None
    return {
        "python_version": platform.python_version(),
        "packages": packages,
        # The six packages above are the reviewed ABI/scientific gate.  This
        # complete inventory additionally makes the lock sensitive to any
        # transitive dependency drift inside the shared formal-run venv.
        "installed_distributions": dict(sorted(installed_distributions.items())),
        "cuda": {
            "available": cuda_available,
            "torch_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "torch_gpu": torch_gpu,
        "nvidia_smi": nvidia_smi,
    }


def validate_environment(
    spec: Mapping[str, Any],
    environment: Mapping[str, Any],
    *,
    expected_gpu: str,
) -> None:
    """Raise when the observed environment diverges from the frozen spec."""

    failures: list[str] = []
    if not isinstance(expected_gpu, str) or not expected_gpu.strip():
        failures.append("正式模型环境必须提供非空 expected_gpu")
    if environment.get("python_version") != spec["python_version"]:
        failures.append(
            "Python 版本不符: "
            f"expected={spec['python_version']}, actual={environment.get('python_version')}"
        )
    observed_packages = environment.get("packages")
    if not isinstance(observed_packages, Mapping):
        failures.append("packages 观测缺失")
    else:
        for name, expected in spec["packages"].items():
            actual = observed_packages.get(name)
            if actual != expected:
                failures.append(f"{name} 版本不符: expected={expected}, actual={actual}")
    cuda = environment.get("cuda")
    if not isinstance(cuda, Mapping):
        failures.append("CUDA 观测缺失")
    else:
        if cuda.get("available") is not True:
            failures.append("CUDA 不可用")
        if cuda.get("torch_runtime_version") != spec["cuda"]["runtime_version"]:
            failures.append(
                "CUDA runtime 不符: "
                f"expected={spec['cuda']['runtime_version']}, "
                f"actual={cuda.get('torch_runtime_version')}"
            )
    torch_gpu = environment.get("torch_gpu")
    nvidia_smi = environment.get("nvidia_smi")
    if not isinstance(torch_gpu, Mapping):
        failures.append("PyTorch GPU 观测缺失")
    if not isinstance(nvidia_smi, Mapping):
        failures.append("nvidia-smi GPU provenance 缺失")
    if isinstance(torch_gpu, Mapping) and isinstance(nvidia_smi, Mapping):
        if torch_gpu.get("name") != nvidia_smi.get("name"):
            failures.append(
                "PyTorch 与 nvidia-smi GPU 名称不一致: "
                f"{torch_gpu.get('name')} != {nvidia_smi.get('name')}"
            )
    actual_name = torch_gpu.get("name") if isinstance(torch_gpu, Mapping) else None
    if actual_name != expected_gpu:
        failures.append(f"GPU 不符: expected={expected_gpu}, actual={actual_name}")
    if failures:
        raise ValueError("; ".join(failures))


def build_lock_payload(
    *,
    spec_path: Path,
    asset_root: Path,
    environment: Mapping[str, Any],
    expected_gpu: str,
) -> dict[str, Any]:
    """Validate all inputs and produce the deterministic lock payload."""

    spec_path = spec_path.expanduser().resolve()
    asset_root = asset_root.expanduser().resolve()
    spec = load_and_validate_spec(spec_path)
    validate_environment(spec, environment, expected_gpu=expected_gpu)
    if not asset_root.is_dir():
        raise FileNotFoundError(asset_root)
    assets: dict[str, dict[str, Any]] = {}
    for key, expected in sorted(spec["assets"].items()):
        path = (asset_root / expected["filename"]).resolve()
        try:
            path.relative_to(asset_root)
        except ValueError as error:
            raise ValueError(f"asset {key} 路径逃逸 asset-root") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_size != expected["size_bytes"]:
            raise ValueError(
                f"asset {key} 大小不符: expected={expected['size_bytes']}, actual={actual_size}"
            )
        if actual_sha != expected["sha256"]:
            raise ValueError(
                f"asset {key} SHA-256 不符: expected={expected['sha256']}, actual={actual_sha}"
            )
        assets[key] = {
            "filename": expected["filename"],
            "path": str(path),
            "official_url": expected["official_url"],
            "size_bytes": actual_size,
            "sha256": actual_sha,
            "provenance_policy": "official_url_from_reviewed_spec",
        }
    payload: dict[str, Any] = {
        "schema_version": LOCK_VERSION,
        "spec": {
            "path": str(spec_path),
            "sha256": sha256_file(spec_path),
            "schema_version": spec["schema_version"],
        },
        "environment": dict(environment),
        "gpu_gate": {
            "expected_name": expected_gpu,
            "exact_name_required": True,
        },
        "assets": assets,
    }
    payload["lock_fingerprint"] = canonical_json_sha256(payload)
    return payload


def validate_lock_fingerprint(payload: Mapping[str, Any]) -> None:
    recorded = payload.get("lock_fingerprint")
    if not _is_sha256(recorded):
        raise ValueError("lock_fingerprint 缺失或非法")
    body = dict(payload)
    body.pop("lock_fingerprint", None)
    recomputed = canonical_json_sha256(body)
    if recorded != recomputed:
        raise ValueError(f"lock_fingerprint 不匹配: recorded={recorded}, recomputed={recomputed}")


def verify_existing_lock(
    *,
    spec_path: Path,
    asset_root: Path,
    lock_path: Path,
    environment: Mapping[str, Any],
    expected_gpu: str,
) -> dict[str, Any]:
    """Recompute the lock and require exact equality with the immutable record."""

    lock_path = lock_path.expanduser().resolve()
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    if lock_path.stat().st_mode & 0o222:
        raise PermissionError(
            f"MODEL_ASSET_ENV_LOCK.json 存在写权限，不满足不可变门禁: {lock_path}"
        )
    observed = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(observed, dict):
        raise ValueError("MODEL_ASSET_ENV_LOCK.json 必须是 object")
    validate_lock_fingerprint(observed)
    expected = build_lock_payload(
        spec_path=spec_path,
        asset_root=asset_root,
        environment=environment,
        expected_gpu=expected_gpu,
    )
    if observed != expected:
        raise ValueError("现有 MODEL_ASSET_ENV_LOCK.json 与重算结果不一致")
    return {
        "status": "pass",
        "schema_version": LOCK_VERSION,
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "lock_fingerprint": observed["lock_fingerprint"],
        "asset_count": len(observed["assets"]),
        "gpu_name": observed["environment"]["torch_gpu"]["name"],
    }


def atomic_write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create a read-only JSON file; never overwrite an existing lock."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖不可变资产环境锁: {path}")
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"拒绝覆盖不可变资产环境锁: {path}") from error
        temporary.unlink()
        path.chmod(0o444)
    finally:
        if temporary.exists():
            temporary.unlink()
