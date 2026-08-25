#!/usr/bin/env python3
"""MG00/MG01 服务器环境、DINOv2 资产和确定性门禁。"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from rsdet.grouping.contracts import PROTOCOL_VERSION, atomic_write_json, sha256_file
from rsdet.grouping.descriptors import DINOV2_COMMIT

EXPECTED_PACKAGES = {
    "numpy": "1.26.4",
    "Pillow": "10.4.0",
    "PyYAML": "6.0.2",
    "scipy": "1.14.1",
    "scikit-learn": "1.5.2",
    "opencv-python-headless": "4.10.0.84",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAR20 grouping environment gate")
    parser.add_argument("--asset-lock", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-gpu")
    return parser.parse_args(argv)


def _git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    failures = []
    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {},
    }
    if sys.version_info[:3] != (3, 10, 12):
        failures.append(f"Python expected=3.10.12, actual={sys.version_info[:3]}")
    for name, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        report["packages"][name] = {"expected": expected, "actual": actual}
        if actual != expected:
            failures.append(f"{name} expected={expected}, actual={actual}")
    try:
        import torch
        import torchvision

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        report["torch"] = {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "gpu": gpu,
        }
        if torch.__version__ != "2.5.1+cu121":
            failures.append(f"torch expected=2.5.1+cu121, actual={torch.__version__}")
        if torchvision.__version__ != "0.20.1+cu121":
            failures.append(f"torchvision expected=0.20.1+cu121, actual={torchvision.__version__}")
        if torch.version.cuda != "12.1":
            failures.append(f"CUDA runtime expected=12.1, actual={torch.version.cuda}")
        if not torch.cuda.is_available():
            failures.append("CUDA 不可用")
        if args.expected_gpu and gpu != args.expected_gpu:
            failures.append(f"GPU expected={args.expected_gpu}, actual={gpu}")
    except ImportError as error:
        failures.append(f"PyTorch import 失败: {error}")

    lock_path = Path(args.asset_lock).expanduser().resolve()
    if not lock_path.is_file():
        failures.append(f"asset lock 不存在: {lock_path}")
        lock: dict[str, Any] = {}
    else:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    dino_file = lock.get("files", {}).get("dinov2_vitb14", {})
    dino_repo = lock.get("repositories", {}).get("dinov2", {})
    asset_report = {
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path) if lock_path.is_file() else None,
        "dinov2_vitb14": dino_file,
        "dinov2_repo": dino_repo,
    }
    if not dino_file or not dino_repo:
        failures.append("asset lock 缺少 dinov2_vitb14 或 dinov2 repo")
    else:
        weight_path = Path(dino_file["path"])
        if not weight_path.is_file():
            failures.append(f"DINOv2-B 权重不存在: {weight_path}")
        elif sha256_file(weight_path) != dino_file["sha256"]:
            failures.append("DINOv2-B 权重 SHA 改变")
        repo_path = Path(dino_repo["path"])
        try:
            actual_commit = _git_commit(repo_path)
        except (OSError, subprocess.CalledProcessError):
            actual_commit = None
        asset_report["dinov2_repo_actual_commit"] = actual_commit
        if actual_commit != dino_repo["commit"] or actual_commit != DINOV2_COMMIT:
            failures.append(
                "DINOv2 repo commit 不匹配: "
                f"lock={dino_repo.get('commit')}, code={DINOV2_COMMIT}, actual={actual_commit}"
            )
    report["assets"] = asset_report
    report["status"] = "pass" if not failures else "fail"
    report["failures"] = failures
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
