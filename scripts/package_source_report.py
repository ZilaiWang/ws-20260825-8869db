#!/usr/bin/env python3
"""构建不含权重、训练图和敏感文件的源码+报告 ZIP。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".engine",
    ".trt",
    ".npy",
    ".npz",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".tar",
    ".gz",
    ".pem",
    ".key",
}
FORBIDDEN_PARTS = {".git", "outputs", "artifacts", "cache", "weights", "checkpoints"}
ALLOWED_ROOTS = {
    "src",
    "scripts",
    "configs",
    "tests",
    "data",
    "submission",
}
ALLOWED_DOCUMENTS = {
    "docs/SUBMISSION_RUNBOOK_20260828.md",
    "reports/submission/README.md",
}
ALLOWED_TOP_FILES = {
    ".gitignore",
    "README.md",
    "NOTICE_FRFDET.md",
    "PR_DESCRIPTION.md",
    "SCOPE_RESULTS_INDEX.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-model.txt",
    "requirements-p03.txt",
    "requirements-mar20-grouping.txt",
}


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [REPO_ROOT / line for line in result.stdout.splitlines() if line]


def _admitted(path: Path) -> bool:
    relative = path.relative_to(REPO_ROOT)
    if any(part in FORBIDDEN_PARTS for part in relative.parts):
        return False
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return False
    if len(relative.parts) == 1:
        return relative.as_posix() in ALLOWED_TOP_FILES
    if relative.as_posix() in ALLOWED_DOCUMENTS:
        return path.is_file()
    if relative.parts[0] not in ALLOWED_ROOTS:
        return False
    if relative.parts[0] == "data" and relative.parts[1] not in {"splits", "groups"}:
        return path.name == "README.md"
    return path.is_file()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "dist" / "source-and-report.zip",
    )
    parser.add_argument("--allow-no-report", action="store_true")
    args = parser.parse_args()
    reports = [path.expanduser().resolve() for path in args.report]
    if not reports and not args.allow_no_report:
        raise ValueError("至少提供一份 --report；准备阶段可显式使用 --allow-no-report")
    for report in reports:
        if not report.is_file():
            raise FileNotFoundError(report)
        if report.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"报告文件类型不允许: {report}")

    files = sorted(path for path in _tracked_files() if _admitted(path))
    manifest = {
        "contract_version": "source_report_bundle_v1",
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_tree_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "contains_weights": False,
        "contains_training_images": False,
        "source_files": [
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
        "reports": [
            {"name": path.name, "size": path.stat().st_size, "sha256": _sha256(path)}
            for path in reports
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for path in files:
            zipf.write(path, f"source/{path.relative_to(REPO_ROOT).as_posix()}")
        for path in reports:
            zipf.write(path, f"report/{path.name}")
        zipf.writestr(
            "BUNDLE_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
    if args.output.stat().st_size > 500 * 1024 * 1024:
        raise RuntimeError("ZIP 超过平台 500MB 上限")
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(args.output.resolve()),
                "size": args.output.stat().st_size,
                "sha256": _sha256(args.output),
                "source_files": len(files),
                "reports": len(reports),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
