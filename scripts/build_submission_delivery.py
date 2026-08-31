#!/usr/bin/env python3
"""生成可直接 ``docker build .`` 的赛事交付目录。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = REPO_ROOT / "submission" / "docker"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", ".git"),
    )


def build_delivery(
    *,
    weights: Path,
    output: Path,
    expected_sha256: str,
    environment: Path,
    config: Path,
    force: bool,
    agreement_weights: Path | None = None,
    agreement_expected_sha256: str | None = None,
    agreement_root: Path | None = None,
    agreement_config: Path | None = None,
) -> dict[str, object]:
    weights = weights.expanduser().resolve()
    environment = environment.expanduser().resolve()
    config_source = config.expanduser().resolve()
    output = output.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not environment.is_file():
        raise FileNotFoundError(environment)
    if not config_source.is_file():
        raise FileNotFoundError(config_source)
    agreement_values = (
        agreement_weights,
        agreement_expected_sha256,
        agreement_root,
        agreement_config,
    )
    agreement_enabled = any(value is not None for value in agreement_values)
    if agreement_enabled != all(value is not None for value in agreement_values):
        raise ValueError("agreement delivery arguments must be provided together")
    agreement_actual_sha: str | None = None
    if agreement_enabled:
        agreement_weights = agreement_weights.expanduser().resolve()  # type: ignore[union-attr]
        agreement_root = agreement_root.expanduser().resolve()  # type: ignore[union-attr]
        agreement_config = agreement_config.expanduser().resolve()  # type: ignore[union-attr]
        if not agreement_weights.is_file():
            raise FileNotFoundError(agreement_weights)
        if not agreement_root.is_dir():
            raise FileNotFoundError(agreement_root)
        if not agreement_config.is_file():
            raise FileNotFoundError(agreement_config)
        agreement_actual_sha = sha256_file(agreement_weights)
        if agreement_actual_sha != str(agreement_expected_sha256).lower():
            raise ValueError(
                "agreement weight SHA256 mismatch: "
                f"expected={str(agreement_expected_sha256).lower()} "
                f"actual={agreement_actual_sha}"
            )
    actual_sha = sha256_file(weights)
    if actual_sha != expected_sha256.lower():
        raise ValueError(
            f"weight SHA256 mismatch: expected={expected_sha256.lower()} actual={actual_sha}"
        )
    if output.exists():
        if not force:
            raise FileExistsError(f"输出目录已存在；加 --force 才可重建: {output}")
        shutil.rmtree(output)

    (output / "app").mkdir(parents=True)
    (output / "models").mkdir()
    shutil.copy2(TEMPLATE_ROOT / "Dockerfile", output / "Dockerfile")
    shutil.copy2(TEMPLATE_ROOT / ".dockerignore", output / ".dockerignore")
    shutil.copy2(environment, output / "environment.yml")
    shutil.copy2(TEMPLATE_ROOT / "app" / "main.py", output / "app" / "main.py")
    shutil.copy2(config_source, output / "app" / "config.json")
    _copy_tree(REPO_ROOT / "src" / "rsdet", output / "app" / "rsdet")
    shutil.copy2(weights, output / "models" / "model.pt")
    if agreement_enabled:
        (output / "vendor").mkdir()
        _copy_tree(agreement_root, output / "vendor" / "dfine")  # type: ignore[arg-type]
        shutil.copy2(agreement_weights, output / "models" / "dfine.pth")
        shutil.copy2(agreement_config, output / "models" / "dfine.yml")
        with (output / "Dockerfile").open("a", encoding="utf-8") as handle:
            handle.write("\nCOPY vendor /app/vendor\n")

    config_path = output / "app" / "config.json"
    materialized_config = json.loads(config_path.read_text(encoding="utf-8"))
    configured_sha = str(materialized_config["model"]["expected_sha256"]).lower()
    if configured_sha != actual_sha:
        raise ValueError(
            "submission/docker/config.json 的 model.expected_sha256 与所选权重不一致；"
            "先冻结配置再构建"
        )
    if agreement_enabled:
        agreement_section = materialized_config.get("agreement_model")
        if not isinstance(agreement_section, dict):
            raise ValueError("dual delivery config lacks agreement_model")
        agreement_section.update(
            {
                "root_path": "/app/vendor/dfine",
                "config_path": "/app/models/dfine.yml",
                "weight_path": "/app/models/dfine.pth",
                "expected_sha256": agreement_actual_sha,
            }
        )
        config_path.write_text(
            json.dumps(materialized_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    included: list[dict[str, object]] = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        included.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest: dict[str, object] = {
        "contract_version": "docker_delivery_manifest_v1",
        "status": "materialized",
        "source_commit": _git_commit(),
        "source_tree_dirty": _git_dirty(),
        "weight_source": str(weights),
        "weight_destination": "models/model.pt",
        "weight_sha256": actual_sha,
        "environment_source": str(environment),
        "config_source": str(config_source),
        "files": included,
    }
    if agreement_enabled:
        manifest["agreement"] = {
            "weight_source": str(agreement_weights),
            "weight_destination": "models/dfine.pth",
            "weight_sha256": agreement_actual_sha,
            "source_root": str(agreement_root),
            "config_source": str(agreement_config),
            "config_destination": "models/dfine.yml",
        }
    manifest_path = output / "BUILD_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    mode = (output / "app" / "main.py").stat().st_mode
    (output / "app" / "main.py").chmod(mode | stat.S_IXUSR)
    return manifest


def _git_commit() -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_dirty() -> bool:
    import subprocess

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument(
        "--environment",
        type=Path,
        default=TEMPLATE_ROOT / "environment.yml",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=TEMPLATE_ROOT / "config.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "dist" / "detector-docker-delivery",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--agreement-weights", type=Path)
    parser.add_argument("--agreement-expected-sha256")
    parser.add_argument("--agreement-root", type=Path)
    parser.add_argument("--agreement-config", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_delivery(
        weights=args.weights,
        output=args.output,
        expected_sha256=args.expected_sha256,
        environment=args.environment,
        config=args.config,
        force=args.force,
        agreement_weights=args.agreement_weights,
        agreement_expected_sha256=args.agreement_expected_sha256,
        agreement_root=args.agreement_root,
        agreement_config=args.agreement_config,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(args.output.expanduser().resolve()),
                "weight_sha256": manifest["weight_sha256"],
                "files": len(manifest["files"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
