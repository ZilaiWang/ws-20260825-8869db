#!/usr/bin/env python3
"""Materialize and optionally build the frozen P40+Hierarchy+Aircraft-D4 image."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOCKER_ROOT = ROOT / "submission" / "docker"
CONFIG = DOCKER_ROOT / "configs" / ("p40_hier_s256v128_t042_p40v060_iou070_aircraft_d4_v1.json")
ASSET_ROOT = ROOT / "outputs" / "HERA-GUARD-APEX-20260904" / ("P40-HIER-D4-DEPLOYMENT-ASSETS")
ASSET_MANIFEST = ASSET_ROOT / "manifest.json"
DEFAULT_OUTPUT = ROOT / "dist" / "p40-hier-d4-rescue-final"
DEFAULT_IMAGE = "xh-detector:p40-hier-d4-rescue-final"
DEFAULT_BASE_IMAGE = "xh-detector:p40-full-s1280-frozen0536-final"


def sha256(path: Path) -> str:
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


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _configured_assets(config: dict[str, Any]) -> dict[str, str]:
    entries = (
        (config["model"], "weight_path", "expected_sha256"),
        (
            config["resolution_expert_model"],
            "weight_path",
            "expected_sha256",
        ),
        (
            config["aircraft_classifier_model"],
            "weight_path",
            "expected_sha256",
        ),
        (
            config["aircraft_classifier_model"],
            "imagenet_weight_path",
            "imagenet_expected_sha256",
        ),
    )
    assets: dict[str, str] = {}
    for section, path_key, sha_key in entries:
        destination = Path(str(section[path_key]))
        if destination.parent != Path("/app/models"):
            raise ValueError(f"unexpected model destination: {destination}")
        assets[destination.name] = str(section[sha_key]).lower()
    if len(assets) != 4:
        raise ValueError(f"expected four distinct assets, got {sorted(assets)}")
    return assets


def materialize(output: Path, *, force: bool) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    asset_manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    if config["workpoint_id"] != asset_manifest["workpoint_id"]:
        raise ValueError("config and asset manifest workpoint_id differ")
    config_sha = sha256(CONFIG)
    if config_sha != str(asset_manifest["docker_config"]["sha256"]):
        raise ValueError("config SHA differs from frozen asset manifest")

    configured = _configured_assets(config)
    manifest_assets = {
        str(item["name"]): str(item["sha256"]).lower() for item in asset_manifest["assets"]
    }
    if configured != manifest_assets:
        raise ValueError("config asset set differs from frozen asset manifest")
    for name, expected in configured.items():
        source = ASSET_ROOT / name
        if not source.is_file():
            raise FileNotFoundError(source)
        actual = sha256(source)
        if actual != expected:
            raise ValueError(f"asset SHA mismatch: {name}: {actual} != {expected}")

    if output.exists():
        if not force:
            raise FileExistsError(f"output exists; pass --force: {output}")
        shutil.rmtree(output)
    (output / "app").mkdir(parents=True)
    (output / "models").mkdir()
    shutil.copy2(DOCKER_ROOT / "Dockerfile.multi-overlay", output / "Dockerfile")
    shutil.copy2(DOCKER_ROOT / ".dockerignore", output / ".dockerignore")
    shutil.copy2(DOCKER_ROOT / "app" / "main.py", output / "app" / "main.py")
    shutil.copy2(CONFIG, output / "app" / "config.json")
    _copy_tree(ROOT / "src" / "rsdet", output / "app" / "rsdet")
    for name in sorted(configured):
        shutil.copy2(ASSET_ROOT / name, output / "models" / name)

    candidate_manifest: dict[str, Any] = {
        "contract_version": "p40_hier_d4_submission_candidate_v1",
        "status": "materialized",
        "candidate_id": config["workpoint_id"],
        "deployment_role": config["deployment_role"],
        "config_sha256": config_sha,
        "asset_manifest_sha256": sha256(ASSET_MANIFEST),
        "assets": [
            {
                "path": f"/app/models/{name}",
                "size_bytes": (ASSET_ROOT / name).stat().st_size,
                "sha256": expected,
            }
            for name, expected in sorted(configured.items())
        ],
        "source_commit": _git("rev-parse", "HEAD"),
        "source_tree_dirty": bool(_git("status", "--porcelain")),
        "base_image": DEFAULT_BASE_IMAGE,
    }
    manifest_path = output / "CANDIDATE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(candidate_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    build_manifest = {
        **candidate_manifest,
        "files": [
            {
                "path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(item for item in output.rglob("*") if item.is_file())
        ],
    }
    (output / "BUILD_MANIFEST.json").write_text(
        json.dumps(build_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return candidate_manifest


def build_image(
    output: Path,
    manifest: dict[str, Any],
    *,
    image: str,
    base_image: str,
) -> None:
    subprocess.run(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "--build-arg",
            f"BASE_IMAGE={base_image}",
            "--build-arg",
            f"RSDET_CANDIDATE_ID={manifest['candidate_id']}",
            "--build-arg",
            f"RSDET_CONFIG_SHA256={manifest['config_sha256']}",
            "--build-arg",
            f"RSDET_ASSET_MANIFEST_SHA256={manifest['asset_manifest_sha256']}",
            "--tag",
            image,
            ".",
        ],
        cwd=output,
        check=True,
    )
    inspect = subprocess.run(
        ["docker", "image", "inspect", image],
        check=True,
        capture_output=True,
        text=True,
    )
    image_meta = json.loads(inspect.stdout)[0]
    if (image_meta["Os"], image_meta["Architecture"]) != ("linux", "amd64"):
        raise RuntimeError("built image is not linux/amd64")
    labels = image_meta["Config"].get("Labels") or {}
    if labels.get("rsdet.candidate.id") != manifest["candidate_id"]:
        raise RuntimeError("built image candidate label mismatch")
    (output / "IMAGE_MANIFEST.json").write_text(
        json.dumps(
            {
                "status": "built",
                "image": image,
                "image_id": image_meta["Id"],
                "os": image_meta["Os"],
                "architecture": image_meta["Architecture"],
                "candidate_id": manifest["candidate_id"],
                "config_sha256": manifest["config_sha256"],
                "asset_manifest_sha256": manifest["asset_manifest_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--base-image", default=DEFAULT_BASE_IMAGE)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--build", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    manifest = materialize(output, force=args.force)
    if args.build:
        build_image(
            output,
            manifest,
            image=str(args.image),
            base_image=str(args.base_image),
        )
    print(
        json.dumps(
            {
                "status": "built" if args.build else "materialized",
                "output": str(output),
                "image": str(args.image) if args.build else None,
                "candidate_id": manifest["candidate_id"],
                "config_sha256": manifest["config_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
