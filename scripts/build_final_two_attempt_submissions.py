#!/usr/bin/env python3
"""Materialize the frozen Attempt 4 or Attempt 5 Docker build context."""

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
ASSET_ROOT = ROOT / "outputs" / "HERA-GUARD-APEX-20260904" / "P40-HIER-D4-DEPLOYMENT-ASSETS"
BASE_IMAGE = "xh-detector:p40-full-s1280-frozen0536-final"
CANDIDATES = {
    "attempt4": DOCKER_ROOT / "configs" / "p40_aircraft_d4_shared_otm_ship23_t0560_v1.json",
    "attempt5": DOCKER_ROOT / "configs" / "p40_aircraft_d4_only_v1.json",
}


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


def _configured_assets(config: dict[str, Any]) -> dict[str, str]:
    assets: dict[str, str] = {}
    for section_name in ("model", "aircraft_classifier_model"):
        section = config[section_name]
        destination = Path(str(section["weight_path"]))
        if destination.parent != Path("/app/models"):
            raise ValueError(f"unexpected model destination: {destination}")
        assets[destination.name] = str(section["expected_sha256"]).lower()
    return assets


def materialize(candidate: str, output: Path, *, force: bool) -> dict[str, Any]:
    config_path = CANDIDATES[candidate]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assets = _configured_assets(config)
    expected_names = {"p40_official_sanitized.pt", "aircraft_d4_full_checkpoint.pt"}
    if set(assets) != expected_names:
        raise ValueError(f"unexpected asset set: {sorted(assets)}")
    for name, expected in assets.items():
        source = ASSET_ROOT / name
        if not source.is_file():
            raise FileNotFoundError(source)
        if sha256(source) != expected:
            raise ValueError(f"asset SHA mismatch: {name}")

    if output.exists():
        if not force:
            raise FileExistsError(f"output exists; pass --force: {output}")
        shutil.rmtree(output)
    (output / "app").mkdir(parents=True)
    (output / "models").mkdir()
    shutil.copy2(DOCKER_ROOT / "Dockerfile.multi-overlay", output / "Dockerfile")
    shutil.copy2(DOCKER_ROOT / ".dockerignore", output / ".dockerignore")
    shutil.copy2(DOCKER_ROOT / "app" / "main.py", output / "app" / "main.py")
    shutil.copy2(config_path, output / "app" / "config.json")
    _copy_tree(ROOT / "src" / "rsdet", output / "app" / "rsdet")
    if config.get("sprint20"):
        _copy_tree(ROOT / "src" / "sprint20", output / "app" / "sprint20")
    for name in sorted(assets):
        shutil.copy2(ASSET_ROOT / name, output / "models" / name)

    manifest = {
        "contract_version": "final_two_attempt_submission_candidate_v1",
        "status": "materialized",
        "candidate": candidate,
        "candidate_id": config["workpoint_id"],
        "deployment_role": config["deployment_role"],
        "config_sha256": sha256(config_path),
        "base_image": BASE_IMAGE,
        "assets": [
            {
                "path": f"/app/models/{name}",
                "size_bytes": (ASSET_ROOT / name).stat().st_size,
                "sha256": expected,
            }
            for name, expected in sorted(assets.items())
        ],
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "source_tree_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
    }
    (output / "CANDIDATE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build(output: Path, manifest: dict[str, Any], image: str) -> None:
    subprocess.run(
        [
            "docker", "build", "--platform", "linux/amd64",
            "--build-arg", f"BASE_IMAGE={BASE_IMAGE}",
            "--build-arg", f"RSDET_CANDIDATE_ID={manifest['candidate_id']}",
            "--build-arg", f"RSDET_CONFIG_SHA256={manifest['config_sha256']}",
            "--build-arg", "RSDET_ASSET_MANIFEST_SHA256=not_applicable_two_asset_candidate",
            "--tag", image, ".",
        ],
        cwd=output,
        check=True,
    )
    inspected = subprocess.run(
        ["docker", "image", "inspect", image],
        check=True,
        capture_output=True,
        text=True,
    )
    image_meta = json.loads(inspected.stdout)[0]
    if (image_meta["Os"], image_meta["Architecture"]) != ("linux", "amd64"):
        raise RuntimeError("built image is not linux/amd64")
    labels = image_meta["Config"].get("Labels") or {}
    if labels.get("rsdet.candidate.id") != manifest["candidate_id"]:
        raise RuntimeError("built image candidate label mismatch")
    if labels.get("rsdet.config.sha256") != manifest["config_sha256"]:
        raise RuntimeError("built image config label mismatch")
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
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    output = (args.output or ROOT / "dist" / f"hera-{args.candidate}-20260905").resolve()
    image = args.image or f"xh-detector:hera-{args.candidate}-20260905"
    manifest = materialize(args.candidate, output, force=args.force)
    if args.build:
        build(output, manifest, image)
    print(
        json.dumps(
            {
                **manifest,
                "status": "built" if args.build else "materialized",
                "output": str(output),
                "image": image,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
