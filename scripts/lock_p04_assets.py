#!/usr/bin/env python3
"""P04-TASK-00 权重/官方源码/SD snapshot 资产锁。"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from rsdet.features.p04_cache import sha256_file, stable_json_sha256
from rsdet.features.p04_teachers import (
    CLEANDIFT_COMMIT,
    CLEANDIFT_SD15_SHA256,
    CONVNEXT_TINY_SHA256,
    DINOV2_COMMIT,
)

SD15_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
DIFT_REFERENCE_COMMIT = "9421eb2034396c5b66f1aff37f03e540c264e52f"
DINO_S_SIZE = 88_283_115
DINO_B_SIZE = 346_378_731


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="冻结 P04 本地资产")
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def _commit(repo: Path, expected: str) -> str:
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != expected:
        raise ValueError(f"{repo} commit 不匹配: expected={expected}, actual={actual}")
    return actual


def _file(path: Path, *, expected_sha: str | None = None, expected_size: int | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise ValueError(f"{path} 大小不匹配: expected={expected_size}, actual={size}")
    sha = sha256_file(path)
    if expected_sha and sha != expected_sha:
        raise ValueError(f"{path} SHA-256 不匹配")
    return {"path": str(path), "size_bytes": size, "sha256": sha}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.asset_root).expanduser().resolve()
    weights = root / "weights"
    repos = root / "repos"
    sd15 = root / "models" / "stable-diffusion-v1-5"
    revision_file = sd15 / ".p04_hf_revision"
    if not revision_file.is_file() or revision_file.read_text(encoding="utf-8").strip() != SD15_REVISION:
        raise ValueError("SD1.5 snapshot revision 记录缺失或不匹配")
    required_sd_files = (
        "model_index.json",
        "unet/config.json",
        "vae/config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer_config.json",
        "scheduler/scheduler_config.json",
    )
    for relative in required_sd_files:
        if not (sd15 / relative).is_file():
            raise FileNotFoundError(f"SD1.5 snapshot 缺少 {relative}")
    if not (sd15 / "README.md").is_file():
        raise FileNotFoundError("SD1.5 snapshot 缺少 README.md")
    if not any(sd15.glob("LICENSE*")):
        raise FileNotFoundError("SD1.5 snapshot 缺少 LICENSE 文件")

    asset_files = {
        "convnext_tiny": _file(
            weights / "convnext_tiny-983f1562.pth", expected_sha=CONVNEXT_TINY_SHA256
        ),
        "dinov2_vits14": _file(
            weights / "dinov2_vits14_pretrain.pth", expected_size=DINO_S_SIZE
        ),
        "dinov2_vitb14": _file(
            weights / "dinov2_vitb14_pretrain.pth", expected_size=DINO_B_SIZE
        ),
        "cleandift_sd15": _file(
            weights / "cleandift_sd15_unet.safetensors",
            expected_sha=CLEANDIFT_SD15_SHA256,
        ),
    }
    sd_inventory: list[dict[str, Any]] = []
    for path in sorted(sd15.rglob("*")):
        relative = path.relative_to(sd15)
        if (
            path.is_file()
            and path.name != ".p04_hf_revision"
            and ".cache" not in relative.parts
        ):
            sd_inventory.append(
                {
                    "relative_path": relative.as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    if not sd_inventory:
        raise ValueError("SD1.5 snapshot inventory 为空")
    payload: dict[str, Any] = {
        "schema_version": "p04_asset_lock_v1",
        "asset_root": str(root),
        "files": asset_files,
        "repositories": {
            "dinov2": {
                "path": str(repos / "dinov2"),
                "commit": _commit(repos / "dinov2", DINOV2_COMMIT),
            },
            "cleandift": {
                "path": str(repos / "cleandift"),
                "commit": _commit(repos / "cleandift", CLEANDIFT_COMMIT),
            },
            "dift_reference": {
                "path": str(repos / "dift_reference"),
                "commit": _commit(repos / "dift_reference", DIFT_REFERENCE_COMMIT),
                "runtime_role": "reference_only_project_audited_port",
            },
        },
        "stable_diffusion_v15": {
            "path": str(sd15),
            "revision": SD15_REVISION,
            "file_count": len(sd_inventory),
            "total_size_bytes": sum(item["size_bytes"] for item in sd_inventory),
            "inventory": sd_inventory,
            "inventory_fingerprint": stable_json_sha256(
                {item["relative_path"]: item["sha256"] for item in sd_inventory}
            ),
        },
    }
    payload["asset_lock_fingerprint"] = stable_json_sha256(payload)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "pass", **payload}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
