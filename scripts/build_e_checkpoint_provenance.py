#!/usr/bin/env python3
"""生成 E 正式测速的 checkpoint provenance（checkpoint_provenance_v1）。

benchmark_10k_pipeline.py 的 _validate_checkpoint_provenance 要求单折语义：

  {"contract_version": "checkpoint_provenance_v1",
   "status": "checkpoint_lineage_verified",
   "engineering_checkpoint_only": false,
   "model_key": "M1",
   "checkpoint": "<path, 相对本文件或绝对>",
   "checkpoint_sha256": "<64 hex>",
   "fold_metadata": "<path>",
   "fold_metadata_sha256": "<64 hex>",
   "oof_metadata": "<path>",
   "oof_metadata_sha256": "<64 hex>"}

本脚本用已找回并验证的 M1 正式三折资产（fold 0 last.pt + fold_metadata +
oof_metadata，SHA 与权威报告逐位匹配）构造该文件，并本地自校验
（checkpoint 实体 SHA、fold_metadata/oof_metadata 的 checkpoint_sha256 交叉
核对）。纯 CPU。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_relative(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def build_provenance(
    *,
    checkpoint: Path,
    fold_metadata: Path,
    oof_metadata: Path,
    model_key: str,
    fold: int,
) -> dict[str, Any]:
    """校验三资产 SHA 与交叉血缘，构造 checkpoint_provenance_v1 文档。

    资产路径写入时转为相对 provenance 输出目录的形式（与 benchmark 的
    _resolve_config_path 一致）。
    """
    checkpoint = checkpoint.expanduser().resolve()
    fold_metadata = fold_metadata.expanduser().resolve()
    oof_metadata = oof_metadata.expanduser().resolve()
    for path, label in (
        (checkpoint, "checkpoint"),
        (fold_metadata, "fold_metadata"),
        (oof_metadata, "oof_metadata"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} 不存在: {path}")

    ckpt_sha = _sha256(checkpoint)
    fold_meta = json.loads(fold_metadata.read_text(encoding="utf-8"))
    oof_meta = json.loads(oof_metadata.read_text(encoding="utf-8"))

    # 交叉血缘：fold_metadata.artifacts.checkpoint_sha256 == oof_metadata.folds[fold]
    fold_row = next(r for r in oof_meta["folds"] if int(r["fold"]) == fold)
    fold_meta_sha = fold_meta["artifacts"]["checkpoint_sha256"]
    oof_sha = fold_row["checkpoint_sha256"]
    if ckpt_sha != fold_meta_sha or ckpt_sha != oof_sha:
        raise ValueError(
            f"checkpoint SHA 与血缘不一致: 实体={ckpt_sha[:12]} "
            f"fold_metadata={fold_meta_sha[:12]} oof_metadata={oof_sha[:12]}"
        )
    if str(fold_meta.get("model_key", "")).strip().upper() != model_key.upper():
        raise ValueError(f"fold_metadata model_key 不一致: {fold_meta.get('model_key')}")
    if str(oof_meta.get("model_key", "")).strip().upper() != model_key.upper():
        raise ValueError(f"oof_metadata model_key 不一致: {oof_meta.get('model_key')}")
    if oof_meta.get("status") != "complete_downstream_ready" or oof_meta.get("downstream_admission") is not True:
        raise ValueError("oof_metadata 未达 downstream 交付状态")
    if fold_meta.get("status") != "fold_delivery_complete":
        raise ValueError(f"fold_metadata 未达交付状态: {fold_meta.get('status')}")

    return {
        "contract_version": "checkpoint_provenance_v1",
        "status": "checkpoint_lineage_verified",
        "engineering_checkpoint_only": False,
        "model_key": model_key,
        "fold": fold,
        "checkpoint": None,  # 占位，输出时转相对路径
        "checkpoint_sha256": ckpt_sha,
        "fold_metadata": None,
        "fold_metadata_sha256": _sha256(fold_metadata),
        "oof_metadata": None,
        "oof_metadata_sha256": _sha256(oof_metadata),
    }


def _relativize(base: Path, value: str) -> str:
    path = Path(value).expanduser().resolve()
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold-metadata", type=Path, required=True)
    parser.add_argument("--oof-metadata", type=Path, required=True)
    parser.add_argument("--model-key", default="M1")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = build_provenance(
        checkpoint=args.checkpoint,
        fold_metadata=args.fold_metadata,
        oof_metadata=args.oof_metadata,
        model_key=args.model_key,
        fold=args.fold,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    base = output.parent
    payload["checkpoint"] = _relativize(base, str(args.checkpoint))
    payload["fold_metadata"] = _relativize(base, str(args.fold_metadata))
    payload["oof_metadata"] = _relativize(base, str(args.oof_metadata))
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"PROVENANCE_SHA256={_sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
