#!/usr/bin/env python3
"""校验找回的 M1 三折 last.pt 的 lineage（E 任务单第 0/6 节血缘门禁）。

对每个 fold，验证三处 SHA256 一致：
  - 本地 last.pt 文件内容；
  - fold_metadata.json 的 artifacts.checkpoint_sha256；
  - oof_metadata.json 的 folds[i].checkpoint_sha256。

同时复核 fold_metadata / oof_metadata 的交付状态字段，产出
checkpoint_provenance.json（E 任务单第 8 节交付物）。纯 CPU。

此前 last.pt 状态为 pending_source_retrieval（还在已释放的 M1 服务器上），
E 正式时延结论只能退而用工程版 best.pt（engineering_checkpoint_only=true）。
本脚本在 last.pt 找回后，将 lineage 校验固化为可复现证据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lastpt-dir",
        type=Path,
        default=Path("artifacts/M1-CV3-OOF/last.pt"),
        help="三折 last.pt 所在目录（fold0_last.pt / fold1_last.pt / fold2_last.pt）",
    )
    parser.add_argument(
        "--fold-metadata-dir",
        type=Path,
        default=Path("outputs/M1-CV3-OOF"),
        help="fold_0/1/2/fold_metadata.json 所在目录",
    )
    parser.add_argument(
        "--oof-metadata",
        type=Path,
        default=Path(
            "outputs/M1-CV3-OOF-return-no-checkpoints-extracted-20260725/"
            "M1-CV3-OOF-aggregate/oof_metadata.json"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/checkpoint_provenance.json"))
    args = parser.parse_args(argv)

    oof_metadata = json.loads(args.oof_metadata.read_text(encoding="utf-8"))
    assert oof_metadata.get("status") == "complete_downstream_ready", oof_metadata.get("status")
    assert oof_metadata.get("downstream_admission") is True
    assert oof_metadata.get("model_key") == "M1"

    folds: list[dict[str, Any]] = []
    for fold in (0, 1, 2):
        lastpt = args.lastpt_dir / f"fold{fold}_last.pt"
        fold_meta_path = args.fold_metadata_dir / f"fold_{fold}" / "fold_metadata.json"
        if not lastpt.is_file():
            raise FileNotFoundError(f"last.pt 不存在: {lastpt}")
        if not fold_meta_path.is_file():
            raise FileNotFoundError(f"fold_metadata 不存在: {fold_meta_path}")

        fold_meta = json.loads(fold_meta_path.read_text(encoding="utf-8"))
        assert fold_meta.get("status") == "fold_delivery_complete"
        assert fold_meta.get("model_key") == "M1"
        assert int(fold_meta.get("held_out_fold")) == fold

        lastpt_sha = sha256(lastpt)
        fold_meta_sha = fold_meta["artifacts"]["checkpoint_sha256"]
        oof_row = next(r for r in oof_metadata["folds"] if int(r["fold"]) == fold)
        oof_sha = oof_row["checkpoint_sha256"]

        if not (lastpt_sha == fold_meta_sha == oof_sha):
            raise ValueError(
                f"fold{fold} lineage 不一致: last.pt={lastpt_sha} "
                f"fold_metadata={fold_meta_sha} oof_metadata={oof_sha}"
            )
        folds.append(
            {
                "fold": fold,
                "checkpoint_sha256": lastpt_sha,
                "checkpoint_size_bytes": lastpt.stat().st_size,
                "fold_metadata_sha256": sha256(fold_meta_path),
                "held_out_fold": fold,
            }
        )
        print(f"fold{fold} lineage OK: {lastpt_sha[:16]}...")

    provenance: dict[str, Any] = {
        "contract_version": "checkpoint_provenance_v1",
        "status": "checkpoint_lineage_verified",
        "engineering_checkpoint_only": False,  # 正式 160-epoch last.pt，非工程 best.pt
        "model_key": "M1",
        "folds": folds,
        "oof_metadata": str(args.oof_metadata.resolve()),
        "oof_metadata_sha256": sha256(args.oof_metadata),
    }
    args.output.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nlineage 校验通过，provenance 写入: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
