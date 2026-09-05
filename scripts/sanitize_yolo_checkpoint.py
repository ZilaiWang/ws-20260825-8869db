#!/usr/bin/env python3
"""把训练 checkpoint 收缩为不依赖增强库的可部署 Ultralytics 权重。"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import torch

from rsdet.data.xh_dataset import FINE_NAMES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, help="Inference metadata; defaults to the source size")
    parser.add_argument("--require-tensor-identical", action="store_true")
    parser.add_argument(
        "--allow-index-names",
        action="store_true",
        help="Accept only the exact 0..24 string-name mapping and normalize it to FINE_NAMES",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite deployment checkpoint: {args.output}")
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or (
        checkpoint.get("model") is None and checkpoint.get("ema") is None
    ):
        raise ValueError("输入不是 Ultralytics checkpoint")
    model = checkpoint.get("ema") or checkpoint["model"]
    names = {index: name for index, name in enumerate(FINE_NAMES)}
    source_names = dict(model.names)
    index_names = {index: str(index) for index in range(len(FINE_NAMES))}
    normalized_index_names = bool(args.allow_index_names and source_names == index_names)
    if source_names != names and not normalized_index_names:
        raise ValueError("source checkpoint has the wrong class mapping")
    original_state = ({k: v.detach().clone() for k, v in model.state_dict().items()}
                      if args.require_tensor_identical else None)
    imgsz = args.imgsz or int(checkpoint.get("train_args", {}).get("imgsz", 1024))
    if imgsz <= 0:
        raise ValueError("imgsz must be positive")
    model.names = names
    if hasattr(model, "yaml") and isinstance(model.yaml, dict):
        model.yaml["nc"] = len(names)
        model.yaml["names"] = names
    model.args = {"task": "detect", "imgsz": imgsz, "single_cls": False}
    if hasattr(model, "criterion"):
        model.criterion = None
    model.half().eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    try:
        from ultralytics import __version__ as ultralytics_version
    except ImportError:
        ultralytics_version = "unknown"
    deploy = {
        "date": dt.datetime.now(dt.timezone.utc).isoformat(),
        "version": ultralytics_version,
        "license": checkpoint.get("license", "AGPL-3.0"),
        "docs": checkpoint.get("docs", "https://docs.ultralytics.com"),
        "epoch": -1,
        "best_fitness": None,
        "model": model,
        "ema": None,
        "updates": None,
        "optimizer": None,
        "scaler": None,
        "train_args": {"task": "detect", "imgsz": imgsz, "single_cls": False},
        "train_metrics": {},
        "train_results": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(deploy, args.output)
    tensor_identity = None
    if original_state is not None:
        restored = torch.load(args.output, map_location="cpu", weights_only=False)["model"].state_dict()
        tensor_identity = (restored.keys() == original_state.keys() and all(
            restored[k].dtype == original_state[k].dtype
            and torch.equal(restored[k], original_state[k]) for k in original_state
        ))
        if not tensor_identity:
            raise RuntimeError("sanitization changed model tensors; do not deploy this output")
    payload = {
        "status": "sanitized",
        "source": str(args.input.resolve()),
        "source_sha256": _sha256(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "output_size": args.output.stat().st_size,
        "fine_names": list(FINE_NAMES),
        "source_names": source_names,
        "normalized_exact_index_names": normalized_index_names,
        "removed_training_state": True,
        "imgsz": imgsz,
        "all_model_tensors_bitwise_identical": tensor_identity,
        "tensor_count": len(model.state_dict()),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
