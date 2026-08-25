#!/usr/bin/env python3
"""Build an FP_BG evidence manifest from a frozen final-chain prediction file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from rsdet.analysis.oof_detection import decompose_official_errors, load_formal_ground_truth
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(path: Path, expected: str, label: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA mismatch: {actual} != {expected}")
    return actual


def _group_predictions(path: Path, image_ids: set[int]) -> dict[int, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("selected_predictions 必须为列表")
    result = {image_id: [] for image_id in sorted(image_ids)}
    for row in payload:
        image_id = int(row["image_id"])
        if image_id not in result:
            raise ValueError(f"预测 image_id 不在正式 ledger: {image_id}")
        item = {key: value for key, value in row.items() if key != "image_id"}
        result[image_id].append(item)
    return result


def _size_bin(box: list[float]) -> str:
    short_edge = min(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))
    if short_edge < 16:
        return "tiny_lt16"
    if short_edge < 32:
        return "small_16_32"
    if short_edge < 64:
        return "medium_32_64"
    return "large_ge64"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/n0_fp_bg_review_r1_6_v3.yaml"),
    )
    value.add_argument("--output", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    if config.get("contract_version") != "n0_fp_bg_review_final_chain_v1":
        raise ValueError("unexpected final-chain FP_BG contract")
    root = Path.cwd().resolve()
    inputs = config["inputs"]
    paths = {
        key: (root / inputs[key]).resolve()
        for key in ("selected_predictions", "formal_crop_manifest", "image_ledger")
    }
    input_sha = {
        key: _require_sha(paths[key], inputs[f"{key}_sha256"], key)
        for key in paths
    }
    with paths["image_ledger"].open("r", encoding="utf-8-sig", newline="") as handle:
        image_rows = {int(row["image_id"]): row for row in csv.DictReader(handle)}
    if len(image_rows) != int(inputs["expected_images"]):
        raise ValueError("image ledger count mismatch")
    predictions = _group_predictions(paths["selected_predictions"], set(image_rows))
    protocol = parse_evaluation_protocol(load_config(config["project_config"]))
    formal = load_formal_ground_truth(
        paths["formal_crop_manifest"],
        expected_sha256=inputs["formal_crop_manifest_sha256"],
        expected_images=int(inputs["expected_images"]),
        expected_annotations=20933,
    )
    summary, cases, _ = decompose_official_errors(
        formal,
        predictions,
        threshold=0.0,
        protocol=protocol,
        model_key="R1_6_FINAL_CHAIN",
        include_cases=True,
    )
    fp_bg_cases = [
        row
        for row in cases
        if row["case_side"] == "prediction" and row["reason"] == "FP_BG"
    ]
    if len(fp_bg_cases) != int(inputs["expected_fp_bg"]):
        raise ValueError(f"FP_BG count mismatch: {len(fp_bg_cases)}")
    records: list[dict[str, Any]] = []
    for case in fp_bg_cases:
        image_id = int(case["image_id"])
        # ``item_uid`` index is the exact position in the frozen per-image list.
        prediction_index = int(str(case["item_uid"]).rsplit("-p", 1)[1])
        prediction = predictions[image_id][prediction_index]
        box = [float(value) for value in prediction["bbox_xyxy"]]
        image = image_rows[image_id]
        records.append(
            {
                "proposal_uid": str(prediction.get("proposal_uid", case["item_uid"])),
                "image_id": image_id,
                "fold": int(image["fold"]),
                "source_group": image["group_id"],
                "category_id": int(prediction["category_id"]),
                "class_name": protocol.category_mapping[int(prediction["category_id"])],
                "score": float(prediction["score"]),
                "bbox_xyxy": box,
                "size_bin": _size_bin(box),
                "official_status": "FP_BG",
            }
        )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_version": "final_chain_fp_bg_evidence_v1",
        "status": "pass",
        "experiment_id": config["experiment_id"],
        "input_sha256": input_sha,
        "final_chain": config["review"]["final_chain"],
        "error_decomposition": summary,
        "record_count": len(records),
        "records": records,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"FINAL_CHAIN_FP_BG_MANIFEST_PASS records={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
