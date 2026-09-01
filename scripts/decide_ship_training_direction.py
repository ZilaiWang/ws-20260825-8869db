#!/usr/bin/env python3
"""Compile reviewed/diagnostic Ship errors into one exclusive training route."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.error_route import decide_ship_training_direction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-errors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.metrics_errors.read_text(encoding="utf-8"))
    protocol = payload.get("metric_protocol")
    if protocol != "platform_observed_20260831":
        raise ValueError(
            "Ship route selection requires metric_protocol="
            "platform_observed_20260831"
        )
    if "error_decomposition_at_0p15" in payload:
        ship = payload["error_decomposition_at_0p15"]["ship"]
    else:
        ship = payload["per_class"]["ship"]
    decision = decide_ship_training_direction(
        {key: int(ship.get(key, 0)) for key in ("FP_BG", "FP_CLS", "FP_LOC", "FP_DUP")},
        {key: int(ship.get(key, 0)) for key in ("FN_MISS", "FN_CLS", "FN_LOC")},
    )
    result = {
        "version": "ship_training_direction_v1",
        "metric_protocol": protocol,
        "source": str(args.metrics_errors),
        "decision": decision.__dict__,
        "direction_selection_only": True,
        "formal_module_admission": False,
        "exclusive_route_enforced": True,
        "fine_tail_and_objectness_must_not_be_combined": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
