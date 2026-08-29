#!/usr/bin/env python3
"""Run the preregistered two-view pseudo-10K candidate-source screen.

The driver only orchestrates existing frozen inference and cross-fit analysis
scripts.  It never selects a checkpoint or threshold from a held-out fold.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(command: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def _write_status(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")


def _complete_json(
    path: Path,
    *,
    required_key: str,
    statuses: tuple[str, ...] = ("complete",),
    expected_threshold_stop: float | None = None,
) -> bool:
    if not path.is_file():
        return False
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if payload.get("status") not in statuses or required_key not in payload:
        return False
    if expected_threshold_stop is not None:
        actual = payload.get("threshold_grid", {}).get("stop")
        if actual != expected_threshold_stop:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, nargs=3, required=True)
    parser.add_argument("--natural-root", type=Path, required=True)
    parser.add_argument("--trial-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument(
        "--configs",
        type=Path,
        nargs="+",
        required=True,
        help="Two-view configs; each workpoint_id must be unique.",
    )
    args = parser.parse_args()

    roots = {"natural": args.natural_root, "trial": args.trial_root}
    for config in args.configs:
        import json

        payload = json.loads(config.read_text(encoding="utf-8"))
        workpoint = str(payload["workpoint_id"])
        views = payload["model"].get("rot90_views")
        if not isinstance(views, list) or len(views) != 2 or 0 not in views:
            raise ValueError(f"{config}: expected exactly two views including identity")
        for profile, pseudo_root in roots.items():
            run_dir = args.output_root / workpoint / profile
            run_summary = run_dir / "run_summary.json"
            predictions = run_dir / "predictions.json"
            if not (
                predictions.is_file()
                and _complete_json(
                    run_summary,
                    required_key="folds",
                    statuses=("cv3_oof_pseudo_inference_complete",),
                )
            ):
                _write_status(args.status_file, f"inference:{workpoint}:{profile}")
                _run(
                    [
                        str(args.python),
                        "scripts/run_cv3_oof_pseudo_eval.py",
                        "--pseudo-root",
                        str(pseudo_root),
                        "--config",
                        str(config),
                        "--weights",
                        *(str(path) for path in args.weights),
                        "--output-dir",
                        str(run_dir),
                    ],
                    cwd=args.workdir,
                    log_path=run_dir / "inference.log",
                )
            analyses = {
                "frontier": "scripts/analyze_cv3_oof_pseudo_frontier.py",
                "coarse": "scripts/analyze_cv3_oof_pseudo_coarse_thresholds.py",
            }
            for analysis, script in analyses.items():
                analysis_output = run_dir / f"{analysis}.json"
                if _complete_json(
                    analysis_output,
                    required_key="protocol",
                    expected_threshold_stop=0.996,
                ):
                    continue
                _write_status(
                    args.status_file, f"analysis:{workpoint}:{profile}:{analysis}"
                )
                _run(
                    [
                        str(args.python),
                        script,
                        "--gt",
                        str(pseudo_root / "ground_truth.json"),
                        "--pred",
                        str(predictions),
                        "--output",
                        str(analysis_output),
                    ],
                    cwd=args.workdir,
                    log_path=run_dir / f"{analysis}.log",
                )

    _write_status(args.status_file, "complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
