#!/usr/bin/env python3
"""Container entrypoint for the stable and optional Sprint20 runtimes."""

from __future__ import annotations

import os
from pathlib import Path

from rsdet.submission.competition import (
    DEFAULT_CONFIG_PATH,
    CompetitionDetector,
    load_submission_config,
    parse_args,
    run_submission,
)


def main() -> int:
    args = parse_args()
    config_path = Path(os.environ.get("RSDET_SUBMISSION_CONFIG", str(DEFAULT_CONFIG_PATH)))
    config = load_submission_config(config_path)
    factory = CompetitionDetector
    if config.get("sprint20"):
        from sprint20.runtime import detector_factory

        factory = detector_factory
    run_submission(args.input, args.output, config_path, detector_factory=factory)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
