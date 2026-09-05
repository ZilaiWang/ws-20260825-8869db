"""Use only after local parity/accuracy gates; this does not submit to a website."""

import argparse
from pathlib import Path

from .runtime import detector_factory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from rsdet.submission.competition import run_submission

    if (args.output / "result.json").exists():
        raise FileExistsError("Refusing to overwrite existing result.json")
    run_submission(args.input, args.output, args.config, detector_factory=detector_factory)


if __name__ == "__main__":
    main()
