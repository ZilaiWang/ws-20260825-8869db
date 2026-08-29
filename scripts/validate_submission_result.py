#!/usr/bin/env python3
"""在 docker push 前严格校验官方 ``result.json``。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rsdet.submission.competition import validate_result_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    if not args.result.is_file():
        raise FileNotFoundError(args.result)
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    summary = validate_result_payload(payload)
    print(json.dumps({"status": "pass", **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
