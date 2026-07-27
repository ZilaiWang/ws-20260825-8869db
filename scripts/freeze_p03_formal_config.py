#!/usr/bin/env python3
"""把 P03 formal 配置模板绑定到已通过门禁的 crop manifest SHA。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import yaml

from rsdet.analysis.formal_replay import (
    CV3_SHA256,
    FORMAL_CROP_VERSION,
    sha256_file,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="freeze P03 formal replay config")
    parser.add_argument("--template", required=True)
    parser.add_argument("--input-audit", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit_path = Path(args.input_audit).expanduser().resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "formal_replay_inputs_ready":
        raise ValueError("formal input audit 未通过")
    if audit.get("cv3_manifest_sha256") != CV3_SHA256:
        raise ValueError("formal input audit 未绑定冻结 CV3 v2")
    if audit.get("formal_manifest_version") != FORMAL_CROP_VERSION:
        raise ValueError("formal crop manifest version 不匹配")
    config = yaml.safe_load(Path(args.template).read_text(encoding="utf-8"))
    configured_sha = config["data"].get("manifest_sha256")
    if configured_sha not in {"__FROM_FORMAL_INPUT_AUDIT__", audit["formal_manifest_sha256"]}:
        raise ValueError("模板 manifest_sha256 与正式输入审计不一致")
    config["data"]["manifest_sha256"] = audit["formal_manifest_sha256"]
    config["data"]["manifest_version"] = FORMAL_CROP_VERSION
    config["data"]["cv3_manifest_sha256"] = CV3_SHA256
    config["data"]["input_audit_sha256"] = sha256_file(audit_path)
    config["data"]["input_audit_path"] = str(audit_path)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "p03_formal_config_frozen",
                "output": str(output),
                "output_sha256": sha256_file(output),
                "manifest_sha256": audit["formal_manifest_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
