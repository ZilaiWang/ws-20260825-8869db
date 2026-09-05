from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import AuditInputError, audit_manifest, load_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline experiment evidence audit; no model execution")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True, help="Allowed root for relative artifact paths")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fail-on-warning", action="store_true")
    args = parser.parse_args()
    if args.out.resolve() == args.manifest.resolve():
        parser.error("Output must not overwrite the input manifest")
    if args.out.exists():
        parser.error("Output already exists; choose a new path to preserve audit history")
    try:
        data = load_json(args.manifest)
        # Keep outputs from overwriting any artifact named in the manifest.
        if isinstance(data, dict):
            for item in data.get("artifacts", []):
                if isinstance(item, dict) and isinstance(item.get("path"), str):
                    if (args.root / item["path"]).resolve() == args.out.resolve():
                        parser.error("Output must not overwrite an audited artifact")
        report = audit_manifest(data, args.root)
        report["manifest_sha256"] = __import__("hashlib").sha256(args.manifest.read_bytes()).hexdigest()
    except (AuditInputError, OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: do not silently replace a concurrent job's report.
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"errors={report['errors']} warnings={report['warnings']}; report={args.out}")
    print("A zero exit code is not accuracy certification or deployment approval.")
    return 2 if report["errors"] or (args.fail_on_warning and report["warnings"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
