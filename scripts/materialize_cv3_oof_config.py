#!/usr/bin/env python3
"""Resolve a checked-in CV3 OOF YAML template for one fold."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

TOKEN_RE = re.compile(r"__[A-Z0-9_]+__")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="解析单折 OOF 训练/推理配置模板")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-view", type=Path, required=True)
    parser.add_argument("--fold-output-dir", type=Path, required=True)
    parser.add_argument("--pretrained-weight", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        text = args.template.read_text(encoding="utf-8")
        replacements = {
            "__FOLD__": str(args.fold),
            "__DATA_ROOT__": str(args.data_root.resolve()),
            "__FOLD_SPLIT_VIEW__": str(args.split_view.resolve()),
            "__FOLD_OUTPUT_DIR__": str(args.fold_output_dir.resolve()),
        }
        if args.pretrained_weight is not None:
            weight = str(args.pretrained_weight.resolve())
            replacements["__OFFICIAL_YOLO26S_PRETRAINED_PATH__"] = weight
            replacements["__OFFICIAL_RTDETR_L_PRETRAINED_PATH__"] = weight
        if args.checkpoint is not None:
            replacements["__FOLD_LAST_CHECKPOINT__"] = str(args.checkpoint.resolve())
        for token, value in replacements.items():
            text = text.replace(token, value)
        unresolved = sorted(set(TOKEN_RE.findall(text)))
        if unresolved:
            raise ValueError(f"配置仍有未解析 token: {unresolved}")
        payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise ValueError("解析后的配置顶层必须是对象")
        if args.output.exists():
            raise FileExistsError(f"输出配置已存在，禁止覆盖: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        print(f"CV3_OOF_CONFIG_FAIL: {error}", file=sys.stderr)
        return 1
    print(f"CV3_OOF_CONFIG_PASS fold={args.fold} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
