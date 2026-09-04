#!/usr/bin/env python3
"""Fail when a paper-PPA run did not retain every expected memory macro."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected", required=True, help="Comma-separated reference names")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    text = args.report.read_text(encoding="utf-8", errors="replace")
    refs = [item.strip() for item in args.expected.split(",") if item.strip()]
    checks = {
        ref: bool(
            re.search(
                rf"(?<![A-Za-z0-9_$]){re.escape(ref)}(?![A-Za-z0-9_$])",
                text,
            )
        )
        for ref in refs
    }
    passed = bool(refs) and all(checks.values())
    result = {
        "status": "PASS" if passed else "FAIL",
        "expected_references": checks,
        "boundary": "Reference-name presence only; macro timing/power models still require review.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
