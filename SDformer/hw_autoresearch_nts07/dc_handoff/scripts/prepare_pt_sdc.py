#!/usr/bin/env python3
"""Remove the DC-corner binding from a mapped SDC for an explicit PT corner."""

from __future__ import annotations

import argparse
from pathlib import Path


def remove_operating_condition_commands(text: str) -> tuple[str, int]:
    kept: list[str] = []
    removing = False
    removed = 0
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if not removing and stripped.startswith("set_operating_conditions"):
            removed += 1
            removing = line.rstrip().endswith("\\")
            continue
        if removing:
            removing = line.rstrip().endswith("\\")
            continue
        kept.append(line)
    if removing:
        raise ValueError("unterminated set_operating_conditions command")
    return "".join(kept), removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operating-condition", required=True)
    parser.add_argument(
        "--allow-corner-neutral-source",
        action="store_true",
        help="admit an explicit source SDC containing no DC corner binding",
    )
    args = parser.parse_args()
    text = args.source.read_text(encoding="utf-8")
    prepared, removed = remove_operating_condition_commands(text)
    admitted_counts = {1}
    if args.allow_corner_neutral_source:
        admitted_counts.add(0)
    if removed not in admitted_counts:
        raise ValueError(
            "expected one DC operating-condition command"
            + (" or a corner-neutral explicit SDC" if args.allow_corner_neutral_source else "")
            + f", found {removed}"
        )
    header = (
        f"# PrimeTime effective SDC: removed_dc_corner_commands={removed}; "
        f"explicit PT corner {args.operating_condition}\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header + prepared, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
