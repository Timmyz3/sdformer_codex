#!/usr/bin/env python3
"""Parse one JSON file while rejecting duplicate keys and non-standard tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def pairs(items: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject(token: str) -> None:
    raise RuntimeError(f"non-standard JSON token: {token}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    json.loads(
        args.path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
