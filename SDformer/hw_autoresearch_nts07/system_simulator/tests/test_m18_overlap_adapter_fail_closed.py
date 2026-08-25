#!/usr/bin/env python3
"""Prove the retired M18 P_DONE adapter rejects both old and certified inputs first."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from simulate_m18_exact_population_direct_overlap import load_inputs


def expect_rejection(path: Path, expected: str, missing_root: Path) -> None:
    try:
        load_inputs(missing_root / "oracle", missing_root / "reconciliation.json", path)
    except ValueError as exc:
        assert expected in str(exc), (expected, str(exc))
    else:
        raise AssertionError("retired direct-overlap adapter accepted a boundary artifact")


def main() -> int:
    tests = 0
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        missing_root = root / "deliberately_missing_heavy_inputs"

        legacy = root / "legacy_v1.json"
        legacy.write_text(json.dumps({
            "schema": "m18_direct_m4_boundaries_v1",
            "status": "PASS_EXACT_DIRECT_M4_BOUNDARIES",
        }), encoding="utf-8")
        expect_rejection(legacy, "legacy M18 direct-boundary overlap is invalidated", missing_root)
        tests += 1

        certified = root / "bn_blocked_v2.json"
        certified.write_text(json.dumps({
            "schema": "m18_direct_m4_bn_blocked_path_certificates_v2",
            "status": "PASS_EXACT_PATH_CERTIFICATES_ALL_BN_BLOCKED_M15_PROHIBITED",
        }), encoding="utf-8")
        expect_rejection(certified, "every historical direct-M4 edge", missing_root)
        tests += 1

    print("PASS m18-overlap-adapter-fail-closed {}/{}".format(tests, tests))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
