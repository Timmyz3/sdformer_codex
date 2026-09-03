#!/usr/bin/env python3
"""Read-only source audit for the M2031 ep34 C1 calibration cohort.

The audit binds the 64-line fixture to the prefix of the sealed M1590 ledger
and recomputes the service-event expectations through the frozen M505 model.
It launches no EDA and writes no result.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HW = Path(__file__).resolve().parents[2]
LEDGER_DIR = HW / "results/m1590_ep34_c1_same_ledger_cycle_model_r1_20260901"
LEDGER = LEDGER_DIR / "ep34_c1_support16_rows.memh"
RESULT = LEDGER_DIR / "m1579_ep34_c1_same_ledger_cycle_model_result_r1.json"
FIXTURE = HW / "tb_m528_dw1rw/fixtures/m2031_ep34_c1_first64_support16.memh"
MODEL = HW / "system_simulator/scripts/analyze_m505_h67_liveness_aware_single_port_parent_scratch.py"

LEDGER_SHA256 = "daa6265115df9c0bae5d96e5a133a4b5fbc9786de75598e53ab2e5812bfdb835"
RESULT_SHA256 = "facfecaf3b25a4c79299517de31283ed3815af26a5dd87c91a6985f6fc68516f"
MODEL_SHA256 = "9d55d960d237a1940fb8e9efaa4e227a4ec1025489f80804d1c677e12bc9aced"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    for path in (LEDGER, RESULT, FIXTURE, MODEL):
        require(path.is_file() and not path.is_symlink(), "missing/non-regular input: " + str(path))
    require(sha256(LEDGER) == LEDGER_SHA256, "M1590 ledger SHA drift")
    require(sha256(RESULT) == RESULT_SHA256, "M1590 result SHA drift")
    require(sha256(MODEL) == MODEL_SHA256, "frozen M505 model SHA drift")

    fixture_lines = FIXTURE.read_bytes().splitlines(keepends=True)
    require(len(fixture_lines) == 64, "fixture row count drift")
    with LEDGER.open("rb") as stream:
        ledger_prefix = [stream.readline() for _ in range(64)]
    require(fixture_lines == ledger_prefix, "fixture is not the exact sealed-ledger prefix")

    spec = importlib.util.spec_from_file_location("m2031_frozen_m505", MODEL)
    require(spec is not None and spec.loader is not None, "cannot load frozen M505 model")
    model = importlib.util.module_from_spec(spec)
    sys.modules["m2031_frozen_m505"] = model
    spec.loader.exec_module(model)

    masks = np.asarray([int(line.strip(), 16) & 0xFFFF for line in fixture_lines], dtype=np.uint16)
    residual, parent = model.M504.cleanroom_subset(masks)
    replay = model.simulate_liveness_task(masks, False)
    expected = {
        "rows": 64,
        "active_rows": int(np.count_nonzero(masks)),
        "input_nnz": int(model.M504.POPCOUNT[masks].sum()),
        "residual_nnz": int(model.M504.POPCOUNT[residual].sum()),
        "exact_parent_rows": int(np.count_nonzero((parent >= 0) & (residual == 0) & (masks != 0))),
        "issue_accepts": int(replay["ideal_1r1w_issue_cycles"]),
        "parent_edges": int(replay["parent_edges"]),
        "dead_write_elisions": int(replay["dead_writes_elided"]),
        "macro_reads": int(replay["macro_reads"]),
        "macro_writes": int(replay["macro_writes"]),
        "forwards": int(replay["forwarded_reads"]),
        "deadline_holds": int(replay["liveness_deadline_holds"]),
        "issue_stalls": int(replay["liveness_stall_cycles"]),
        "psum_commits": int(np.count_nonzero(masks)),
        "row_completions": int(np.count_nonzero(masks)),
    }
    require(expected == {
        "rows": 64,
        "active_rows": 64,
        "input_nnz": 565,
        "residual_nnz": 192,
        "exact_parent_rows": 4,
        "issue_accepts": 196,
        "parent_edges": 58,
        "dead_write_elisions": 31,
        "macro_reads": 54,
        "macro_writes": 33,
        "forwards": 4,
        "deadline_holds": 6,
        "issue_stalls": 14,
        "psum_commits": 64,
        "row_completions": 64,
    }, "frozen expectation drift")

    print(json.dumps({
        "status": "PASS_M2031_EP34_C1_FIRST64_SOURCE_AUDIT__NO_EDA",
        "fixture_sha256": sha256(FIXTURE),
        "ledger_sha256": LEDGER_SHA256,
        "result_sha256": RESULT_SHA256,
        "model_sha256": MODEL_SHA256,
        "fixture_is_exact_ledger_prefix": True,
        "expected": expected,
        "claim_boundary": {
            "source_audit_only": True,
            "vcs": False,
            "rtl_cycle_speedup": False,
            "full_network": False,
            "system_speedup": False,
        },
    }, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
