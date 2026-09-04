#!/usr/bin/env python3
"""H67 fullres 组件证据完成后，纯 CPU 生成 TESC/RQTB 架构筛选报告。"""

from __future__ import annotations

import fcntl
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from evidence_provenance import (
    validate_motion_rqtb_provenance,
    validate_motion_tesc_provenance,
)


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PROFILE = ROOT / "results/h67_fullres_ep30_t450_profile100_20260805/nts11_hardware_p0_profile.json"
UPSTREAM_STATUS = ROOT / "results/h67_fullres_ep30_t450_profile_watcher_20260805.log"
UPSTREAM_COMPLETE = "ALL COMPLETE H67 ep30 fullres T450 profile100/all12 trace audit"
TESC_OUT = ROOT / "results/motion_temporal_equivalence_fullres_t450_20260806"
RQTB_OUT = ROOT / "results/motion_rqtb_fullres_t450_20260806"
STATUS = ROOT / "results/motion_rqtb_fullres_watcher_20260806.log"
LOCK = ROOT / "results/motion_rqtb_fullres_watcher_20260806.lock"
PYTHON = "/opt/conda/envs/sdformerflow/bin/python"
WATCHER = Path(__file__).resolve()
TEST_LOG = TESC_OUT / "provenance_unittest.log"


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def upstream_complete() -> bool:
    return (
        UPSTREAM_STATUS.is_file()
        and UPSTREAM_COMPLETE
        in UPSTREAM_STATUS.read_text(encoding="utf-8", errors="replace")
    )


def output_complete() -> bool:
    report = RQTB_OUT / "report.json"
    if not report.is_file():
        return False
    try:
        result = json.loads(report.read_text(encoding="utf-8"))
        tesc = json.loads((TESC_OUT / "report.json").read_text(encoding="utf-8"))
        validate_motion_tesc_provenance(tesc)
        validate_motion_rqtb_provenance(result)
    except (json.JSONDecodeError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return (
        result.get("status") == "BOUNDED_MODEL_COMPLETE"
        and result.get("source", {}).get("temporal_tokens") == 450
        and Path(str(result.get("source", {}).get("profile", ""))).resolve()
        == PROFILE.resolve()
    )


def run(command: list[str], label: str) -> None:
    record(f"START {label}: {' '.join(command)}")
    result = subprocess.run(command, cwd=REPO)
    record(f"END {label}: exit_code={result.returncode}")
    if result.returncode:
        raise RuntimeError(f"{label} failed")


def run_provenance_tests() -> None:
    TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
    command = [
        PYTHON,
        "-m",
        "unittest",
        "-v",
        "tests.test_new_dual_line_architecture_models",
        "tests.test_model_motion_reversible_quotient_bundle",
        "tests.test_motion_model_provenance",
    ]
    record(f"START Motion provenance tests: {' '.join(command)}")
    with TEST_LOG.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    record(f"END Motion provenance tests: exit_code={result.returncode}")
    if result.returncode:
        raise RuntimeError("Motion provenance tests failed")


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            record("EXIT another Motion RQTB watcher owns the lock")
            return 0
        if output_complete():
            record("REUSE completed Motion fullres T450 RQTB report")
            return 0
        while not upstream_complete():
            record("WAIT H67 fullres T450 component evidence")
            time.sleep(300)
        if not PROFILE.is_file():
            raise FileNotFoundError(PROFILE)
        run_provenance_tests()
        run(
            [
                PYTHON,
                "hw_autoresearch_nts07/scripts/analyze_motion_temporal_equivalence.py",
                "--profile",
                str(PROFILE),
                "--output-dir",
                str(TESC_OUT),
                "--watcher",
                str(WATCHER),
                "--test-log",
                str(TEST_LOG),
            ],
            "Motion fullres T450 TESC profile model",
        )
        run(
            [
                PYTHON,
                "hw_autoresearch_nts07/scripts/model_motion_reversible_quotient_bundle.py",
                "--compact",
                str(PROFILE),
                "--tesc",
                str(TESC_OUT / "report.json"),
                "--out",
                str(RQTB_OUT),
                "--watcher",
                str(WATCHER),
                "--test-log",
                str(TEST_LOG),
            ],
            "Motion fullres T450 RQTB screen",
        )
        if not output_complete():
            raise RuntimeError("Motion fullres T450 RQTB output identity check failed")
        record("ALL COMPLETE Motion fullres T450 TESC/RQTB profile models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
