from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/report_local5_score_active_cross_head.py"
SPEC = importlib.util.spec_from_file_location("score_active_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_log_requires_one_exact_pass(tmp_path: Path) -> None:
    log = tmp_path / "sim.log"
    log.write_text(
        "PASS Local5 cross-head OUT32 seed=17717 cycles=263583 "
        "heads=3 partial=43200 final=14400 result_stall=14392 "
        "group_stall=0\n",
        encoding="utf-8",
    )
    parsed = MODULE.parse_log(log)
    assert parsed["cycles"] == 263583
    assert parsed["partial"] == 43200
    assert parsed["final"] == 14400


def test_parse_log_rejects_failure_marker(tmp_path: Path) -> None:
    log = tmp_path / "sim.log"
    log.write_text(
        "ERROR: injected\n"
        "PASS Local5 cross-head OUT32 seed=17717 cycles=263583 "
        "heads=3 partial=43200 final=14400 result_stall=14392 "
        "group_stall=0\n",
        encoding="utf-8",
    )
    try:
        MODULE.parse_log(log)
    except ValueError:
        return
    raise AssertionError("failure marker must be rejected")


def test_tool_log_requires_clean_marker(tmp_path: Path) -> None:
    log = tmp_path / "tool.log"
    log.write_text("Found and reported 0 problems.\n", encoding="utf-8")
    assert MODULE.tool_log_is_clean(
        log, required="Found and reported 0 problems."
    )
    log.write_text(
        "ERROR: injected\nFound and reported 0 problems.\n",
        encoding="utf-8",
    )
    assert not MODULE.tool_log_is_clean(
        log, required="Found and reported 0 problems."
    )
