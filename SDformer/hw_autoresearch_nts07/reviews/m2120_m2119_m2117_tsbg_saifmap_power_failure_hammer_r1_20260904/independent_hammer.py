#!/usr/bin/env python3
"""Read-only M2120 hammer for the consumed M2119 failure.

The only subprocess used is the M2117 Python parser on synthetic files under
/tmp.  This script never invokes a license query, VCS, simv, DC, or PT.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
Q = HW / "results/m2119_m2117_m2018_tsbg_rtl_saifmap_power_r1_20260904.failed.1771297.quarantine"
ATTEMPT = HW / "results/.m2119_m2117_tsbg_rtl_saifmap_power_attempt_consumed"
LOCK = HW / "results/.m2119_m2117_tsbg_rtl_saifmap_power_launch_lock"
CANON = HW / "results/m2119_m2117_m2018_tsbg_rtl_saifmap_power_r1_20260904"
WORK = HW / "results/.m2119_m2117_work.cnnwk984"
PARSER = HW / "system_simulator/scripts/parse_m2117_m2018_tsbg_rtl_saifmap_power.py"
RUNNER = HW / "dc_handoff/scripts/run_m2117_m2018_tsbg_rtl_saifmap_power_one_shot.py"
TB = HW / "tb_m2018/tb_m2117_m2018_tsbg_rtl_saifmap_power.sv"
BASE_TB = HW / "tb_m2018/tb_m2051_ep34_tsbg_full40_cycle.sv"
UCLI = HW / "dc_handoff/scripts/m2117_m2018_tsbg_ordinary_rtl_saif.ucli.tcl"
M2118 = HW / "reviews/m2118_m2117_m2018_tsbg_rtl_saifmap_power_source_hammer_r1_20260904"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOC_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise AssertionError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          AssertionError(f"nonfinite JSON: {token}")))


def need(value: bool, label: str, checks: dict[str, bool]) -> None:
    checks[label] = bool(value)
    if not value:
        raise AssertionError(label)


def expect_failure(callback, label: str, checks: dict[str, bool],
                   contains: str | None = None) -> None:
    try:
        callback()
    except Exception as exc:
        if contains is not None and contains not in str(exc):
            raise AssertionError(f"{label}: wrong failure {exc}") from exc
        checks[label] = True
        return
    checks[label] = False
    raise AssertionError(label)


def verify_sealed_dir(root: Path) -> dict[str, object]:
    if not root.is_dir() or root.is_symlink():
        raise AssertionError(f"bad sealed root: {root}")
    manifest, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    if outer.read_text().split() != [sha(manifest), "SHA256SUMS"]:
        raise AssertionError(f"bad outer seal: {root}")
    expected = {}
    for line in manifest.read_text().splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise AssertionError(f"bad manifest syntax: {root}")
        rel = Path(fields[1].lstrip("*"))
        path = root / rel
        if rel.is_absolute() or ".." in rel.parts or rel.as_posix() in expected \
                or not path.is_file() or path.is_symlink() or sha(path) != fields[0]:
            raise AssertionError(f"bad manifest member: {root}/{rel}")
        expected[rel.as_posix()] = fields[0]
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file()
              and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    if actual != set(expected):
        raise AssertionError(f"nonexhaustive manifest: {root}")
    return {
        "members": len(expected), "manifest_sha256": sha(manifest),
        "outer_seal_file_sha256": sha(outer),
        "symlinks": sum(path.is_symlink() for path in root.rglob("*")),
    }


def write_saif(path: Path, duration: float) -> None:
    critical = [
        "mem_req_valid", "mem_rsp_valid", "bridge_valid", "commit_valid",
        "mem_req_accept", "mem_rsp_accept", "bridge_accept", "commit_accept",
    ]
    names = critical + [f"signal_{index}" for index in range(92)]
    rows = []
    for index, name in enumerate(names):
        t0, t1 = duration / 2.0, duration / 2.0
        rows.append(f"({name} (T0 {t0}) (T1 {t1}) (TX 0) (TC {1 + index % 3}))")
    path.write_text("(SAIFILE\n(TIMESCALE 1 ns)\n"
                    f"(DURATION {duration})\n(INSTANCE dut\n(NET\n"
                    + "\n".join(rows) + "\n)))\n")


def main() -> None:
    checks: dict[str, bool] = {}
    frozen = [RUNNER, PARSER, TB, BASE_TB, UCLI, DOC359]
    before = {path.as_posix(): sha(path) for path in frozen}

    attempt_seal = verify_sealed_dir(ATTEMPT)
    quarantine_seal = verify_sealed_dir(Q)
    need(attempt_seal["members"] == 1 and attempt_seal["symlinks"] == 0,
         "attempt_exhaustive_double_seal", checks)
    need(quarantine_seal["members"] == 7 and quarantine_seal["symlinks"] == 0,
         "quarantine_exhaustive_double_seal", checks)
    need(verify_sealed_dir(M2118)["members"] == 5,
         "m2118_source_review_exhaustive_double_seal", checks)
    attempt = strict_json(ATTEMPT / "attempt.json")
    counts = strict_json(Q / "execution_counts.json")
    need(attempt["status"] == "M2119_ATTEMPT_CONSUMED"
         and attempt["automatic_retry"] is False,
         "attempt_consumed_no_retry", checks)
    need(attempt["budget"] == {"license_queries": 1, "vcs_compiles": 1,
                               "simv_runs": 2, "dc_runs": 2,
                               "ptpx_runs": 2, "saif_files": 2},
         "attempt_budget_exact", checks)
    need(counts == {"license_queries": 1, "vcs_compiles": 1,
                    "simv_runs": 1, "dc_runs": 0,
                    "ptpx_runs": 0, "saif_files": 0},
         "executed_counts_exact", checks)
    need(not CANON.exists() and not LOCK.exists(),
         "canonical_and_lock_absent", checks)
    need(WORK.is_dir() and not (WORK / "SHA256SUMS").exists()
         and sum(path.is_symlink() for path in WORK.rglob("*")) == 2,
         "residual_work_tree_unsealed_and_noncitable", checks)
    failure = (Q / "FAILED_DO_NOT_CITE.txt").read_text()
    need("status=FAILED_DO_NOT_CITE" in failure
         and "ordinary_lru4/saif_parse.log" in failure
         and "automatic_retry=false" in failure,
         "failure_marker_exact", checks)
    need(not any((Q / "ordinary_lru4/dc").iterdir())
         and not any((Q / "ordinary_lru4/ptpx").iterdir()),
         "dc_and_ptpx_never_started", checks)
    need(not list(Q.rglob("power.rpt")) and not list(Q.rglob("result.json")),
         "no_power_or_result_number_exists", checks)

    simlog = (Q / "ordinary_lru4/rtl_sim.log").read_text(errors="replace")
    parse_log = (Q / "ordinary_lru4/saif_parse.log").read_text()
    need(simlog.count("M2117_RTL_SAIF_WINDOW_BEGIN") == 1
         and simlog.count("M2117_RTL_SAIF_WINDOW_END axis=ordinary_lru4") == 1,
         "ordinary_functional_window_reached_both_stops", checks)
    need("Fatal:" not in simlog and "Assertion failed" not in simlog,
         "ordinary_sim_no_function_or_assertion_fatal", checks)
    time_matches = re.findall(r"^Time:\s*([0-9]+) ps$", simlog, flags=re.MULTILINE)
    need(time_matches == ["62041510"], "vcs_final_timestamp_exact", checks)
    need(parse_log.strip() == "M2117_FAIL_CLOSED: duration 60877.5 != 60876.0",
         "parser_failure_exact", checks)

    saif_text = (Q / "ordinary_lru4/rtl_execute.saif").read_text(errors="replace")
    duration_matches = re.findall(r"\(DURATION\s+([0-9.]+)\)", saif_text)
    need(duration_matches == ["60877.50"], "raw_saif_duration_60877p5ns", checks)
    rows = re.findall(
        r"\(T0\s+([0-9.eE+-]+)\)\s*\(T1\s+([0-9.eE+-]+)\)\s*"
        r"\(TX\s+([0-9.eE+-]+)\)\s*\(TC\s+([0-9.eE+-]+)\)", saif_text)
    tx_sum = sum(float(row[2]) for row in rows)
    tx_nonzero = sum(float(row[2]) > 0.0 for row in rows)
    need(len(rows) == 93971 and tx_nonzero == 58277
         and tx_sum == 40619426.0,
         "actual_saif_global_tx_failure_exact", checks)

    # Timestamp/phase derivation: final stop minus recorded duration is the
    # first stop.  With a 3-ns clock, remainder 10 ps is negedge+settle and
    # remainder 1510 ps is posedge+settle.
    end_ps = int(time_matches[0])
    duration_ps = float(duration_matches[0]) * 1000.0
    start_ps = end_ps - duration_ps
    need(start_ps == 1164010.0, "saif_start_timestamp_1164010ps", checks)
    need(start_ps % 3000 == 10 and end_ps % 3000 == 1510,
         "window_starts_negedge_ends_posedge", checks)
    need(duration_ps == (20292 + 0.5) * 3000,
         "half_cycle_excess_arithmetic_exact", checks)
    need(duration_ps - 20292 * 3000 == 1500,
         "duration_excess_exactly_1p5ns", checks)

    tb_text, base_text, ucli_text = TB.read_text(), BASE_TB.read_text(), UCLI.read_text()
    need("always #1.5 clk_core = ~clk_core" in base_text,
         "base_clock_is_3ns", checks)
    need("@(negedge clk_core);\n            load_valid_base = 0" in base_text
         and "full_execute_start_cycle = tb_cycle;" in base_text,
         "full_execute_marker_follows_load_negedge", checks)
    block = tb_text[tb_text.index("initial begin : m2117_fixed_window") :]
    need("wait (core.full_execute_start_cycle >= 0);\n        #0.01;" in block
         and "wait (core.base_done_cycle >= 0);" in block
         and "wait (core.tsbg_done_cycle >= 0);" in block
         and "@(negedge core.clk_core);" not in block,
         "m2117_wrapper_has_mixed_phase_stops", checks)
    need(ucli_text.count("run") == 2
         and ucli_text.index("power -enable") < ucli_text.rindex("run")
         < ucli_text.index("power -disable"),
         "ucli_brackets_exact_tb_stops", checks)

    # The production parser is right to reject the observed interval.  Prove
    # the erroneous +0.5-cycle interval is rejected and an otherwise identical
    # exact-cycle synthetic interval is accepted.
    spec = importlib.util.spec_from_file_location("m2117_parser_m2120", PARSER)
    parser = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(parser)
    with tempfile.TemporaryDirectory(prefix="m2120_window_mutation.") as tmp_name:
        tmp = Path(tmp_name)
        wrong = tmp / "wrong_60877p5.saif"
        correct = tmp / "correct_60876.saif"
        write_saif(wrong, 60877.5)
        write_saif(correct, 60876.0)
        expect_failure(lambda: parser.parse_saif(wrong, "ordinary_lru4"),
                       "wrong_half_cycle_window_rejected", checks,
                       "duration 60877.5 != 60876.0")
        accepted = parser.parse_saif(correct, "ordinary_lru4")
        need(accepted["duration_ns"] == 60876.0
             and accepted["tx_sum"] == 0.0,
             "correct_exact_cycle_window_accepted", checks)
        actual_duration_repaired = tmp / "actual_duration_only_repaired.saif"
        actual_duration_repaired.write_text(saif_text.replace(
            "(DURATION 60877.50)", "(DURATION 60876.00)", 1))
        expect_failure(lambda: parser.parse_saif(
            actual_duration_repaired, "ordinary_lru4"),
            "duration_only_repair_still_rejected_for_tx", checks,
            "SAIF TX sum nonzero")

    need(sha(DOC359) == DOC_SHA, "docs359_identity_preserved", checks)
    after = {path.as_posix(): sha(path) for path in frozen}
    need(before == after, "all_frozen_sources_unchanged", checks)
    result = {
        "schema": "m2120_m2119_failure_independent_hammer_r1_v1",
        "status": "PASS_M2120_M2119_FAILURE_HAMMER__NO_POWER__FRESH_VCS_DIAGNOSTIC_SOURCE_ONLY_ALLOWED",
        "eda_or_license_invoked": False,
        "check_count": len(checks),
        "checks": checks,
        "seals": {"attempt": attempt_seal, "quarantine": quarantine_seal},
        "observed": {
            "execution_counts": counts,
            "vcs_end_ps": end_ps,
            "saif_start_ps": start_ps,
            "duration_ns": float(duration_matches[0]),
            "contract_duration_ns": 20292 * 3.0,
            "phase_excess_ns": 1.5,
            "saif_records": len(rows),
            "tx_nonzero_records": tx_nonzero,
            "tx_sum": tx_sum,
        },
        "identity": {
            "attempt_json_sha256": sha(ATTEMPT / "attempt.json"),
            "failure_marker_sha256": sha(Q / "FAILED_DO_NOT_CITE.txt"),
            "execution_counts_sha256": sha(Q / "execution_counts.json"),
            "saif_sha256": sha(Q / "ordinary_lru4/rtl_execute.saif"),
            "sim_log_sha256": sha(Q / "ordinary_lru4/rtl_sim.log"),
            "saif_parse_log_sha256": sha(Q / "ordinary_lru4/saif_parse.log"),
            "vcs_compile_log_sha256": sha(Q / "vcs_compile.log"),
            "docs359_sha256": sha(DOC359),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
