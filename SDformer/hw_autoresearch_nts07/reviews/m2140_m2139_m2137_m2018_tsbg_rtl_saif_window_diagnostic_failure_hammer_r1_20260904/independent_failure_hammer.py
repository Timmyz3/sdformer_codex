#!/usr/bin/env python3
"""Read-only M2140 audit of the consumed M2139 RTL-SAIF diagnostic.

This script invokes no EDA executable, license query, or GPU process.  It
checks the sealed evidence already emitted by M2139 and characterizes the
nonzero-TX records without changing any predecessor artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
REPO = HW.parent
Q = (HW / "results/"
     "m2139_m2137_m2018_tsbg_rtl_saif_window_diagnostic_r1_20260904"
     ".failed.2153005.quarantine")
ATTEMPT = (HW / "results/"
           ".m2139_m2137_tsbg_rtl_saif_window_diagnostic_attempt_consumed")
RESULT = (HW / "results/"
          "m2139_m2137_m2018_tsbg_rtl_saif_window_diagnostic_r1_20260904")
LOCK = (HW / "results/"
        ".m2139_m2137_tsbg_rtl_saif_window_diagnostic_launch_lock")
M2137 = (HW / "dc_handoff/scripts/"
         "run_m2137_m2018_tsbg_rtl_saif_window_diagnostic_one_shot.py")
M2137_CONTRACT = (HW / "contracts/"
                  "m2137_m2018_tsbg_rtl_saif_window_diagnostic_source_contract_r1_20260904.json")
M2137_SELFCHECK = (HW / "reviews/"
                   "m2137_m2018_tsbg_rtl_saif_window_diagnostic_source_selfcheck_r1_20260904")
M2138 = (HW / "reviews/"
         "m2138_m2137_m2018_tsbg_rtl_saif_window_diagnostic_source_hammer_r1_20260904")
M2125_SELFCHECK = (HW / "reviews/"
                   "m2125_m2018_tsbg_rtl_saif_window_diagnostic_source_selfcheck_r1_20260904")
M2126 = (HW / "reviews/"
         "m2126_m2125_m2018_tsbg_rtl_saif_window_diagnostic_source_hammer_r1_20260904")
M2125_TB = (HW / "tb_m2018/"
            "tb_m2125_m2018_tsbg_rtl_saif_window_diagnostic.sv")
M2125_UCLI = (HW / "dc_handoff/scripts/"
              "m2125_m2018_tsbg_ordinary_rtl_saif_window_diagnostic.ucli.tcl")
M2018_RTL = (HW / "rtl_m2018/"
             "m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv")
M803_RTL = (HW / "rtl_m803/"
            "m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv")
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOC359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            assert key not in value, "duplicate JSON key: " + key
            value[key] = item
        return value
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          AssertionError("nonfinite JSON: " + token)))


def verify_dir(root: Path) -> dict[str, str]:
    assert root.is_dir() and not root.is_symlink(), root
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    assert outer.read_text().split() == [sha(manifest), "SHA256SUMS"]
    rows = {}
    for line in manifest.read_text().splitlines():
        digest, rel_text = line.split(maxsplit=1)
        rel = Path(rel_text.lstrip("*"))
        assert not rel.is_absolute() and ".." not in rel.parts
        path = root / rel
        assert path.is_file() and not path.is_symlink()
        assert sha(path) == digest and rel.as_posix() not in rows
        rows[rel.as_posix()] = digest
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file()
              and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    assert set(rows) == actual
    assert not any(path.is_symlink() for path in root.rglob("*"))
    return rows


def parse_saif(path: Path) -> dict:
    text = path.read_text(errors="replace")
    duration = re.findall(r"\(DURATION\s+([0-9.eE+-]+)\)", text)
    assert duration == ["60876.00"]
    rows = re.findall(
        r"\(T0\s+([0-9.eE+-]+)\)\s*\(T1\s+([0-9.eE+-]+)\)\s*"
        r"\(TX\s+([0-9.eE+-]+)\)\s*\(TC\s+([0-9.eE+-]+)\)", text)
    assert len(rows) == 93971
    assert all(abs(sum(map(float, row[:3])) - 60876.0) <= 1e-6 for row in rows)

    # Associate each activity tuple with the immediately preceding SAIF name.
    named = re.findall(
        r"\(([^()\s]+)\s*\n\s*\(T0\s+([0-9.eE+-]+)\)\s*"
        r"\(T1\s+([0-9.eE+-]+)\)\s*\(TX\s+([0-9.eE+-]+)\)\s*"
        r"\(TC\s+([0-9.eE+-]+)\)", text)
    assert len(named) == 93971
    bad = [(name, float(tx)) for name, _t0, _t1, tx, _tc in named
           if float(tx) != 0.0]
    families = {}
    tx_histogram = {}
    for name, tx in bad:
        family = name.split(r"\[")[0]
        families[family] = families.get(family, 0) + 1
        key = str(int(tx)) if tx.is_integer() else str(tx)
        tx_histogram[key] = tx_histogram.get(key, 0) + 1
    expected_families = {
        "row_live_q": 191,
        "cache_valid_q": 3,
        "bridge_overflow": 15,
        "slot_valid_q": 7,
        "rsp_shape_legal": 7,
    }
    expected_histogram = {
        "1": 23, "34": 1, "64": 1, "97": 1, "127": 1,
        "160": 1, "190": 1, "397": 192, "817": 1, "1219": 1,
    }
    assert families == expected_families
    assert tx_histogram == expected_histogram
    expected_names = {
        rf"row_live_q\[{ctx}\]\[{group}\]"
        for ctx in range(4) for group in range(48)
        if (ctx, group) != (0, 0)
    }
    expected_names |= {rf"cache_valid_q\[{index}\]" for index in range(1, 4)}
    expected_names |= {rf"bridge_overflow\[{index}\]" for index in range(1, 16)}
    expected_names |= {rf"slot_valid_q\[{index}\]" for index in range(1, 8)}
    expected_names |= {rf"rsp_shape_legal\[{index}\]" for index in range(1, 8)}
    assert {name for name, _tx in bad} == expected_names
    assert len(bad) == 223 and sum(tx for _name, tx in bad) == 78955.0
    return {
        "duration_ns": 60876.0,
        "record_count": len(rows),
        "conservation_failures": 0,
        "tx_nonzero_record_count": len(bad),
        "tx_sum": sum(tx for _name, tx in bad),
        "tx_signal_families": families,
        "tx_duration_histogram_ns": tx_histogram,
        "saif_sha256": sha(path),
    }


def main() -> None:
    checks = {}

    def need(value: bool, label: str) -> None:
        checks[label] = bool(value)
        assert value, label

    attempt_members = verify_dir(ATTEMPT)
    quarantine_members = verify_dir(Q)
    m2137_selfcheck_members = verify_dir(M2137_SELFCHECK)
    m2138_members = verify_dir(M2138)
    m2125_selfcheck_members = verify_dir(M2125_SELFCHECK)
    m2126_members = verify_dir(M2126)
    need(set(attempt_members) == {"attempt.json"},
         "attempt_exhaustive_one_member")
    need(set(quarantine_members) == {
        "FAILED_DO_NOT_CITE.txt", "execution_commands.json",
        "execution_counts.json", "license_preflight.log",
        "ordinary_lru4/rtl_execute.saif", "ordinary_lru4/rtl_sim.log",
        "ordinary_lru4/runtime_parse.log", "ordinary_lru4/saif_parse.log",
        "vcs_compile.log"}, "quarantine_exhaustive_nine_members")
    need(bool(m2137_selfcheck_members) and bool(m2138_members)
         and bool(m2125_selfcheck_members) and bool(m2126_members),
         "authority_and_inherited_reviews_exhaustively_sealed")

    attempt = strict_json(ATTEMPT / "attempt.json")
    counts = strict_json(Q / "execution_counts.json")
    commands = strict_json(Q / "execution_commands.json")
    need(attempt["status"] == "M2139_ATTEMPT_CONSUMED"
         and attempt["automatic_retry"] is False,
         "attempt_consumed_no_retry")
    need(attempt["budget"] == {"license_queries": 1, "vcs_compiles": 1,
         "simv_runs": 2, "saif_files": 2, "dc_runs": 0,
         "ptpx_runs": 0}, "attempt_budget_exact")
    need(counts == {"license_queries": 1, "vcs_compiles": 1,
         "simv_runs": 1, "saif_files": 0, "dc_runs": 0,
         "ptpx_runs": 0}, "executed_counts_exact")
    need(not RESULT.exists() and not LOCK.exists(),
         "canonical_result_and_launch_lock_absent")
    failure = (Q / "FAILED_DO_NOT_CITE.txt").read_text()
    need("status=FAILED_DO_NOT_CITE" in failure
         and "automatic_retry=false" in failure,
         "failure_disposition_exact")

    contract = strict_json(M2137_CONTRACT)
    review = strict_json(M2138 / "review.json")
    need(sha(M2137) == "a1a72dcdfbbf0f1f0cbae52424b1dac08b023edd612223236f9c2fb77e7445d4",
         "m2137_runner_sha_exact")
    need(sha(M2137_CONTRACT) ==
         "42d2394942f25e80a28b6b448ad966715366dc3d71ea60e5cf1899b07b89b2cd",
         "m2137_contract_sha_exact")
    sidecar = Path(str(M2137_CONTRACT) + ".sha256")
    outer = Path(str(M2137_CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [sha(M2137_CONTRACT), M2137_CONTRACT.name]
         and outer.read_text().split() == [sha(sidecar), sidecar.name],
         "m2137_contract_double_seal")
    need(contract["status"] ==
         "SOURCE_ONLY__M2138_INDEPENDENT_REVIEW_REQUIRED__NO_EDA",
         "m2137_source_only_status")
    need(review["status"] ==
         "PASS_M2138_M2137_SOURCE_HAMMER__OPTION_AWARE_GUARD_EXACT__M2139_SINGLE_ATTEMPT_AUTHORIZED"
         and review["score_over_100"] == 100
         and review["severity_counts"] == {"p0": 0, "p1": 0, "p2": 0},
         "m2138_authority_pass_exact")
    need(review["authorization"] == {"license_queries": 1,
         "vcs_compiles": 1, "simv_runs": 2, "saif_files": 2,
         "dc_runs": 0, "ptpx_runs": 0, "automatic_retry": False,
         "p1_serial": True, "reuse_old_artifacts": False},
         "m2138_authorization_exact")
    need(review["identity"]["runner_sha256"] == sha(M2137)
         and review["identity"]["contract_sha256"] == sha(M2137_CONTRACT),
         "m2138_binds_m2137_identity")
    need(sha(DOC359) == DOC359_SHA, "docs359_unchanged")

    need(commands["timing_surface"] == {"active_input_count": 7,
         "explicit_sdf_options": 0, "explicit_unit_delay_defines": 0,
         "path_operands_may_contain_sdf_substring": True},
         "timing_surface_clean")
    compile_cmd = commands["vcs_compile"]
    sim_cmd = commands["simv"]["ordinary_lru4"]
    need(compile_cmd.count("+vcs+initreg+random") == 1
         and sim_cmd.count("+vcs+initreg+0") == 1
         and sim_cmd.count("+WORKLOAD_SLOT=42") == 1
         and sim_cmd.count("+M2125_AXIS_ORDINARY") == 1,
         "compile_runtime_identity_exact")
    need("CPU time:" in (Q / "vcs_compile.log").read_text()
         and "simv up to date" in (Q / "vcs_compile.log").read_text(),
         "vcs_compile_completed")

    runtime = (Q / "ordinary_lru4/rtl_sim.log").read_text(errors="replace")
    need(runtime.count("M2125_RTL_SAIF_WINDOW_BEGIN") == 1
         and runtime.count("M2125_RTL_SAIF_WINDOW_END") == 1
         and runtime.count("PASS_M2125_RTL_SAIF_WINDOW_DIAGNOSTIC_AXIS") == 1,
         "runtime_markers_unique")
    need("measurement_cycles=20292 scalar_weight_reads=14304 duration_ns=60876.00"
         in runtime and "ledger_exact=1" in runtime,
         "ordinary_runtime_ledger_and_duration_exact")
    need(not re.search(r"(^|\n)(Fatal:|Error:)|Assertion failed", runtime),
         "ordinary_runtime_no_functional_fatal")
    runtime_parse = strict_json(Q / "ordinary_lru4/runtime_parse.log")
    need(runtime_parse["completion_ledger_exact"] is True
         and runtime_parse["measurement_cycles"] == 20292
         and runtime_parse["scalar_weight_reads"] == 14304
         and runtime_parse["duration_ns"] == 60876.0,
         "runtime_parser_accepts_exact_ledger")

    saif = parse_saif(Q / "ordinary_lru4/rtl_execute.saif")
    need((Q / "ordinary_lru4/saif_parse.log").read_text().strip() ==
         "M2125_FAIL_CLOSED: SAIF unknown activity: records=223 sum=78955.0",
         "parser_rejection_exact")
    need(saif["record_count"] == 93971
         and saif["conservation_failures"] == 0
         and saif["tx_nonzero_record_count"] == 223
         and saif["tx_sum"] == 78955.0,
         "actual_saif_fingerprint_exact")

    ucli = M2125_UCLI.read_text().splitlines()
    need(ucli == [
        "power -gate_level all mda sv",
        "power tb_m2125_m2018_tsbg_rtl_saif_window_diagnostic.core.dut_base.implementation",
        "run", "power -enable", "run", "power -disable",
        "power -report $::env(M2125_RTL_SAIF_FILE) 1e-9 tb_m2125_m2018_tsbg_rtl_saif_window_diagnostic.core.dut_base.implementation",
        "quit"], "late_enable_ucli_sequence_exact")
    tb = M2125_TB.read_text()
    need("check_selected_known();\n        measurement_begin_time" in tb
         and "row_live_q" not in tb and "cache_valid_q" not in tb
         and "slot_valid_q" not in tb and "bridge_overflow" not in tb
         and "rsp_shape_legal" not in tb,
         "tb_checks_public_not_internal_tx_families")
    m2018_rtl = M2018_RTL.read_text()
    m803_rtl = M803_RTL.read_text()
    need("row_live_q[ctx][group] <= 0;" in m2018_rtl
         and "cache_valid_q[entry] <= 0;" in m2018_rtl
         and "slot_valid_q[slot] <= 0;" in m803_rtl,
         "stateful_tx_families_have_explicit_reset_loops")
    selfcheck = strict_json(M2125_SELFCHECK / "selfcheck.json")
    need(selfcheck["execution_performed"] == {"license_queries": 0,
         "vcs_compiles": 0, "simv_runs": 0, "saif_files": 0,
         "dc_runs": 0, "ptpx_runs": 0},
         "m2125_selfcheck_was_static_no_eda")
    source_review = strict_json(M2126 / "review.json")
    source_mechanical = strict_json(M2126 / "mechanical_checks.json")
    need(source_review["parser_admission"]["record_count_each_axis"] == 93971
         and source_review["parser_admission"]["every_record_tx_zero"] is True
         and source_mechanical["checks"]["saif_positive_93971_tx0_conserved_active"] is True,
         "m2126_synthetic_parser_fixture_passed")

    result = {
        "schema": "m2140_m2139_m2137_m2018_tsbg_rtl_saif_window_diagnostic_failure_mechanical_checks_r1_v1",
        "status": "PASS_M2140_MECHANICAL_FAILURE_AUDIT__NO_EDA",
        "checks": checks,
        "check_count": len(checks),
        "all_checks_pass": all(checks.values()),
        "execution_invoked_by_review": {"license_queries": 0,
            "vcs_compiles": 0, "simv_runs": 0, "saif_files": 0,
            "dc_runs": 0, "ptpx_runs": 0, "icc2_runs": 0,
            "gpu_runs": 0},
        "m2139_execution": {**counts, "raw_saif_files_written": 1,
            "admitted_saif_files": 0},
        "saif": saif,
        "identity": {
            "attempt_manifest_sha256": sha(ATTEMPT / "SHA256SUMS"),
            "attempt_outer_sha256": sha(ATTEMPT / "SHA256SUMS.seal.sha256"),
            "quarantine_manifest_sha256": sha(Q / "SHA256SUMS"),
            "quarantine_outer_sha256": sha(Q / "SHA256SUMS.seal.sha256"),
            "m2137_runner_sha256": sha(M2137),
            "m2137_contract_sha256": sha(M2137_CONTRACT),
            "m2138_review_json_sha256": sha(M2138 / "review.json"),
            "m2125_tb_sha256": sha(M2125_TB),
            "m2125_ordinary_ucli_sha256": sha(M2125_UCLI),
            "m2018_rtl_sha256": sha(M2018_RTL),
            "m803_rtl_sha256": sha(M803_RTL),
            "docs359_sha256": sha(DOC359),
        },
    }
    (HERE / "mechanical_checks.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
