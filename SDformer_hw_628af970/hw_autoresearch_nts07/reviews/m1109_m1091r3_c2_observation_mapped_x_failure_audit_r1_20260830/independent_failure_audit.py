#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Read-only independent audit of the frozen M1091r3 failed attempt.

This program deliberately does not import or invoke the M1091r3 engine, DC,
VCS, or simv.  It validates sealed evidence and extracts only failure-local
facts.  Run from any directory; JSON is written to stdout only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
ATTEMPT = HW / "results/.m1091r3_m1090r3_c2_observation_dc_mapped_vcs_attempt_consumed"
QUARANTINE = HW / (
    "results/m1091r3_m1090r3_c2_observation_dc_mapped_vcs_r1_20260830."
    "failed_or_incomplete.3507131.quarantine"
)
NETLIST = QUARANTINE / "dc/netlist/m1090r3_c2_k1_observation_wrapper_mapped.v"
TB = HW / "dc_handoff/tb/tb_m1090r3_c2_k1_observation_mapped_case0_short.sv"
WRAPPER = HW / "rtl_m1090r3/m1090r3_c2_k1_observation_wrapper.sv"
SERVICE = HW / "rtl_m1058/m1058_fc2_k1_reset_hygiene_registered_release_service_island.sv"
ENGINE = HW / "dc_handoff/scripts/m1091r3_m1090r3_c2_observation_authorized_engine_r1.py"
CELL = Path(
    "/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/"
    "digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/"
    "tcbn28hpcplusbwp35p140.v"
)
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
ALLOWED_SYMLINK = Path("mapped_vcs/csrc/_3721051_archive_1.so")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="strict")


def verify_sealed_dir(directory: Path, allowed_symlink: Path | None = None) -> dict:
    assert directory.is_dir() and not directory.is_symlink()
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    assert stat.S_ISREG(manifest.lstat().st_mode) and not manifest.is_symlink()
    assert stat.S_ISREG(outer.lstat().st_mode) and not outer.is_symlink()
    expected: dict[Path, str] = {}
    for line in text(manifest).splitlines():
        value, name = line.split(None, 1)
        relative = Path(name.lstrip("*"))
        assert not relative.is_absolute() and ".." not in relative.parts
        assert relative not in expected
        expected[relative] = value
    actual: set[Path] = set()
    symlinks: list[str] = []
    root = directory.resolve(strict=True)
    for member in directory.rglob("*"):
        relative = member.relative_to(directory)
        if relative in {Path("SHA256SUMS"), Path("SHA256SUMS.seal.sha256")}:
            continue
        mode = member.lstat().st_mode
        if stat.S_ISREG(mode) or stat.S_ISLNK(mode):
            actual.add(relative)
        if stat.S_ISLNK(mode):
            assert allowed_symlink is not None and relative == allowed_symlink
            resolved = member.resolve(strict=True)
            assert resolved != root and root in resolved.parents
            assert stat.S_ISREG(resolved.lstat().st_mode)
            symlinks.append(relative.as_posix())
    assert actual == set(expected)
    for relative, value in expected.items():
        member = directory / relative
        assert digest(member) == value
    assert text(outer).split() == [digest(manifest), "SHA256SUMS"]
    return {
        "manifest_members": len(expected),
        "actual_members": len(actual),
        "member_coverage_exact": True,
        "member_bytes_all_match": True,
        "symlinks": symlinks,
        "manifest_sha256": digest(manifest),
        "outer_seal_file_sha256": digest(outer),
    }


def main() -> None:
    attempt_seal = verify_sealed_dir(ATTEMPT)
    quarantine_seal = verify_sealed_dir(QUARANTINE, ALLOWED_SYMLINK)
    attempt = json.loads(text(ATTEMPT / "attempt.json"))
    failure = json.loads(text(QUARANTINE / "failure.json"))
    case_log = text(QUARANTINE / "mapped_vcs/case0.log")
    compile_log = text(QUARANTINE / "mapped_vcs/compile.log")
    tb = text(TB)
    wrapper = text(WRAPPER)
    service = text(SERVICE)
    engine = text(ENGINE)
    netlist = text(NETLIST)
    cell = text(CELL)

    match = re.search(
        r"M1090R3_FIRST_X cycle=(\d+) signal=(\S+) value=([01xz]+)", case_log
    )
    assert match
    x_value = match.group(3)
    x_bits = [31 - index for index, value in enumerate(x_value) if value.lower() == "x"]
    assert x_bits == [27, 25, 18, 13, 8, 3, 0]

    checks = re.findall(r"`M1090R3_FAIL_X\(([^)]+)\)", tb)
    first_index = checks.index(match.group(2))
    assert first_index == 11 and len(checks) == 22

    group_flops = re.findall(
        r"DFQD(?:1|2)BWP35P140\s+\S+\s*\(\s*\.D\([^)]*\),\s*"
        r"\.CP\(clk_core\),\s*\.Q\(obs_service_group_count\[(\d+)\]\)\s*\)",
        netlist,
    )
    assert sorted(map(int, group_flops)) == list(range(32))
    for token in (
        "CKND0BWP35P140 U105775 ( .I(rst_core), .ZN(n166977) )",
        "CKND0BWP35P140 U96568 ( .I(n166977), .ZN(n187524) )",
        "NR2D1BWP35P140 U121672 ( .A1(n187524), .A2(n110613), .ZN(n146196) )",
        "MAOI22D0BWP35P140 U167549",
        ".A1(obs_service_group_count[0]), .A2(n146126)",
        ".B1(n146196), .B2(obs_service_group_count[0]), .ZN(n82769)",
    ):
        assert token in netlist

    assert "obs_service_group_count=debug_group_count;" in wrapper
    assert ".debug_group_accept_count(debug_group_count)" in wrapper
    assert "assign debug_group_accept_count = group_accept_count_q;" in service
    assert service.count("group_accept_count_q <= 0;") >= 2
    assert "group_accept_count_q <= group_accept_count_q + 1'b1;" in service
    assert "always #1.5 clk_core=~clk_core;" in tb
    assert "repeat(5)@(posedge clk_core);" in tb
    assert "@(negedge clk_core);rst_core=0;header_valid=1;" in tb

    vcs_command = engine[engine.index("str(VCS), \"-full64\"") : engine.index(
        "], mapped / \"compile.log\"", engine.index("str(VCS), \"-full64\"")
    )]
    assert "UNIT_DELAY" not in vcs_command
    assert "initreg" not in vcs_command.lower()
    assert "sdf" not in vcs_command.lower()
    dfqd1 = cell[cell.index("module DFQD1BWP35P140") : cell.index(
        "module DFQD2BWP35P140"
    )]
    assert "module DFQD1BWP35P140 (D, CP, Q);" in dfqd1
    assert "(posedge CP => (Q+:D)) = (0, 0);" in dfqd1

    area = text(QUARANTINE / "dc/reports/area.rpt")
    qor = text(QUARANTINE / "dc/reports/qor.rpt")
    setup = text(QUARANTINE / "dc/reports/timing_setup.rpt")
    hold = text(QUARANTINE / "dc/reports/timing_hold_diagnostic.rpt")
    assert "Total cell area:                125766.647183" in area
    assert "Leaf Cell Count:             155228" in qor
    assert "Sequential Cell Count:        31480" in qor
    assert "Macro Count:                      0" in qor
    assert "slack (MET)                                                      0.0044" in setup
    assert "slack (VIOLATED)                                                -0.0190" in hold
    assert "No. of Hold Violations:    29442.00" in qor

    assert attempt["status"] == "M1091R3_ATTEMPT_CONSUMED_AFTER_M1093R2_M1096R2"
    assert attempt["random_initialization"] is False
    assert attempt["dc_attempts"] == 1 and attempt["mapped_cases"] == 1
    assert failure == {
        "m1080_retry": False,
        "message": "short observation window found X/stall; inspect quarantine case0.log",
        "phase": "FRESH_MAPPED_VCS_CASE0_SHORT_128",
        "status": "FAILED_DIAGNOSTIC_DO_NOT_CITE",
    }
    assert text(QUARANTINE / "dc/dc.rc").strip() == "0"
    assert text(QUARANTINE / "mapped_vcs/compile.rc").strip() == "0"
    assert text(QUARANTINE / "mapped_vcs/case0.rc").strip() == "0"
    assert "PASS_M1090R3_OBSERVATION_SHORT_WINDOW" not in case_log
    assert compile_log.count("Warning-[TFIPC]") == 3
    assert 'print("M1091r2 failure: " + str(exc)' in engine
    assert digest(DOC359) == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

    result = {
        "schema": "m1109_m1091r3_c2_observation_mapped_x_failure_mechanical_v1",
        "status": "PASS_READ_ONLY_FAILURE_LOCAL_AUDIT__M1091R3_DO_NOT_RETRY",
        "scope": {
            "quarantine_read_only": True,
            "dc_rerun": False,
            "vcs_rerun": False,
            "simv_rerun": False,
            "repair_implemented": False,
            "docs359_modified": False,
        },
        "sealed_evidence": {
            "attempt": attempt_seal,
            "quarantine": quarantine_seal,
            "attempt_json_sha256": digest(ATTEMPT / "attempt.json"),
            "failure_json_sha256": digest(QUARANTINE / "failure.json"),
        },
        "first_failure": {
            "time_ps": 16500,
            "window_cycle": int(match.group(1)),
            "signal": match.group(2),
            "value": x_value,
            "x_bit_indices": x_bits,
            "observations_known_by_order_before_first_x": checks[:first_index],
            "observations_not_evaluated_after_fatal": checks[first_index + 1 :],
            "case_return_code": 0,
            "pass_token_absent": True,
            "engine_fail_closed_on_missing_token": True,
        },
        "mapped_cone": {
            "wrapper_direct_fanout": True,
            "rtl_sync_reset_to_zero": True,
            "mapped_group_count_nonreset_dfqd_flops": 32,
            "bit0_old_q_reset_reconvergence_tokens_present": True,
            "classification": "SYNCHRONOUS_RESET_MUX_FACTORED_INTO_X_RECONVERGENT_MAPPED_D_CONE",
            "counter_constant_propagated_away": False,
        },
        "excluded_causes": {
            "unit_delay_or_sdf": True,
            "retained_payload_reset": True,
            "memory_model": True,
            "short_128_window_length": True,
        },
        "dc_failed_flow_diagnostic_only": {
            "cell_area_um2": 125766.647183,
            "leaf_cells": 155228,
            "sequential_cells": 31480,
            "macro_count": 0,
            "setup_slack_ns": 0.0044,
            "hold_slack_ns": -0.0190,
            "hold_violating_paths": 29442,
            "pre_macro_logic_only": True,
            "paper_citable": False,
        },
        "diagnostic_label_drift": {
            "severity": "P1",
            "actual_namespace": "M1091r3",
            "stderr_label": "M1091r2 failure",
            "receipt_or_branch_logic_affected": False,
        },
        "identity": {
            "case0_log_sha256": digest(QUARANTINE / "mapped_vcs/case0.log"),
            "mapped_netlist_sha256": digest(NETLIST),
            "wrapper_sha256": digest(WRAPPER),
            "service_rtl_sha256": digest(SERVICE),
            "tb_sha256": digest(TB),
            "engine_sha256": digest(ENGINE),
            "cell_model_sha256": digest(CELL),
            "docs359_sha256": digest(DOC359),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
