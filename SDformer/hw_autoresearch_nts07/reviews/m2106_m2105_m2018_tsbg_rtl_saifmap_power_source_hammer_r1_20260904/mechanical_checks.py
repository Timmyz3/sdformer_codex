#!/usr/bin/env python3
"""Read-only mechanical hammer for the frozen M2105 source bundle.

The test creates only temporary parser fixtures under /tmp.  It never invokes
VCS, simv, DC, PrimeTime, lmstat, or a GPU tool.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
REPO = HW.parent
RUNNER = HW / "dc_handoff/scripts/run_m2105_m2018_tsbg_rtl_saifmap_power_one_shot.py"
PARSER = HW / "system_simulator/scripts/parse_m2105_m2018_tsbg_rtl_saifmap_power.py"
CONTRACT = HW / "contracts/m2105_m2018_tsbg_rtl_saifmap_power_source_contract_r1_20260904.json"
VCS_F = HW / "dc_handoff/filelists/tcasii_m2105_m2018_tsbg_rtl_saif_vcs.f"
DC_F = HW / "dc_handoff/filelists/tcasii_m2105_m2018_tsbg_saifmap_dc.f"
DC_TCL = HW / "dc_handoff/scripts/run_dc_m2105_m2018_tsbg_saifmap_axis.tcl"
PT_TCL = HW / "dc_handoff/scripts/run_ptpx_m2105_m2018_tsbg_rtl_saifmap_axis.tcl"
TB = HW / "tb_m2018/tb_m2105_m2018_tsbg_rtl_saifmap_power.sv"
UCLI_ORD = HW / "dc_handoff/scripts/m2105_m2018_tsbg_ordinary_rtl_saif.ucli.tcl"
UCLI_TSBG = HW / "dc_handoff/scripts/m2105_m2018_tsbg_tsbg_rtl_saif.ucli.tcl"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(value: bool, label: str, checks: dict[str, bool]) -> None:
    checks[label] = bool(value)
    if not value:
        raise AssertionError(label)


def strict_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise AssertionError(f"duplicate key {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          AssertionError(f"nonfinite {token}")))


def run_static(path: Path, *args: str) -> dict:
    completed = subprocess.run(
        ["python3.12", str(path), *args], cwd=REPO,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=30, check=False)
    if completed.returncode != 0:
        raise AssertionError(completed.stderr.strip())
    return json.loads(completed.stdout)


def write_saif(path: Path, duration: float, tx_bad: bool = False) -> None:
    critical = (
        "mem_req_valid", "mem_rsp_valid", "bridge_valid", "commit_valid",
        "mem_req_accept", "mem_rsp_accept", "bridge_accept", "commit_accept",
    )
    names = list(critical) + [f"signal_{index}" for index in range(92)]
    rows = []
    for index, name in enumerate(names):
        tx = 1.0 if tx_bad and index == 99 else 0.0
        t0 = duration / 2.0 - tx
        t1 = duration / 2.0
        rows.append(
            f"({name} (T0 {t0}) (T1 {t1}) (TX {tx}) (TC {1 + index % 3}))")
    path.write_text(
        "(SAIFILE\n(TIMESCALE 1 ns)\n"
        f"(DURATION {duration})\n(INSTANCE dut\n(NET\n"
        + "\n".join(rows) + "\n)))\n")


def expect_failure(callback, label: str, checks: dict[str, bool]) -> None:
    try:
        callback()
    except Exception:
        checks[label] = True
        return
    checks[label] = False
    raise AssertionError(label)


def main() -> None:
    checks: dict[str, bool] = {}
    frozen_before = {path.as_posix(): sha(path) for path in (
        RUNNER, PARSER, CONTRACT, VCS_F, DC_F, DC_TCL, PT_TCL, TB,
        UCLI_ORD, UCLI_TSBG, DOC359)}

    contract = strict_json(CONTRACT)
    inventory = contract["source_inventory"]
    need(len(inventory) == 19, "source_inventory_has_19_members", checks)
    need(all(sha(REPO / rel) == digest for rel, digest in inventory.items()),
         "source_inventory_19_of_19_hashes_match", checks)
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [sha(CONTRACT), CONTRACT.name],
         "contract_inner_seal_matches", checks)
    need(outer.read_text().split() == [sha(sidecar), sidecar.name],
         "contract_outer_seal_matches", checks)
    need(sha(DOC359) ==
         "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
         "docs359_identity_preserved", checks)

    runner_static = run_static(RUNNER, "--static")
    parser_static = run_static(PARSER, "static")
    need(runner_static["status"] == "PASS_M2105_STATIC_RUNNER",
         "runner_static_pass", checks)
    need(parser_static["status"] == "PASS_M2105_STATIC_PARSER"
         and all(parser_static["checks"].values()), "parser_static_pass", checks)

    expected_budget = {
        "license_queries": 1, "vcs_compiles": 1, "simv_runs": 2,
        "dc_runs": 2, "ptpx_runs": 2, "saif_files": 2,
        "automatic_retry": False, "p1_serial": True,
        "reuse_old_artifacts": False,
    }
    need(contract["execution_budget"] == expected_budget,
         "one_shot_budget_exact", checks)
    runner_text = RUNNER.read_text()
    need("source_validation(require_review=True)" in runner_text
         and "ATTEMPT.mkdir()" in runner_text
         and runner_text.index("ATTEMPT.mkdir()") < runner_text.index(
             "run([str(LMUTIL), \"lmstat\"")
         and '"automatic_retry": False' in runner_text,
         "review_and_attempt_consumed_before_license", checks)
    need("for axis, cfg in AXES.items():" in runner_text
         and runner_text.count('counts["dc_runs"] += 1') == 1
         and runner_text.count('counts["ptpx_runs"] += 1') == 1,
         "two_axes_execute_serially", checks)

    vcs_sources = [line for line in VCS_F.read_text().splitlines() if line.strip()]
    dc_sources = [line for line in DC_F.read_text().splitlines() if line.strip()]
    need(len(vcs_sources) == 6 and vcs_sources[-1].endswith(
         "tb_m2105_m2018_tsbg_rtl_saifmap_power.sv"),
         "vcs_filelist_closed", checks)
    need(dc_sources == [
        "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
        "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"],
        "dc_filelist_is_dut_only", checks)

    tb = TB.read_text()
    for token in (
        "FROZEN_WORKLOAD_SLOT = 42", "FROZEN_PRELOAD_CYCLES = 383",
        "FROZEN_BASE_CYCLES = 20292", "FROZEN_TSBG_CYCLES = 7569",
        "FROZEN_BASE_SCALAR = 14304", "FROZEN_TSBG_SCALAR = 4608",
        "core.sample_id != 0", "core.layer_id != 28", "core.is_fc2 != 0",
        "core.token_start != 0", "core.real_source_groups != 48",
        "requires exactly one axis plusarg", "completion ledger drift",
    ):
        need(token in tb, f"tb_contract_{token}", checks)

    ordinary_ucli = UCLI_ORD.read_text()
    tsbg_ucli = UCLI_TSBG.read_text()
    need("core.dut_base.implementation" in ordinary_ucli
         and "core.dut_tsbg.implementation" not in ordinary_ucli,
         "ordinary_dut_only_saif_scope", checks)
    need("core.dut_tsbg.implementation" in tsbg_ucli
         and "core.dut_base.implementation" not in tsbg_ucli,
         "tsbg_dut_only_saif_scope", checks)
    need(all(text.count("power -enable") == 1
             and text.count("power -disable") == 1
             and "M2105_RTL_SAIF_FILE" in text
             for text in (ordinary_ucli, tsbg_ucli)),
         "ucli_single_measurement_window", checks)

    dc = DC_TCL.read_text()
    need('elaborate $design_name -parameters "SCHEDULE_MODE=>$mode"' in dc
         and 'if {$mode ne "0" && $mode ne "1"}' in dc,
         "dc_schedule_mode_is_only_axis_parameter", checks)
    need("saif_map -start" in dc and "saif_map -write_map" in dc
         and "-type ptpx -essential" in dc,
         "dc_native_saif_map_default_and_essential", checks)

    pt = PT_TCL.read_text()
    need(pt.index("source $default_map") < pt.index("source $essential_map")
         < pt.index("read_saif"), "pt_map_source_order_closed", checks)
    first_power = pt.index("\nreport_power")
    for token in (
        "M2105_FAIL_ANNOTATION_GATE_BEFORE_POWER",
        "M2105_FAIL_NONZERO_TOGGLE_COVERAGE_BEFORE_POWER",
        "M2105_FAIL_INCONSISTENT_ANNOTATION_BEFORE_POWER",
        "M2105_FAIL_ZERO_CRITICAL_CONE_BEFORE_POWER",
        "M2105_FAIL_CHECK_POWER",
    ):
        need(token in pt and pt.index(token) < first_power,
             f"pt_prepower_gate_{token}", checks)
    need("weight_sram_capacity_bytes=294912" in pt
         and "weight_sram_dynamic_energy_in_ptpx=false" in pt
         and "weight_sram_area_in_ptpx=false" in pt,
         "external_288kib_sram_disclosed_separately", checks)

    spec = importlib.util.spec_from_file_location("m2105_parser", PARSER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="m2106_parser_hammer.") as tmp_name:
        tmp = Path(tmp_name)
        valid_saif = tmp / "valid.saif"
        write_saif(valid_saif, 20292 * 3.0)
        parsed = module.parse_saif(valid_saif, "ordinary_lru4")
        need(parsed["tx_sum"] == 0.0 and parsed["record_count"] == 100,
             "synthetic_valid_saif_pass", checks)
        bad_tx = tmp / "bad_tx.saif"
        write_saif(bad_tx, 20292 * 3.0, tx_bad=True)
        expect_failure(lambda: module.parse_saif(bad_tx, "ordinary_lru4"),
                       "synthetic_tx_nonzero_rejected", checks)
        bad_duration = tmp / "bad_duration.saif"
        write_saif(bad_duration, 20292 * 3.0 + 1.0)
        expect_failure(lambda: module.parse_saif(
            bad_duration, "ordinary_lru4"),
            "synthetic_duration_drift_rejected", checks)

        default_map = tmp / "default.tcl"
        essential_map = tmp / "essential.tcl"
        default_map.write_text(
            "set_rtl_to_gate_name -rtl {state_q} -gate U0\n"
            "set_rtl_to_gate_name -rtl {count_q} -gate U1\n")
        essential_map.write_text(
            "set_rtl_to_gate_name -rtl {state_q} -gate U0/Q\n"
            "set_rtl_to_gate_name -rtl {valid_q} -gate U2/Q\n")
        maps = module.classify_maps(default_map, essential_map)
        need(maps["intersection_entries"] == 1
             and maps["union_entries"] == 3
             and maps["intersection_target_difference_entries"] == 1,
             "map_intersection_union_and_difference_preserved", checks)
        conflict_map = tmp / "conflict.tcl"
        conflict_map.write_text(
            "set_rtl_to_gate_name -rtl {state_q} -gate U0\n"
            "set_rtl_to_gate_name -rtl {state_q} -gate U9\n")
        expect_failure(lambda: module.map_rows(conflict_map),
                       "intra_class_map_conflict_rejected", checks)

        power = tmp / "power.rpt"
        power.write_text(
            "Net Switching Power = 1.00000000\n"
            "Cell Internal Power = 2.00000000\n"
            "Cell Leakage Power = 0.50000000\n"
            "Total Power = 3.50000000\n")
        parsed_power = module.parse_power(power, 30.0)
        need(abs(parsed_power["energy_nj"] - 0.105) < 1e-12,
             "power_component_and_energy_arithmetic_pass", checks)
        power.write_text(power.read_text().replace(
            "Total Power = 3.50000000", "Total Power = 9.50000000"))
        expect_failure(lambda: module.parse_power(power, 30.0),
                       "power_component_mismatch_rejected", checks)

    frozen_after = {path.as_posix(): sha(path) for path in (
        RUNNER, PARSER, CONTRACT, VCS_F, DC_F, DC_TCL, PT_TCL, TB,
        UCLI_ORD, UCLI_TSBG, DOC359)}
    need(frozen_before == frozen_after, "source_identity_stable_during_hammer", checks)
    output = {
        "schema": "m2106_m2105_source_mechanical_checks_r1_v1",
        "status": "PASS_M2106_MECHANICAL_CHECKS_NO_EDA",
        "eda_invoked": False,
        "check_count": len(checks),
        "checks": checks,
        "source_sha256": frozen_after,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
