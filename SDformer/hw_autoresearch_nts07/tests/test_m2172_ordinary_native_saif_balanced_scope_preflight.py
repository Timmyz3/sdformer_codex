#!/opt/anaconda3/bin/python3
"""No-EDA mutation tests for the additive M2172 parser/runner source."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile


HW = Path(__file__).resolve().parents[1]
PARSER = HW / "system_simulator/scripts/parse_m2172_m2018_ordinary_native_saif_balanced_scope_preflight.py"
TB = HW / "tb_m2018/tb_m2160_m2018_ordinary_native_saif_report_reset_preflight.sv"
FILELIST = HW / "dc_handoff/filelists/tcasii_m2160_m2018_ordinary_native_saif_report_reset_preflight_vcs.f"
UCLI = HW / "dc_handoff/scripts/m2160_m2018_ordinary_native_saif_report_reset_preflight.ucli.tcl"
RUNNER = HW / "dc_handoff/scripts/run_m2172_m2018_ordinary_native_saif_balanced_scope_preflight_one_shot.py"


def load_parser():
    spec = importlib.util.spec_from_file_location("m2172_parser", PARSER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_parser()


def must_fail(callable_) -> None:
    try:
        callable_()
    except M.Failure:
        return
    raise AssertionError("mutation unexpectedly passed")


def runtime_text() -> str:
    return "\n".join([
        "M2160_UCLI_PHASE order=1 action=power_enable timing=before_first_run scope=single_ordinary_dut",
        "M2160_INTERNAL_KNOWNNESS_CENSUS phase=pre_power_reset row_live=192/192 row_live_one=149 cache_valid=4/4 cache_valid_one=0 slot_valid=8/8 slot_valid_one=0 bridge_overflow=16/16 bridge_overflow_one=0 rsp_shape_legal=8/8 rsp_shape_legal_one=8 total=228/228 observe_only=1 force=0 deposit=0 mask=0 rtl_edit=0",
        "M2160_RTL_SAIF_WINDOW_BEGIN sampling=settled_negedge global_slot=42 sample=0 layer=28 is_fc2=0 token_start=0 source_groups=48 preload_cycles=383 time_ns=1167.01 next_ucli_action=disable_report_prehistory_then_reset",
        "M2160_UCLI_PHASE order=2 action=first_run_returned census_and_begin_preceded_return=1",
        "M2160_UCLI_PHASE order=3 action=prehistory_power_disable timing=before_diagnostic_report",
        "M2160_UCLI_PHASE order=4 action=prehistory_power_report scope=single_ordinary_dut diagnostic_only=1",
        "M2160_UCLI_PHASE order=5 action=power_reset_requested timing=after_prehistory_report_before_measurement_enable",
        "M2160_UCLI_PHASE order=6 action=measurement_power_enable timing=after_reset_before_second_run",
        "M2160_RTL_SAIF_WINDOW_END axis=ordinary_lru4 sampling=settled_negedge measurement_cycles=20292 rows=149 issues=1278 products=29472 commits=24 bundles=1788 scalar_weight_reads=14304 duration_ns=60876.00",
        "PASS_M2160_ORDINARY_SINGLE_AXIS_NATIVE_SAIF_PREFLIGHT ledger_exact=1 arithmetic_scoreboard_exact=1 internal_census_exact=1 enable_before_reset_preload=1 prehistory_report_requested=1 power_reset_requested=1 frontends=1 schedule_mode=0 second_axis=0 initreg_diagnostic_only=1 paper_citable=0",
        "M2160_UCLI_PHASE order=7 action=second_run_returned end_and_pass_preceded_return=1",
        "M2160_UCLI_PHASE order=8 action=measurement_power_disable timing=before_measurement_report",
        "M2160_UCLI_PHASE order=9 action=measurement_power_report scope=single_ordinary_dut admitted_candidate=1",
        "",
    ])


def activity_rows(duration: float, *, count: int, tx_first: int = 0,
                  mute_first_critical: bool = False,
                  names: list[str] | None = None) -> list[str]:
    names = list(M.CRITICAL) if names is None else names
    rows: list[str] = []
    for index in range(count):
        name = names[index] if index < len(names) else f"filler_{index}"
        tx = tx_first if index == 0 else 0
        tc = 0 if mute_first_critical and index == 0 else 2
        rows.append(
            f"({name} (T0 {duration - 1 - tx:.2f}) (T1 1) "
            f"(TX {tx}) (TC {tc}) (IG 0))")
    return rows


def saif_text(*, role: str, count: int, target: str = "dut_ordinary",
              tx_first: int = 0, duration: float | None = None,
              mute_first_critical: bool = False,
              records_outside: bool = False, duplicate_target: bool = False,
              empty_target: bool = False) -> str:
    if duration is None:
        duration = 60876 if role == "measurement" else 1167.01
    names = list(M.CRITICAL) if role == "measurement" else ["load_valid"]
    rows = activity_rows(duration, count=count, tx_first=tx_first,
                         mute_first_critical=mute_first_critical, names=names)
    target_body = [] if empty_target else ["(INSTANCE implementation", "(NET", *rows, ")", ")"]
    root_rows = rows if records_outside else []
    duplicate = ["(INSTANCE dut_ordinary)"] if duplicate_target else []
    return "\n".join([
        "/** legal Synopsys comment before the SAIF root **/",
        "(SAIFILE", '(SAIFVERSION "2.0")', "(TIMESCALE 1 ns)",
        f"(DURATION {duration})", f"(INSTANCE {target}", *target_body, ")",
        *duplicate, *root_rows, ")", "",
    ])


def seal_file(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n")
    outer_digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    Path(str(sidecar) + ".seal.sha256").write_text(
        f"{outer_digest}  {sidecar.name}\n")


def main() -> None:
    assert M.static_check()["status"] == "PASS_M2172_STATIC_PARSER"
    tb, filelist, ucli, runner = (path.read_text() for path in
                                  (TB, FILELIST, UCLI, RUNNER))
    topology = M.audit_single_axis_source(tb, filelist)
    assert topology["direct_m2018_frontends"] == 1
    assert topology["second_axis_symbols"] == 0
    assert "power -report" in ucli and "power -reset" in ucli
    assert runner.count("BASE.production") == 0
    assert "M2174" in runner and "M2173" in runner and "M2175" in runner

    topology_mutations = {
        "parent_tb": tb + "\ntb_m2051 injected_parent();\n",
        "second_dut": tb + "\nlogic dut_tsbg;\n",
        "second_valid": tb + "\nlogic load_valid_tsbg;\n",
        "mode_one": tb.replace(".SCHEDULE_MODE(0)", ".SCHEDULE_MODE(1)"),
        "second_wait": tb + "\ninitial wait (tsbg_done_cycle >= 0);\n",
        "second_path": tb + "\ninitial if (tsbg.busy) $fatal;\n",
    }
    for mutated in topology_mutations.values():
        must_fail(lambda mutated=mutated:
                  M.audit_single_axis_source(mutated, filelist))
    must_fail(lambda: M.audit_single_axis_source(
        tb, filelist + "hw_autoresearch_nts07/tb_m2018/tb_m2051_ep34_tsbg_full40_cycle.sv\n"))
    must_fail(lambda: M.audit_single_axis_source(
        tb, filelist + "hw_autoresearch_nts07/rtl_m2020/m2020_m2018_vcs_public_name_adapter.sv\n"))

    with tempfile.TemporaryDirectory(prefix="m2172_source_test.") as raw:
        root = Path(raw)
        runtime = root / "rtl_sim.log"
        prehistory = root / "rtl_prehistory.saif"
        measurement = root / "rtl_measurement.saif"
        runtime.write_text(runtime_text())
        # Keep source tests fast while the production parser remains frozen at 93,971.
        M.EXPECTED["records"] = 32
        prehistory.write_text(saif_text(role="diagnostic_prehistory", count=32))
        measurement.write_text(saif_text(role="measurement", count=32))
        seal_file(prehistory)
        seal_file(measurement)
        assert M.parse_runtime(runtime)["completion_ledger"]["products"] == 29472
        assert M.parse_saif(prehistory, role="diagnostic_prehistory")["record_count"] == 32
        parsed = M.parse_saif(measurement, role="measurement")
        assert parsed["record_count"] == 32 and parsed["balanced_hierarchy"]
        assert parsed["outside_target_record_count"] == 0
        result = M.final_result(root, root / "result.json")
        assert result["claim_boundary"]["paper_citable"] is False

        runtime_mutations = [
            runtime_text().replace("order=5 action=power_reset_requested",
                                   "order=5 action=power_reset_requested_BAD"),
            runtime_text().replace("row_live=192/192", "row_live=191/192"),
            runtime_text().replace("products=29472", "products=29471"),
            runtime_text().replace("second_axis=0", "second_axis=1"),
            runtime_text() + "Warning-[SAIF_REPORT_BEFORE_RESET] Toggle reporting not done\n",
            runtime_text() + "This request to reset power information will be ignored.\n",
            runtime_text() + "Warning: This reset request was ignored.\n",
            runtime_text() + "Warning: Power information reset request ignored.\n",
            runtime_text() + "Warning-[POWER_RESET_IGNORED] Switching counters were not cleared.\n",
            runtime_text() + "Warning: request to reset switching activity has been ignored.\n",
            runtime_text() + "Error: switching activity counters remain uncleared.\n",
        ]
        for mutation in runtime_mutations:
            runtime.write_text(mutation)
            must_fail(lambda: M.parse_runtime(runtime))
        runtime.write_text(runtime_text())

        mutations = [
            saif_text(role="measurement", count=32, tx_first=1),
            saif_text(role="measurement", count=32, duration=60877),
            saif_text(role="measurement", count=31),
            saif_text(role="measurement", count=32, mute_first_critical=True),
            saif_text(role="measurement", count=32, target="impostor"),
            saif_text(role="measurement", count=32, duplicate_target=True),
            saif_text(role="measurement", count=32, records_outside=True,
                      empty_target=True),
            saif_text(role="measurement", count=32)[:-2],
            saif_text(role="measurement", count=32) + ")\n",
        ]
        for mutation in mutations:
            measurement.write_text(mutation)
            seal_file(measurement)
            must_fail(lambda: M.parse_saif(measurement, role="measurement"))

        measurement.write_text(saif_text(role="measurement", count=32))
        seal_file(measurement)
        text = measurement.read_text().replace("(T0 60875.00)", "(T0 60874.00)", 1)
        measurement.write_text(text)
        seal_file(measurement)
        must_fail(lambda: M.parse_saif(measurement, role="measurement"))
        measurement.write_text(saif_text(role="measurement", count=32))
        seal_file(measurement)
        Path(str(measurement) + ".sha256").write_text(
            "0" * 64 + f"  {measurement.name}\n")
        must_fail(lambda: M.parse_saif(measurement, role="measurement"))
        prehistory.write_text(saif_text(role="diagnostic_prehistory", count=32,
                                        duration=1168.01))
        seal_file(prehistory)
        must_fail(lambda: M.parse_saif(prehistory, role="diagnostic_prehistory"))

    print("PASS_M2172_SOURCE_TESTS tests=42 topology_mutations=8 "
          "runtime_mutations=11 saif_mutations=12 eda_runs=0")


if __name__ == "__main__":
    main()
