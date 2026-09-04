#!/opt/anaconda3/bin/python3
"""No-EDA source, topology, mutation, runtime, and SAIF tests for M2149."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile


HW = Path(__file__).resolve().parents[1]
PARSER = HW / "system_simulator/scripts/parse_m2149_m2018_ordinary_single_axis_native_saif_preflight.py"
TB = HW / "tb_m2018/tb_m2149_m2018_ordinary_single_axis_native_saif_preflight.sv"
FILELIST = HW / "dc_handoff/filelists/tcasii_m2149_m2018_ordinary_single_axis_native_saif_preflight_vcs.f"
UCLI = HW / "dc_handoff/scripts/m2149_m2018_ordinary_single_axis_native_saif_preflight.ucli.tcl"


def load_parser():
    spec = importlib.util.spec_from_file_location("m2149_parser", PARSER)
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
        "M2149_UCLI_PHASE order=1 action=power_enable timing=before_first_run scope=single_ordinary_dut",
        "M2149_UCLI_PHASE order=2 action=run_reset_and_preload observer_enabled=1",
        "M2149_INTERNAL_KNOWNNESS_CENSUS phase=pre_power_reset row_live=192/192 row_live_one=149 cache_valid=4/4 cache_valid_one=0 slot_valid=8/8 slot_valid_one=0 bridge_overflow=16/16 bridge_overflow_one=0 rsp_shape_legal=8/8 rsp_shape_legal_one=8 total=228/228 observe_only=1 force=0 deposit=0 mask=0 rtl_edit=0",
        "M2149_RTL_SAIF_WINDOW_BEGIN sampling=settled_negedge global_slot=42 sample=0 layer=28 is_fc2=0 token_start=0 source_groups=48 preload_cycles=383 time_ns=1153.51 next_ucli_action=power_reset",
        "M2149_UCLI_PHASE order=3 action=first_stop_reached internal_census_preceded_stop=1",
        "M2149_UCLI_PHASE order=4 action=power_reset timing=after_first_stop_before_measurement_run",
        "M2149_RTL_SAIF_WINDOW_END axis=ordinary_lru4 sampling=settled_negedge measurement_cycles=20292 rows=149 issues=1278 products=29472 commits=24 bundles=1788 scalar_weight_reads=14304 duration_ns=60876.00",
        "PASS_M2149_ORDINARY_SINGLE_AXIS_NATIVE_SAIF_PREFLIGHT ledger_exact=1 arithmetic_scoreboard_exact=1 internal_census_exact=1 enable_before_reset_preload=1 power_reset_at_first_stop=1 frontends=1 schedule_mode=0 second_axis=0 initreg_diagnostic_only=1 paper_citable=0",
        "M2149_UCLI_PHASE order=5 action=second_stop_reached exact_window_complete=1",
        "M2149_UCLI_PHASE order=6 action=power_disable timing=before_report",
        "M2149_UCLI_PHASE order=7 action=power_report scope=single_ordinary_dut",
        "",
    ])


def saif_text(*, tx_first: int = 0, duration: int = 60876,
              records: int = 93971, mute_first_critical: bool = False) -> str:
    names = list(M.CRITICAL)
    rows = []
    for index in range(records):
        name = names[index] if index < len(names) else f"filler_{index}"
        tx = tx_first if index == 0 else 0
        tc = 0 if mute_first_critical and index == 0 else 2
        rows.append(
            f"({name} (T0 {duration - 1 - tx}) (T1 1) (TX {tx}) (TC {tc}))")
    return "\n".join([
        "(SAIFILE", "(TIMESCALE 1 ns)", f"(DURATION {duration})",
        *rows, ")", "",
    ])


def main() -> None:
    assert M.static_check()["status"] == "PASS_M2149_STATIC_PARSER"
    tb = TB.read_text()
    filelist = FILELIST.read_text()
    ucli = UCLI.read_text()
    topology = M.audit_single_axis_source(tb, filelist)
    assert topology["direct_m2018_frontends"] == 1
    assert topology["second_axis_symbols"] == 0
    assert "force " not in "\n".join(
        line for line in tb.splitlines() if not line.lstrip().startswith("//"))
    assert "deposit " not in "\n".join(
        line for line in tb.splitlines() if not line.lstrip().startswith("//"))
    assert "power -enable\n" in ucli
    assert ucli.index("power -enable") < ucli.index("run\n")
    assert ucli.index("run\n") < ucli.index("power -reset") < ucli.rindex("run\n")
    assert ucli.count("power tb_m2149") == 1
    assert ucli.count("power -report") == 1

    # Dedicated topology mutations: each of the exact M2143 escape classes
    # must fail before any future license query or compilation.
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

    with tempfile.TemporaryDirectory(prefix="m2149_static_test.") as raw:
        root = Path(raw)
        runtime = root / "rtl_sim.log"
        saif = root / "rtl_execute.saif"
        runtime.write_text(runtime_text())
        saif.write_text(saif_text())
        parsed_runtime = M.parse_runtime(runtime)
        assert parsed_runtime["completion_ledger"]["products"] == 29472
        assert parsed_runtime["second_axis_executed"] is False
        assert M.parse_saif(saif)["record_count"] == 93971
        result = M.final_result(root, root / "result.json")
        assert result["claim_boundary"]["paper_citable"] is False
        assert result["claim_boundary"]["second_axis_run"] is False

        runtime.write_text(runtime_text().replace(
            "order=4 action=power_reset", "order=4 action=power_reset_BAD"))
        must_fail(lambda: M.parse_runtime(runtime))
        runtime.write_text(runtime_text().replace(
            "row_live=192/192", "row_live=191/192"))
        must_fail(lambda: M.parse_runtime(runtime))
        runtime.write_text(runtime_text().replace(
            "products=29472", "products=29471"))
        must_fail(lambda: M.parse_runtime(runtime))
        runtime.write_text(runtime_text().replace(
            "second_axis=0", "second_axis=1"))
        must_fail(lambda: M.parse_runtime(runtime))

        saif.write_text(saif_text(tx_first=1))
        must_fail(lambda: M.parse_saif(saif))
        saif.write_text(saif_text(duration=60877))
        must_fail(lambda: M.parse_saif(saif))
        saif.write_text(saif_text(records=93970))
        must_fail(lambda: M.parse_saif(saif))
        saif.write_text(saif_text(mute_first_critical=True))
        must_fail(lambda: M.parse_saif(saif))

    print("PASS_M2149_SOURCE_TESTS tests=24 topology_mutations=8 eda_runs=0")


if __name__ == "__main__":
    main()
