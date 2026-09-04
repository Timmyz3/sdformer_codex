#!/opt/anaconda3/bin/python3
"""No-EDA source and parser tests for the M2142 acquisition preflight."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile


HW = Path(__file__).resolve().parents[1]
PARSER = HW / "system_simulator/scripts/parse_m2142_m2018_tsbg_ordinary_late_enable_saif_preflight.py"
TB = HW / "tb_m2018/tb_m2142_m2018_tsbg_ordinary_late_enable_saif_preflight.sv"
UCLI = HW / "dc_handoff/scripts/m2142_m2018_tsbg_ordinary_late_enable_saif_preflight.ucli.tcl"


def load_parser():
    spec = importlib.util.spec_from_file_location("m2142_parser", PARSER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_parser()


def runtime_text() -> str:
    return "\n".join([
        "M2142_UCLI_PHASE order=1 action=power_enable timing=before_first_run scope=ordinary_implementation",
        "M2142_UCLI_PHASE order=2 action=run_reset_and_preload observer_enabled=1",
        "M2142_INTERNAL_KNOWNNESS_CENSUS phase=pre_power_reset row_live=192/192 row_live_one=149 cache_valid=4/4 cache_valid_one=0 slot_valid=8/8 slot_valid_one=0 bridge_overflow=16/16 bridge_overflow_one=0 rsp_shape_legal=8/8 rsp_shape_legal_one=8 total=228/228 observe_only=1 force=0 deposit=0 mask=0 rtl_edit=0",
        "M2142_RTL_SAIF_WINDOW_BEGIN sampling=settled_negedge global_slot=42 sample=0 layer=28 is_fc2=0 token_start=0 source_groups=48 preload_cycles=383 time_ns=1153.51 next_ucli_action=power_reset",
        "M2142_UCLI_PHASE order=3 action=first_stop_reached internal_census_preceded_stop=1",
        "M2142_UCLI_PHASE order=4 action=power_reset timing=after_first_stop_before_measurement_run",
        "M2142_RTL_SAIF_WINDOW_END axis=ordinary_lru4 sampling=settled_negedge measurement_cycles=20292 scalar_weight_reads=14304 duration_ns=60876.00",
        "PASS_M2142_ORDINARY_LATE_ENABLE_SAIF_PREFLIGHT ledger_exact=1 internal_census_exact=1 enable_before_reset_preload=1 power_reset_at_first_stop=1 initreg_diagnostic_only=1 paper_citable=0",
        "M2142_UCLI_PHASE order=5 action=second_stop_reached exact_window_complete=1",
        "M2142_UCLI_PHASE order=6 action=power_disable timing=before_report",
        "M2142_UCLI_PHASE order=7 action=power_report scope=ordinary_implementation",
        "",
    ])


def saif_text(*, tx_first: int = 0, duration: int = 60876,
              records: int = M.EXPECTED_RECORDS) -> str:
    names = list(M.CRITICAL)
    rows = []
    for index in range(records):
        name = names[index] if index < len(names) else f"filler_{index}"
        tx = tx_first if index == 0 else 0
        rows.append(
            f"({name} (T0 {duration - 1 - tx}) (T1 1) (TX {tx}) (TC 2))")
    return "\n".join([
        "(SAIFILE", "(TIMESCALE 1 ns)", f"(DURATION {duration})",
        *rows, ")", "",
    ])


def must_fail(callable_) -> None:
    try:
        callable_()
    except M.Failure:
        return
    raise AssertionError("mutation unexpectedly passed")


def main() -> None:
    assert M.static_check()["status"] == "PASS_M2142_STATIC_PARSER"
    tb = TB.read_text()
    ucli = UCLI.read_text()
    assert "force " not in "\n".join(
        line for line in tb.splitlines() if not line.lstrip().startswith("//"))
    assert "deposit " not in "\n".join(
        line for line in tb.splitlines() if not line.lstrip().startswith("//"))
    assert "power -enable\n" in ucli and ucli.index("power -enable") < ucli.index("run\n")
    assert ucli.index("run\n") < ucli.index("power -reset") < ucli.rindex("run\n")
    assert "power -disable" in ucli and "power -report" in ucli

    with tempfile.TemporaryDirectory(prefix="m2142_static_test.") as raw:
        root = Path(raw)
        runtime = root / "rtl_sim.log"
        saif = root / "rtl_execute.saif"
        runtime.write_text(runtime_text())
        saif.write_text(saif_text())
        assert M.parse_runtime(runtime)["internal_knownness"]["total"] == 228
        assert M.parse_saif(saif)["record_count"] == M.EXPECTED_RECORDS
        result = M.final_result(root, root / "result.json")
        assert result["claim_boundary"]["paper_citable"] is False

        runtime.write_text(runtime_text().replace(
            "order=4 action=power_reset", "order=4 action=power_reset_BAD"))
        must_fail(lambda: M.parse_runtime(runtime))
        runtime.write_text(runtime_text().replace("row_live=192/192", "row_live=191/192"))
        must_fail(lambda: M.parse_runtime(runtime))

        saif.write_text(saif_text(tx_first=1))
        must_fail(lambda: M.parse_saif(saif))
        saif.write_text(saif_text(duration=60877))
        must_fail(lambda: M.parse_saif(saif))
        saif.write_text(saif_text(records=M.EXPECTED_RECORDS - 1))
        must_fail(lambda: M.parse_saif(saif))

    print("PASS_M2142_SOURCE_TESTS tests=10 eda_runs=0")


if __name__ == "__main__":
    main()
