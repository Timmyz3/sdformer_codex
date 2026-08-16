from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "dc_handoff" / "scripts"


def load_function(script: str, function: str):
    spec = importlib.util.spec_from_file_location(script, SCRIPTS / script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function)


def run_script(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def write_bound_saif_manifest(tmp_path: Path, saif: Path) -> Path:
    source_vcd = tmp_path / "source.vcd"
    source_vcd.write_text("$timescale 1ps $end\n#1\n", encoding="utf-8")
    trace_root = tmp_path / "trace.input"
    trace_root.write_text("real trace\n", encoding="utf-8")
    contract = {
        "status": "PASS",
        "design_name": "h67_rqtb2s_mssb5_dc_top",
        "source_vcd": str(source_vcd),
        "source_vcd_sha256": hashlib.sha256(source_vcd.read_bytes()).hexdigest(),
        "trace_root": str(trace_root),
        "trace_sha256": hashlib.sha256(trace_root.read_bytes()).hexdigest(),
        "simulator": "verilator",
        "strip_path": "tb/dut",
        "warmup_cycles": 100,
        "measured_cycles": 1000,
        "busy_cycles": 800,
        "measurement_overhead_cycles": 200,
        "measurement_scope": "fair_lfsr_row_execution",
        "activity_purpose": "paper_power_compute",
        "paper_power_eligible": True,
        "workload_kind": "motion_row",
        "trace_scope": "sample0/window0/138 rows",
    }
    contract_path = tmp_path / "activity_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    manifest = dict(contract)
    manifest.pop("status")
    manifest["saif_sha256"] = hashlib.sha256(saif.read_bytes()).hexdigest()
    manifest["activity_contract"] = str(contract_path)
    manifest["identity_root"] = str(tmp_path.resolve())
    manifest["activity_contract_sha256"] = hashlib.sha256(
        contract_path.read_bytes()
    ).hexdigest()
    manifest_path = tmp_path / "trace.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_switching_coverage_is_fail_closed() -> None:
    parse = load_function("audit_synopsys_postrun.py", "switching_coverage")
    assert parse("SAIF annotation coverage: 97.25%") == 97.25
    assert parse("no percentage in this tool version") is None
    assert parse("Coverage 99.0%\nAnnotated pin coverage 91.5%") is None
    assert parse("Clock gating coverage 0%\nSAIF annotation coverage: 96.0%") == 96.0
    assert parse("SAIF annotation coverage: 100%\nSAIF annotation coverage: 90%") is None


def test_unannotated_objects_are_fail_closed() -> None:
    parse = load_function("audit_synopsys_postrun.py", "unannotated_object_count")
    assert parse("Total number of unannotated objects: 0") == 0
    assert parse("Total number of unannotated nets: 2\nNumber of pins not annotated: 3") == 5
    assert parse("No unannotated objects.") == 0
    assert parse("clean report with no explicit total") is None


def test_paper_population_contract_is_workload_specific() -> None:
    admit = load_function("report_activity_vcd.py", "paper_population_contract")
    fixed_rows = [
        {"slots": 450, "equal": 0, "emitted": 1} for _ in range(138)
    ]
    for row in fixed_rows[:124]:
        row["equal"] = 225
    fixed_rows[124]["equal"] = 101
    eligible, checks, totals = admit(
        "h67_fixed2s_mssb5_dc_top",
        0,
        138,
        "motion_row",
        fixed_rows,
        112589,
        "fair_lfsr_row_execution",
    )
    assert eligible
    assert checks["frozen_slot_total"]
    assert totals["slots"] == 62100
    eligible, _, _ = admit(
        "h67_fixed2s_mssb5_dc_top",
        0,
        137,
        "motion_row",
        fixed_rows[:137],
        112589,
        "fair_lfsr_row_execution",
    )
    assert not eligible
    local_rows = [
        {"score_service": 1 if index < 30 else 0} for index in range(100)
    ]
    eligible, checks, totals = admit(
        "local5_unified_out2_dc_top",
        0,
        100,
        "local5_group",
        local_rows,
        155791,
        "full_load_compute_readback",
    )
    assert eligible
    assert checks["at_least_30_nontrivial_groups"]
    assert totals["nontrivial_groups"] == 30
    eligible, checks, _ = admit(
        "local5_unified_out2_1rw_dc_top",
        0,
        100,
        "local5_group",
        local_rows,
        170269,
        "busy_projection",
    )
    assert eligible
    assert checks["matched_tile_scope"]
    local_rows[29]["score_service"] = 0
    eligible, _, _ = admit(
        "local5_unified_out2_dc_top",
        0,
        100,
        "local5_group",
        local_rows,
        155791,
        "full_load_compute_readback",
    )
    assert not eligible


def test_activity_contract_checks_vcd_time_and_log_integrity(tmp_path: Path) -> None:
    vcd = tmp_path / "trace.vcd"
    vcd.write_text(
        "$timescale 1ps $end\n"
        "$scope module TOP $end\n$scope module tb $end\n$scope module dut $end\n"
        "$var wire 1 ! dump_active $end\n$enddefinitions $end\n"
        "#0\n1!\n#20000\n0!\n",
        encoding="utf-8",
    )
    trace = tmp_path / "trace.input"
    trace.write_text("real trace\n", encoding="utf-8")
    log = tmp_path / "sim.log"
    base = (
        "MOTION_ACTIVITY_ROW mode=fixed row=0 cycles=1 slots=2 equal=0 emitted=1\n"
        "SAIF_MEASUREMENT design=h67_fixed2s_mssb5_dc_top "
        "start_group=0 groups=1 measured_cycles=2 scope=smoke\n"
        "PASS Motion wrapper activity\n"
    )
    log.write_text(base, encoding="utf-8")
    output = tmp_path / "contract.json"
    args = (
        "--design", "h67_fixed2s_mssb5_dc_top",
        "--vcd", str(vcd), "--log", str(log), "--trace-root", str(trace),
        "--strip-path", "TOP/tb/dut", "--purpose", "identity_smoke",
        "--measurement-scope", "smoke", "--output", str(output),
    )
    passed = run_script("report_activity_vcd.py", *args)
    assert passed.returncode == 0, passed.stderr
    assert json.loads(output.read_text())["checks"]["vcd_active_duration"]

    log.write_text("ERROR: injected failure\n" + base, encoding="utf-8")
    failed = run_script("report_activity_vcd.py", *args)
    assert failed.returncode != 0

    log.write_text(base.replace("#20000", "#19000"), encoding="utf-8")
    vcd.write_text(vcd.read_text().replace("#20000", "#19000"), encoding="utf-8")
    failed = run_script("report_activity_vcd.py", *args)
    assert failed.returncode != 0

def test_saif_manifest_binds_design_hash_and_interval(tmp_path: Path) -> None:
    saif = tmp_path / "trace.saif"
    saif.write_text("(SAIFILE (DURATION 100))\n", encoding="utf-8")
    manifest = write_bound_saif_manifest(tmp_path, saif)
    passed = run_script(
        "audit_saif_manifest.py",
        "--design",
        "h67_rqtb2s_mssb5_dc_top",
        "--saif",
        str(saif),
        "--strip-path",
        "tb/dut",
        "--manifest",
        str(manifest),
    )
    assert passed.returncode == 0, passed.stderr

    data = json.loads(manifest.read_text(encoding="utf-8"))
    source_vcd = Path(data["source_vcd"])
    original_vcd = source_vcd.read_text(encoding="utf-8")
    source_vcd.write_text(original_vcd + "#2\n", encoding="utf-8")
    failed = run_script(
        "audit_saif_manifest.py",
        "--design",
        "h67_rqtb2s_mssb5_dc_top",
        "--saif",
        str(saif),
        "--strip-path",
        "tb/dut",
        "--manifest",
        str(manifest),
    )
    assert failed.returncode != 0
    source_vcd.write_text(original_vcd, encoding="utf-8")

    data["measured_cycles"] = 0
    manifest.write_text(json.dumps(data), encoding="utf-8")
    failed = run_script(
        "audit_saif_manifest.py",
        "--design",
        "h67_rqtb2s_mssb5_dc_top",
        "--saif",
        str(saif),
        "--strip-path",
        "tb/dut",
        "--manifest",
        str(manifest),
    )
    assert failed.returncode != 0


def test_expected_macro_reference_must_exist(tmp_path: Path) -> None:
    report = tmp_path / "references.rpt"
    report.write_text("SRAM_A 4\nSRAM_AB 2\n", encoding="utf-8")
    output = tmp_path / "audit.json"
    failed = run_script(
        "audit_expected_macro_refs.py",
        "--report",
        str(report),
        "--expected",
        "SRAM_A,SRAM_B",
        "--output",
        str(output),
    )
    assert failed.returncode != 0
    passed = run_script(
        "audit_expected_macro_refs.py",
        "--report",
        str(report),
        "--expected",
        "SRAM_A",
        "--output",
        str(output),
    )
    assert passed.returncode == 0


def test_ptpx_artifact_audit_enforces_coverage(tmp_path: Path) -> None:
    required = (
        "ptpx.log",
        "ptpx_run_manifest.json",
        "reports/ptpx_scope.rpt",
        "reports/ptpx_annotated_parasitics.rpt",
        "reports/ptpx_check_timing.rpt",
        "reports/ptpx_check_power.rpt",
        "reports/ptpx_unannotated.rpt",
        "reports/ptpx_switching_summary.rpt",
        "reports/ptpx_power_hierarchy.rpt",
        "reports/ptpx_power.rpt",
        "reports/ptpx_timing_setup.rpt",
    )
    for name in required:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("clean report\n", encoding="utf-8")
    unannotated = tmp_path / "reports/ptpx_unannotated.rpt"
    unannotated.write_text(
        "Total number of unannotated objects: 0\n", encoding="utf-8"
    )
    summary = tmp_path / "reports/ptpx_switching_summary.rpt"
    summary.write_text("SAIF annotation coverage: 94.9%\n", encoding="utf-8")
    failed = run_script(
        "audit_synopsys_postrun.py",
        "--mode",
        "ptpx",
        "--run-dir",
        str(tmp_path),
        "--min-saif-coverage-pct",
        "95.0",
    )
    assert failed.returncode != 0
    summary.write_text("SAIF annotation coverage: 95.1%\n", encoding="utf-8")
    passed = run_script(
        "audit_synopsys_postrun.py",
        "--mode",
        "ptpx",
        "--run-dir",
        str(tmp_path),
        "--min-saif-coverage-pct",
        "95.0",
    )
    assert passed.returncode == 0, passed.stderr

    unannotated.write_text(
        "Total number of unannotated objects: 1\n", encoding="utf-8"
    )
    failed = run_script(
        "audit_synopsys_postrun.py",
        "--mode",
        "ptpx",
        "--run-dir",
        str(tmp_path),
        "--min-saif-coverage-pct",
        "95.0",
    )
    assert failed.returncode != 0


def test_ptpx_rejects_spef_without_matching_explicit_netlist(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    pt_shell = fake_bin / "pt_shell"
    pt_shell.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    pt_shell.chmod(0o755)
    lib = tmp_path / "corner.db"
    lib.write_text("db\n", encoding="utf-8")
    saif = tmp_path / "trace.saif"
    saif.write_text("(SAIFILE (DURATION 100))\n", encoding="utf-8")
    spef = tmp_path / "route.spef"
    spef.write_text("*SPEF IEEE 1481-1998\n", encoding="utf-8")
    manifest = write_bound_saif_manifest(tmp_path, saif)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "DESIGN_NAME": "h67_rqtb2s_mssb5_dc_top",
            "LIB_DB": str(lib),
            "OPERATING_CONDITION": "tt_test",
            "SAIF_FILE": str(saif),
            "SAIF_INSTANCE": "tb/dut",
            "SAIF_MANIFEST": str(manifest),
            "SPEF_FILE": str(spef),
        }
    )
    result = subprocess.run(
        [str(ROOT / "dc_handoff" / "run_ptpx.sh")],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert result.returncode == 10
    assert "NETLIST_FILE" in result.stderr


def test_dc_artifact_audit_rejects_default_toggle_power(tmp_path: Path) -> None:
    design = "h67_rqtb2s_mssb5_dc_top"
    required = (
        "dc.log",
        "dc_run_manifest.json",
        f"netlist/{design}_mapped.v",
        f"netlist/{design}_mapped.sdc",
        f"netlist/{design}.ddc",
        f"netlist/{design}.svf",
        "reports/qor.rpt",
        "reports/area.rpt",
        "reports/power_scope.rpt",
        "reports/references.rpt",
        "reports/timing_setup.rpt",
        "reports/timing_hold.rpt",
        "reports/timing_unconstrained.rpt",
        "reports/constraint_violators.rpt",
        "reports/clock_gating.rpt",
        "reports/check_design_postcompile.rpt",
        "reports/check_timing_postcompile.rpt",
    )
    for name in required:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact\n", encoding="utf-8")
    scope = tmp_path / "reports/power_scope.rpt"
    scope.write_text("scope=NO_SAIF_POWER_NOT_RUN\n", encoding="utf-8")
    passed = run_script(
        "audit_dc_artifacts.py",
        "--design",
        design,
        "--run-dir",
        str(tmp_path),
    )
    assert passed.returncode == 0, passed.stderr

    (tmp_path / "reports/power.rpt").write_text(
        "default toggle power must not survive\n", encoding="utf-8"
    )
    failed = run_script(
        "audit_dc_artifacts.py",
        "--design",
        design,
        "--run-dir",
        str(tmp_path),
    )
    assert failed.returncode != 0


def test_synopsys_entrypoints_admit_local5_1rw_before_tool_check() -> None:
    env = os.environ.copy()
    env["PATH"] = "/usr/bin:/bin"
    env["DESIGN_NAME"] = "local5_unified_out2_1rw_dc_top"
    for script in ("run_dc.sh", "run_formality.sh", "run_ptsta.sh", "run_ptpx.sh"):
        result = subprocess.run(
            [str(ROOT / "dc_handoff" / script)],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )
        assert result.returncode == 3, (script, result.stdout, result.stderr)
        assert "不支持" not in result.stderr


def test_ptsta_supports_separate_corner_run_directories() -> None:
    text = (ROOT / "dc_handoff/run_ptsta.sh").read_text(encoding="utf-8")
    assert 'PT_RUN_DIR="${PT_RUN_DIR:-$DC_RUN_DIR}"' in text
    assert 'export OUTPUT_DIR="$PT_RUN_DIR"' in text
    assert '--run-dir "$PT_RUN_DIR"' in text
