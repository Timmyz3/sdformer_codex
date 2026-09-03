#!/usr/bin/python3.12
"""Pure source/parser hammer for M2061.  Never launches EDA or lmstat."""
import importlib.util
import json
from pathlib import Path
import tempfile


HW = Path(__file__).resolve().parents[2]
PARSER_PATH = HW / "system_simulator/scripts/parse_m2061_m2056_m2018_tsbg_settled_mapped_energy_result.py"
RUNNER_PATH = HW / "dc_handoff/scripts/run_m2061_m2056_m2018_tsbg_settled_mapped_energy_one_shot.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise AssertionError("module load failed " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = load("m2061_parser_hammer", PARSER_PATH)
R = load("m2061_runner_hammer", RUNNER_PATH)
checks = []


def ok(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def rejects(name, function, contains):
    try:
        function()
    except BaseException as exc:
        ok(name, contains in str(exc))
        return
    raise AssertionError(name + " did not reject")


def pass_line():
    return P.PASS_PREFIX + " ".join(
        key + "=" + value for key, value in P.EXPECTED_PASS.items())


def runtime_text(axis):
    return "\n".join((P.BEGIN_MARKER, P.AXES[axis]["end_marker"], pass_line())) + "\n"


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


ok("static_source_identity", P.validate_sources()["status"] == "PASS_M2061_STATIC_SOURCE_IDENTITY")
ok("source_inventory_33", len(P.SOURCE_SHA256) == 33)
ok("new_m2061_namespaces", all("m2061_" in str(path) and "m2058_" not in str(path)
                                for path in (R.ATTEMPT, R.RESULT, R.FAILURE, R.PRIVATE)))
ok("zero_retry_budget", R.COUNTS == {"license_preflight_lmstat": 1, "vcs_compiles": 2,
                                     "simv_runs": 2, "saif_files": 2, "ptpx_runs": 2})

runner_text = RUNNER_PATH.read_text()
parser_text = PARSER_PATH.read_text()
tb_text = (HW / "tb_m2018/tb_m2061_m2018_tsbg_matched_mapped_energy.sv").read_text()
pt_text = P.PT_TCL.read_text()
ok("runner_no_retry_loop", "automatic_retry=False" in runner_text
   and "M2061_ATTEMPT_CONSUMED_NO_RETRY" in runner_text
   and "for axis in PARSER.AXIS_ORDER" in runner_text)
ok("m2058_permanent_failure_guard", "m2058_failed_no_retry" in runner_text
   and "m2058_outputs_reused\": False" in runner_text
   and "m2058_no_retry" in parser_text)
ok("two_stops", tb_text.count("$stop;") == 2)
ok("settled_negedge", "always @(negedge core.clk_core)" in tb_text
   and tb_text.count("#0.01;") >= 3)
ok("per_signal_unknown_diagnostic", "$isunknown(value)" in tb_text
   and "signal=%s" in tb_text and "$isunknown({" not in tb_text)
ok("valid_gated_sidebands", all(token in tb_text for token in (
    "if (core.load_valid_base)", "if (core.base.mem_req_valid[bank])",
    "if (core.base.mem_rsp_valid[bank])",
    "if (core.base.bridge_valid && core.base.bridge_bank_valid[bank])",
    "if (core.base.commit_valid)", "if (core.load_valid_tsbg)",
    "if (core.tsbg.mem_req_valid[bank])", "if (core.tsbg.mem_rsp_valid[bank])",
    "if (core.tsbg.bridge_valid && core.tsbg.bridge_bank_valid[bank])",
    "if (core.tsbg.commit_valid)")))
ok("unconditional_qualifiers_and_counters", all(token in tb_text for token in (
    'require_known(core.base.mem_req_valid', 'require_known(core.base.bridge_valid',
    'require_known(core.base.protocol_error', 'require_known(core.base.cycle_count',
    'require_known(core.tsbg.mem_req_valid', 'require_known(core.tsbg.bridge_valid',
    'require_known(core.tsbg.protocol_error', 'require_known(core.tsbg.cycle_count')))
ok("duration_formula_parser_and_pt", 'cfg["cycles"] * 3.0' in parser_text
   and "double($measurement_cycles) * 3.0" in pt_text)
ok("pt_both_libraries_before_sdc", pt_text.index("read_db $ssg_lib_db")
   < pt_text.index("read_db $tt_lib_db") < pt_text.index("read_sdc $mapped_sdc"))
ok("pt_exact_annotation_and_scope", "M2061_FAIL_EXACT_ANNOTATION" in pt_text
   and "read_saif -strip_path $saif_scope" in pt_text)

with tempfile.TemporaryDirectory(prefix="m2062_hammer_") as temp_name:
    temp = Path(temp_name)
    for axis in P.AXIS_ORDER:
        log = temp / (axis + ".runtime.log")
        write(log, runtime_text(axis))
        parsed = P.parse_runtime(log, axis)
        ok("runtime_positive_" + axis,
           parsed["stop_markers"] == 2 and parsed["final_m2051_passes"] == 1)

        write(log, P.AXES[axis]["end_marker"] + "\n" + P.BEGIN_MARKER
              + "\n" + pass_line() + "\n")
        rejects("runtime_reject_order_" + axis,
                lambda log=log, axis=axis: P.parse_runtime(log, axis),
                "two-stop/final-PASS ordering")
        write(log, runtime_text(axis) + pass_line() + "\n")
        rejects("runtime_reject_duplicate_pass_" + axis,
                lambda log=log, axis=axis: P.parse_runtime(log, axis),
                "final M2051 PASS count")
        write(log, runtime_text(axis).replace("products=29472", "products=29471"))
        rejects("runtime_reject_ledger_" + axis,
                lambda log=log, axis=axis: P.parse_runtime(log, axis),
                "M2051 PASS identity/ledger drift")
        write(log, runtime_text(axis) + "M2061 mapped X/Z signal=injected\n")
        rejects("runtime_reject_xz_" + axis,
                lambda log=log, axis=axis: P.parse_runtime(log, axis),
                "runtime fatal/XZ")

        saif = temp / (axis + ".saif")
        duration = P.AXES[axis]["cycles"] * 3
        good_saif = ("(SAIFILE\n  (TIMESCALE 1 ns)\n  (DURATION %d)\n"
                     "  (INSTANCE mapped_implementation\n"
                     "    (NET (n (T0 1) (T1 1) (TX 0) (TC 2)))))\n)\n") % duration
        write(saif, good_saif)
        ok("saif_positive_" + axis, P.parse_saif(saif, axis)["duration_ns"] == duration)
        write(saif, good_saif.replace("(DURATION %d)" % duration,
                                      "(DURATION %d)" % (duration + 3)))
        rejects("saif_reject_duration_" + axis,
                lambda saif=saif, axis=axis: P.parse_saif(saif, axis), "SAIF duration")
        write(saif, good_saif.replace("(TX 0)", "(TX 1)"))
        rejects("saif_reject_tx_" + axis,
                lambda saif=saif, axis=axis: P.parse_saif(saif, axis), "nonzero SAIF TX")

        compile_log = temp / (axis + ".compile.log")
        command = ["vcs", "-top", P.TOP, "-f", str(P.AXES[axis]["filelist"])]
        write(compile_log, "M2061_COMMAND_JSON=" + json.dumps(command) + "\n")
        ok("compile_command_positive_" + axis,
           P.parse_command_log(compile_log, axis)["command"] == command)
        write(compile_log, "M2061_COMMAND_JSON=" + json.dumps(
            ["vcs", "-top", "wrong", "-f", str(P.AXES[axis]["filelist"])] ) + "\n")
        rejects("compile_reject_top_" + axis,
                lambda compile_log=compile_log, axis=axis:
                P.parse_command_log(compile_log, axis), "compile top")

        pt_root = temp / (axis + ".ptpx")
        reports = pt_root / "reports"
        reports.mkdir(parents=True)
        write(pt_root / "ptpx.log", "clean\n")
        write(pt_root / "PTPX_INTERNAL_COMPLETE.txt",
              "PASS_M2061_M2018_TSBG_SETTLED_MAPPED_PTPX_PENDING_RESULT_HAMMER\n"
              "axis=%s\nmeasurement_cycles=%d\n" % (axis, P.AXES[axis]["cycles"]))
        write(reports / "saif_annotation_summary.rpt",
              "Total number of nets = 10\nNumber of annotated nets = 10 (100.0%)\n"
              "Total number of leaf cells = 9\nNumber of fully annotated leaf cells = 9 (100.0%)\n")
        boundary = {
            "milestone": "M2061", "axis": axis, "design": P.AXES[axis]["design"],
            "sampling": "settled_negedge_valid_gated_sideband_checker",
            "mapped_simulation": "zero_delay_functional_no_SDF",
            "unit_delay_fix_claimed": "false",
            "window_alignment": "first_settled_execute_negedge_to_settled_completion_negedge",
            "first_half_cycle_transition_excluded": "true",
            "analysis": "averaged_prelayout_standard_cell_power",
            "power_corner": "tt0p9v25c", "clock_period_ns": "3.0",
            "measurement_cycles": str(P.AXES[axis]["cycles"]),
            "measurement_duration_ns": str(duration),
            "descriptor_preload_cycles_excluded": "383",
            "workload": "ep34_full40_global_slot42_sample0_layer28_fc1_token0_g48",
            "saif_scope": P.AXES[axis]["scope"], "clock_network": "ideal_no_cts",
            "wireload": "ZeroWireload", "spef": "false", "macro_count": "0",
            "external_weight_sram_excluded": "true"}
        write(reports / "scope_and_boundary.rpt",
              "".join(key + "=" + value + "\n" for key, value in boundary.items()))
        write(reports / "power.rpt",
              "Report : Averaged Power\n-unit mW\nNet Switching Power = 1.0\n"
              "Cell Internal Power = 2.0\nCell Leakage Power = 0.5\nTotal Power = 3.5\n")
        ok("ptpx_positive_" + axis,
           P.parse_ptpx(pt_root, axis)["execute_energy_pj"]["total"] == 3.5 * duration)
        write(reports / "scope_and_boundary.rpt",
              "".join(key + "=" + ("wrong" if key == "saif_scope" else value) + "\n"
                      for key, value in boundary.items()))
        rejects("ptpx_reject_scope_" + axis,
                lambda pt_root=pt_root, axis=axis: P.parse_ptpx(pt_root, axis),
                "boundary field saif_scope")

    tree = temp / "work"
    write(tree / "candidate/axis/mapped_sim.log", "evidence\n")
    (tree / "build").mkdir()
    (tree / "build/tool_link").symlink_to("../candidate/axis/mapped_sim.log")
    fingerprint = R.tree_fingerprint(tree)
    ok("failure_fingerprint_regular_and_symlink",
       {row["type"] for row in fingerprint} == {"file", "symlink"}
       and any(row.get("target") == "../candidate/axis/mapped_sim.log"
               for row in fingerprint))
    copied = temp / "copied"
    R.copy_failure_evidence(tree, copied)
    ok("failure_copy_excludes_symlink",
       (copied / "candidate/axis/mapped_sim.log").is_file()
       and not any(path.is_symlink() for path in copied.rglob("*")))

power = tempfile.NamedTemporaryFile(mode="w", delete=False)
try:
    power.write("Report : Averaged Power\n-unit mW\nNet Switching Power = 1.0\n"
                "Cell Internal Power = 2.0\nCell Leakage Power = 0.5\nTotal Power = 3.5\n")
    power.close()
    ok("power_positive", P.parse_power_report(power.name)["total_mw"] == 3.5)
    Path(power.name).write_text(Path(power.name).read_text().replace("Total Power = 3.5",
                                                                    "Total Power = 3.4"))
    rejects("power_reject_subtotal", lambda: P.parse_power_report(power.name), "power subtotal")
finally:
    if Path(power.name).exists():
        Path(power.name).unlink()

print(json.dumps({"status": "PASS_M2062_M2061_SOURCE_HAMMER_NO_EDA",
                  "checks": checks, "check_count": len(checks)},
                 indent=2, sort_keys=True))
