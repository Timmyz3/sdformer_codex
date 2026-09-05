#!/usr/bin/env python3
"""Continue from existing good SAIF; fresh DC/PT, no VCS or new seal chain.

M2235 failed because its DC filelist is repository-relative, not HW-relative.
Its DC output is never reused. Each new DC uses a clean working directory.
Unchanged runtime/SAIF and the existing power parser retain measurement scope.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

import run_m2233_ep34_tsbg_matched_power_repair_one_shot as cfg

ORIGIN = cfg.HW / (
    "results/m2235_m2233_ep34_tsbg_matched_power_repair_r1_20260905"
    ".failed.3679426.quarantine")
DEFAULT_OUTPUT = cfg.HW / "results/m2242_tsbg_power_continue_20260905"
WRAPPER = Path(__file__).with_name("m2242_source_or_exit.tcl")


def parser_module():
    spec = importlib.util.spec_from_file_location("m2242_power_parser", cfg.PARSER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight():
    sources = [cfg.REPO / line.strip()
               for line in cfg.DC_FILELIST.read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#")]
    if len(sources) != 2 or not all(path.is_file() for path in sources):
        raise RuntimeError("DC source paths must resolve against repository root")
    for axis in cfg.AXES:
        for stratum in cfg.STRATA:
            point = ORIGIN / axis / stratum
            for name in ("rtl_sim.log", "rtl_measurement.saif", "rtl_prehistory.saif"):
                if not (point / name).is_file():
                    raise RuntimeError("Missing preserved activity: " + str(point / name))
            for role in ("measurement", "diagnostic_prehistory"):
                if not (point / f"parse_{role}.log").is_file():
                    raise RuntimeError("Missing completed SAIF parse")
    return sources


def run_tool(tool, script, cwd, env, log):
    cfg.run([str(tool), "-f", str(WRAPPER)], cwd,
            cfg.clean_env({**env, "M2242_TOOL_SCRIPT": str(script)}),
            21600, log)
    cfg.validate_log(log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    sources = preflight()
    if args.check:
        print("Inputs present; DC repository root:", cfg.REPO)
        print("\n".join(str(path) for path in sources))
        return
    cfg.no_same_uid_eda()
    output = args.output.resolve()
    output.mkdir()  # New output only; do not overwrite another attempt.
    parser = parser_module()
    cfg.write_json(output / "progress.json", {
        "status": "RUNNING", "activity_origin": str(ORIGIN),
        "reused_failed_dc": False, "vcs_runs": 0,
        "dc_root_fix": "repository-relative filelist resolved against REPO"})
    try:
        # Preserve good activity, not the failed synthesis. Existing SAIF
        # sidecars are copied only because the existing parser consumes them.
        for axis in cfg.AXES:
            for stratum in cfg.STRATA:
                origin = ORIGIN / axis / stratum
                point = output / axis / stratum
                point.mkdir(parents=True)
                for path in origin.iterdir():
                    if path.is_file() and (path.name.startswith("rtl_") or
                                           path.name.startswith("parse_")):
                        shutil.copy2(path, point / path.name)
        for axis, mode in cfg.AXES.items():
            dc = output / axis / "dc"
            work = dc / "work"
            work.mkdir(parents=True)
            print("Starting fresh DC:", axis, flush=True)
            run_tool(cfg.DC, cfg.DC_TCL, work, {
                "M2242_DC_WORK": str(work / "WORK"),
                "M2217_SCHEDULE_MODE": str(mode),
                "M2217_HW_ROOT": str(cfg.REPO),
                "M2217_RTL_FILELIST": str(cfg.DC_FILELIST),
                "M2217_LIB_DB": str(cfg.SLOW_DB),
                "M2217_MIN_LIB_DB": str(cfg.FAST_DB),
                "M2217_SDC_FILE": str(cfg.SDC),
                "M2217_OUTPUT_DIR": str(dc),
                "M2217_OPERATING_CONDITION": "ssg0p9v125c",
            }, dc / "dc.log")
            identity = dict(line.split("=", 1) for line in
                            (dc / "reports/identity.rpt").read_text().splitlines())
            # The power run loads TT cells. Keep timing/I/O constraints, but
            # do not ask that session to resolve the SSG-only library names.
            tt_sdc = dc / "netlist/m2018_axis_power_tt.sdc"
            sdc_text = (dc / "netlist/m2018_axis_mapped.sdc").read_text()
            tt_sdc.write_text(sdc_text.replace(
                "tcbn28hpcplusbwp35p140ssg0p9v125c",
                "tcbn28hpcplusbwp35p140tt0p9v25c").replace(
                "set_operating_conditions ssg0p9v125c",
                "set_operating_conditions tt0p9v25c"))
            for stratum in cfg.STRATA:
                point = output / axis / stratum
                pt = point / "ptpx"
                pt.mkdir()
                expected = parser.expected(axis, stratum)
                print("Starting PTPX:", axis, stratum, flush=True)
                run_tool(cfg.PT, cfg.PT_TCL, pt, {
                    "M2217_AXIS": axis, "M2217_STRATUM": stratum,
                    "M2217_DESIGN_NAME": identity["design"],
                    "M2217_TT_LIB_DB": str(cfg.TT_DB),
                    "M2217_MAPPED_NETLIST": str(dc / "netlist/m2018_axis_mapped.v"),
                    "M2217_MAPPED_SDC": str(tt_sdc),
                    "M2217_DEFAULT_MAP": str(dc / "netlist/m2018_axis.ptpx_map.default.tcl"),
                    "M2217_ESSENTIAL_MAP": str(dc / "netlist/m2018_axis.ptpx_map.essential.tcl"),
                    "M2217_RTL_SAIF": str(point / "rtl_measurement.saif"),
                    "M2217_OUTPUT_DIR": str(pt),
                    "M2217_MEASUREMENT_CYCLES": str(expected["cycles"]),
                    "M2217_ACCEPTED_BANK_REQUESTS": str(expected["accepted_bank_requests"]),
                }, pt / "ptpx.log")
        result = parser.final_result(output, output / "parsed_power.json")
        result.update(schema="m2242_tsbg_matched_power_continuation_v1",
                      status="MEASUREMENTS_COMPLETE_PENDING_TECHNICAL_REVIEW")
        result["aggregate"]["scope"] = "three fixed representative windows, not population/frame energy"
        comparison = result["aggregate"]["comparison"]
        comparison["fixed_three_window_index_weights"] = comparison.pop("fixed_population_tercile_weights")
        result["activity_origin"] = str(ORIGIN)
        result["weight_scope"] = "deterministic verification INT8 weights, not checkpoint FC weights"
        result["synthesis_corner"] = "SSG0P9V125C; min FFG1P05VM40C; PTPX TT0P9V25C"
        cfg.write_json(output / "result.json", result)
        cfg.write_json(output / "progress.json", {"status": "COMPLETE", "vcs_runs": 0})
    except Exception as exc:
        cfg.write_json(output / "progress.json", {"status": "STOPPED", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
