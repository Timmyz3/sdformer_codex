#!/usr/bin/env python3
"""Finish TSBG synthesis and the six corrected matched power windows."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import run_m2233_ep34_tsbg_matched_power_repair_one_shot as cfg
from run_m2246_power_window import run as power_window

OUT = cfg.HW / "results/m2248_matched_power"
OLD = cfg.HW / "results/m2242_tsbg_power_continue_20260905"
STATE = cfg.HW / "results/m2247_state_probe_windowed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args()
    output = args.output.resolve()
    cfg.no_same_uid_eda()
    output.mkdir()
    progress = output / "progress.json"
    def status(stage):
        progress.write_text(json.dumps({"stage": stage}, indent=2) + "\n")
        print(stage, flush=True)
    try:
        dc = output / "tsbg_b4/dc"
        dc.mkdir(parents=True)
        env = {**os.environ, "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
            "SNPSLMD_LICENSE_FILE": cfg.LICENSE_SERVER, "LM_LICENSE_FILE": cfg.LICENSE_FILE,
            "M2248_REPO": str(cfg.REPO), "M2248_OUTPUT": str(dc), "M2248_MODE": "1",
            "M2248_SLOW": str(cfg.SLOW_DB), "M2248_FAST": str(cfg.FAST_DB)}
        status("TSBG_DC_RUNNING")
        with (dc / "dc.log").open("w") as log:
            subprocess.run([str(cfg.DC), "-f", str(Path(__file__).with_name("m2248_matched_dc.tcl"))],
                cwd=dc, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=21600, check=True)
        text = (dc / "netlist/m2018_axis_mapped.sdc").read_text()
        (dc / "netlist/m2018_axis_power_tt.sdc").write_text(text.replace(
            "tcbn28hpcplusbwp35p140ssg0p9v125c", "tcbn28hpcplusbwp35p140tt0p9v25c").replace(
            "set_operating_conditions ssg0p9v125c", "set_operating_conditions tt0p9v25c"))
        for axis in cfg.AXES:
            axis_dc = dc if axis == "tsbg_b4" else OLD / axis / "dc"
            for window in cfg.STRATA:
                status(f"PTPX_RUNNING {axis}/{window}")
                power_window(axis_dc, OLD / axis / window / "rtl_measurement.saif",
                    output / axis / window, STATE / axis / f"{window}_state.saif")
        status("TOOL_RUNS_COMPLETE_REVIEW_NUMBERS")
    except Exception as exc:
        status("STOPPED " + str(exc))
        raise


if __name__ == "__main__":
    main()
