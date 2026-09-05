#!/usr/bin/env python3
"""Reuse completed DC and SAIF; run one corrected, inspectable PTPX window."""
import argparse
import os
from pathlib import Path
import re
import subprocess

import run_m2233_ep34_tsbg_matched_power_repair_one_shot as cfg


def run(dc, saif, output, state_saif=None):
    output.mkdir(parents=True)
    netlist = dc / "netlist/m2018_axis_mapped.v"
    design = re.search(r"^module\s+(m2018_c2_tsbg_b4\w+)", netlist.read_text(), re.M).group(1)
    # Keep the existing user's environment, including HOME needed by the tool
    # setup. Nothing reassigns or redirects the user's home directory.
    env = {**os.environ, "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
        "SNPSLMD_LICENSE_FILE": cfg.LICENSE_SERVER, "LM_LICENSE_FILE": cfg.LICENSE_FILE,
        "SYNOPSYS_LC_ROOT": "/opt/synopsys/lc/V-2023.12-SP3",
        "M2246_DESIGN": design, "M2246_LIB": str(cfg.TT_DB),
        "M2246_NETLIST": str(netlist), "M2246_SDC": str(dc / "netlist/m2018_axis_power_tt.sdc"),
        "M2246_MAP": str(dc / "netlist/m2018_axis.ptpx_map.default.tcl"),
        "M2246_SAIF": str(saif), "M2246_OUTPUT": str(output)}
    if state_saif:
        env["M2246_STATE_SAIF"] = str(state_saif)
    with (output / "ptpx.log").open("w") as log:
        subprocess.run([str(cfg.PT), "-f", str(Path(__file__).with_name("m2246_rtl_activity_power.tcl"))],
            cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200, check=True)
    print("PTPX completed:", output, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dc", type=Path, required=True)
    p.add_argument("--saif", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--state-saif", type=Path)
    args = p.parse_args()
    cfg.no_same_uid_eda()
    run(args.dc.resolve(), args.saif.resolve(), args.output.resolve(),
        args.state_saif.resolve() if args.state_saif else None)
