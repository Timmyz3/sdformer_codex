#!/usr/bin/env python3
"""Reuse completed DC and SAIF; run one corrected, inspectable PTPX window."""
import argparse
import os
from pathlib import Path
import re
import subprocess

import run_m2233_ep34_tsbg_matched_power_repair_one_shot as cfg


def run(dc, saif, output, state_saif=None, map_source=None, gate_level=False):
    output.mkdir(parents=True)
    netlist = dc / "netlist/m2018_axis_mapped.v"
    # Gated netlists may put helper clock-gate modules before the actual top.
    design = re.search(r"\bmodule\s+(m2018_c2_tsbg_b4\w+)", netlist.read_text()).group(1)
    timing = (dc / "netlist/m2018_axis_mapped.sdc").read_text()
    power_sdc = output / "power_tt.sdc"
    power_sdc.write_text(timing.replace("tcbn28hpcplusbwp35p140ssg0p9v125c",
        "tcbn28hpcplusbwp35p140tt0p9v25c").replace(
        "set_operating_conditions ssg0p9v125c", "set_operating_conditions tt0p9v25c"))
    # Keep the existing user's environment, including HOME needed by the tool
    # setup. Nothing reassigns or redirects the user's home directory.
    env = {**os.environ, "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
        "SNPSLMD_LICENSE_FILE": cfg.LICENSE_SERVER, "LM_LICENSE_FILE": cfg.LICENSE_FILE,
        "SYNOPSYS_LC_ROOT": "/opt/synopsys/lc/V-2023.12-SP3",
        "M2246_DESIGN": design, "M2246_LIB": str(cfg.TT_DB),
        "M2246_NETLIST": str(netlist), "M2246_SDC": str(power_sdc),
        "M2246_MAP": str((map_source or dc) / "netlist/m2018_axis.ptpx_map.default.tcl"),
        "M2246_GATE_LEVEL": "1" if gate_level else "0",
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
    p.add_argument("--map-source", type=Path,
                   help="Original RTL map for a mapped ECO; annotation must be checked again")
    p.add_argument("--gate-level", action="store_true",
                   help="Direct activity from this exact mapped netlist; do not load an RTL map")
    args = p.parse_args()
    cfg.no_same_uid_eda()
    run(args.dc.resolve(), args.saif.resolve(), args.output.resolve(),
        args.state_saif.resolve() if args.state_saif else None,
        args.map_source.resolve() if args.map_source else None, args.gate_level)
