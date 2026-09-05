#!/usr/bin/env python3
"""Repair the existing mapped C2 axes without relaxing 3ns/I/O constraints."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import run_m2233_ep34_tsbg_matched_power_repair_one_shot as cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=("ordinary", "tsbg"), required=True)
    ap.add_argument("--fm", type=Path, help="Run mapped-to-mapped preservation with this fm_shell")
    ap.add_argument("--gate-clock", action="store_true", help="Common clock-gating sensitivity, not a sparsity claim")
    args = ap.parse_args()
    cfg.no_same_uid_eda()
    dc = cfg.HW / ("results/m2242_tsbg_power_continue_20260905/ordinary_lru4/dc"
        if args.axis == "ordinary" else "results/m2248_matched_power/tsbg_b4/dc")
    if not (dc / "netlist/m2018_axis.ddc").is_file():
        raise RuntimeError("Mapped input is not ready")
    label = "gated" if args.gate_clock else "hold"
    out = Path(tempfile.mkdtemp(prefix=f"m2250_{label}_{args.axis}_", dir=cfg.HW / "results"))
    env = {**os.environ, "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
        "SNPSLMD_LICENSE_FILE": cfg.LICENSE_SERVER, "LM_LICENSE_FILE": cfg.LICENSE_FILE,
        "M2250_INPUT": str(dc), "M2250_OUTPUT": str(out),
        "M2250_SLOW": str(cfg.SLOW_DB), "M2250_FAST": str(cfg.FAST_DB),
        "M2250_GATE_CLOCK": "1" if args.gate_clock else "0"}
    print("Hold repair output:", out, flush=True)
    with (out / "dc.log").open("w") as log:
        subprocess.run([str(cfg.DC), "-f", str(Path(__file__).with_name("m2250_hold_repair.tcl"))],
            cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=7200, check=True)
    if re.search(r"^Error:", (out / "dc.log").read_text(), re.M):
        raise RuntimeError("DC error; inspect " + str(out / "dc.log"))
    slacks = {}
    for kind in ("setup", "hold"):
        for when in ("before", "after"):
            text = (out / f"reports/{kind}_{when}.rpt").read_text()
            slacks[f"{kind}_{when}_ns"] = float(re.search(r"slack \([^)]*\)\s+([-\d.]+)", text).group(1))
    area = float(re.search(r"Total cell area:\s+([\d.]+)",
        (out / "reports/area.rpt").read_text()).group(1))
    result = dict(axis=args.axis, input_dc=str(dc), output=str(out), **slacks, area_um2=area,
        clock_gating_requested=args.gate_clock,
        setup_and_hold_met=slacks["setup_after_ns"] >= 0 and slacks["hold_after_ns"] >= 0,
        scope="Mapped DC slow-max/fast-min, unchanged ideal-clock constraints; no routing",
        post_repair_formality_done=False, post_repair_power_done=False)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.fm and result["setup_and_hold_met"]:
        reference = dc / "netlist/m2018_axis_mapped.v"
        design = re.search(r"^module\s+(\w+)", reference.read_text(), re.M).group(1)
        fmout = out / "formality"
        fmout.mkdir()
        fmenv = {**env, "M2250_FM_OUTPUT": str(fmout), "M2250_FM_LIBRARY": str(cfg.SLOW_DB),
            "M2250_FM_DESIGN": design, "M2250_FM_REFERENCE": str(reference),
            "M2250_FM_IMPLEMENTATION": str(out / "netlist/m2018_axis_mapped.v")}
        print("Checking mapped hold-repair equivalence", flush=True)
        with (fmout / "fm.log").open("w") as log:
            subprocess.run([str(args.fm), "-f", str(Path(__file__).with_name("m2250_hold_equivalence.tcl"))],
                cwd=fmout, env=fmenv, stdout=log, stderr=subprocess.STDOUT, timeout=3600, check=True)
        result["post_repair_formality_done"] = (fmout / "PASS.txt").is_file()
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
