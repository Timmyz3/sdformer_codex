#!/usr/bin/env python3
"""Bounded endpoint buffer ECO; do not remap/remove the inserted hold cells."""
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
    ap.add_argument("--input-dc", type=Path, required=True)
    ap.add_argument("--upsize", action="append", default=[], metavar="CELL=LIBCELL")
    args = ap.parse_args()
    cfg.no_same_uid_eda()
    source = args.input_dc.resolve()
    out = Path(tempfile.mkdtemp(prefix="m2255_hold_buffers_", dir=cfg.HW / "results"))
    print("Targeted hold ECO output:", out, flush=True)
    env = {**os.environ, "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
        "SNPSLMD_LICENSE_FILE": cfg.LICENSE_SERVER, "LM_LICENSE_FILE": cfg.LICENSE_FILE,
        "M2255_INPUT": str(source), "M2255_OUTPUT": str(out),
        "M2255_UPSIZE": " ".join(value.replace("=", " ") for value in args.upsize),
        "M2255_SLOW": str(cfg.SLOW_DB), "M2255_FAST": str(cfg.FAST_DB)}
    with (out / "dc.log").open("w") as log:
        subprocess.run([str(cfg.DC), "-f", str(Path(__file__).with_name("m2255_targeted_hold_buffers.tcl"))],
            cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=True)
    result = dict(input_dc=str(source), output=str(out), timing_scope="Unchanged 3ns SSG-max/FFG-min, ideal clock, no routing")
    for kind in ("setup", "hold"):
        text = (out / f"reports/{kind}_after.rpt").read_text()
        result[kind + "_ns"] = float(re.search(r"slack \([^)]*\)\s+([-\d.]+)", text).group(1))
    result["area_um2"] = float(re.search(r"Total cell area:\s+([\d.]+)", (out / "reports/area.rpt").read_text()).group(1))
    result["setup_and_hold_met"] = result["setup_ns"] >= 0 and result["hold_ns"] >= 0
    result["formality_pass"] = False
    (out / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
