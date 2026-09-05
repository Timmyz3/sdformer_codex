#!/usr/bin/env python3
"""Mapped-to-mapped functional preservation after clock gating and hold ECO."""
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
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--implementation", type=Path, required=True)
    ap.add_argument("--svf", type=Path, action="append", default=[])
    args = ap.parse_args()
    cfg.no_same_uid_eda()
    out = Path(tempfile.mkdtemp(prefix="m2255_mapped_preservation_", dir=cfg.HW / "results"))
    reference = args.reference.resolve() / "netlist/m2018_axis_mapped.v"
    implementation = args.implementation.resolve() / "netlist/m2018_axis_mapped.v"
    design = re.search(r"\bmodule\s+(m(?:2018_c2_tsbg_b4|2249_c2_consumer_scoped_bank_fill)\w+)", reference.read_text()).group(1)
    env = {**os.environ, "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
        "SNPSLMD_LICENSE_FILE": cfg.LICENSE_SERVER, "LM_LICENSE_FILE": cfg.LICENSE_FILE,
        "M2250_FM_OUTPUT": str(out), "M2250_FM_LIBRARY": str(cfg.SLOW_DB),
        "M2250_FM_DESIGN": design, "M2250_FM_REFERENCE": str(reference),
        "M2250_FM_IMPLEMENTATION": str(implementation), "M2250_GATE_CLOCK": "1",
        "M2255_SVF_LIST": ":".join(str(p.resolve()) for p in args.svf)}
    print("Mapped preservation:", out, flush=True)
    with (out / "fm.log").open("w") as log:
        subprocess.run(["/opt/synopsys/fm/V-2023.12-SP3/bin/fm_shell", "-f",
            str(Path(__file__).with_name("m2250_hold_equivalence.tcl"))],
            cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=3600, check=True)
    result = dict(reference=str(reference), implementation=str(implementation),
                  passed=(out / "PASS.txt").is_file(),
                  scope="Mapped-to-mapped, clock gating and data/reset buffer preservation; not RTL proof or timing")
    (out / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
