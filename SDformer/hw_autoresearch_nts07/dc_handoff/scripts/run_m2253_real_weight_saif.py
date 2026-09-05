#!/usr/bin/env python3
"""Real ep34 FC weight-code power sensitivity, not a new accuracy result."""
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
    ap.add_argument("--axis", choices=tuple(cfg.AXES), required=True)
    ap.add_argument("--weights", type=Path, default=cfg.HW / "results/m2251_fc_power_weight_inputs")
    args = ap.parse_args()
    cfg.no_same_uid_eda()
    out = Path(tempfile.mkdtemp(prefix=f"m2253_{args.axis}_", dir=cfg.HW / "results"))
    print("Real-weight SAIF output:", out, flush=True)
    sources = [cfg.REPO / s.strip() for s in cfg.VCS_FILELIST.read_text().splitlines()
               if s.strip() and not s.startswith("#")]
    env = {**os.environ, "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
           "SNPSLMD_LICENSE_FILE": cfg.LICENSE_SERVER, "LM_LICENSE_FILE": cfg.LICENSE_FILE,
           "VCS_HOME": str(cfg.VCS.parent.parent), "VCS_ARCH_OVERRIDE": "linux"}
    with (out / "compile.log").open("w") as log:
        subprocess.run([str(cfg.VCS), "-full64", "-sverilog", "-timescale=1ns/1ps",
            "+vcs+initreg+random", f"+define+M2217_SCHEDULE_MODE={cfg.AXES[args.axis]}",
            "+define+M2253_CAPTURED_WEIGHTS", "-debug_access+r", "-assert", "svaext", "-lca",
            *map(str, sources), str(cfg.HW / "tb_m2018/m2247_plain_vector_state_power_probe.sv"),
            "-top", "tb_m2217_m2018_tsbg_matched_native_saif_power", "-o", str(out / "simv")],
            cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200, check=True)
    rows = []
    for window in cfg.STRATA:
        point = out / window
        point.mkdir()
        with (point / "vcs.log").open("w") as log:
            subprocess.run([str(out / "simv"), "-no_save", f"+M2217_STRATUM={window}",
                f"+M2253_WEIGHTS={args.weights.resolve() / (window + '_weights.memh')}", "-ucli",
                "-i", str(Path(__file__).with_name("m2253_real_weight_power.ucli.tcl"))],
                cwd=point, env={**env, "M2253_OUTPUT": str(point)}, stdout=log,
                stderr=subprocess.STDOUT, timeout=1200, check=True)
        text = (point / "vcs.log").read_text()
        if "PASS_M2217_SINGLE_DUT_NATIVE_SAIF" not in text:
            raise RuntimeError("Real-weight arithmetic/ledger did not pass: " + str(point))
        cycles = int(re.search(r"M2217_WINDOW_END .*?cycles=(\d+)", text).group(1))
        for file, unit in (("activity.saif", "ns"), ("state.saif", "ps")):
            activity = (point / file).read_text()
            scale = re.search(r"\(TIMESCALE\s+([\d.]+)\s+(\w+)\)", activity)
            duration = float(re.search(r"\(DURATION\s+([\d.]+)\)", activity).group(1))
            duration_ns = duration * float(scale.group(1)) * {"ns": 1, "ps": .001}[scale.group(2)]
            if abs(duration_ns - 3 * cycles) > .001:
                raise RuntimeError("Wrong SAIF measurement window: " + str(point / file))
        rows.append(dict(window=window, cycles=cycles, numeric_pass=True, activity=str(point / "activity.saif"),
                         state=str(point / "state.saif")))
        print("Real-weight VCS and SAIF PASS:", args.axis, window, flush=True)
    (out / "result.json").write_text(json.dumps(dict(axis=args.axis, rows=rows,
        scope="Candidate quantized ep34 FC weight-code sensitivity; not an AEE or power result"), indent=2) + "\n")


if __name__ == "__main__":
    main()
