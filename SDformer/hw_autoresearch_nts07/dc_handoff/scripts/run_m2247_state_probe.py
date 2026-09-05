#!/usr/bin/env python3
"""Supplement missing enum-bit activity using a plain-vector VCS probe.

Reuses original workload, SRAM responses, clock, and measurement boundaries.
Only four state bits are recorded. No RTL edits, hashes, or new approval chain.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import run_m2233_ep34_tsbg_matched_power_repair_one_shot as cfg

OUT = cfg.HW / "results/m2247_state_probe"
PROBE = cfg.HW / "tb_m2018/m2247_plain_vector_state_power_probe.sv"


def command(argv, cwd, env, logfile):
    with logfile.open("w") as log:
        subprocess.run(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                       timeout=1200, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--reuse-builds", type=Path)
    args = ap.parse_args()
    cfg.no_same_uid_eda()
    output = args.output.resolve()
    output.mkdir()
    sources = [cfg.REPO / l.strip() for l in cfg.VCS_FILELIST.read_text().splitlines()
               if l.strip() and not l.lstrip().startswith("#")]
    env = {**os.environ, "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
        "SNPSLMD_LICENSE_FILE": cfg.LICENSE_SERVER, "LM_LICENSE_FILE": cfg.LICENSE_FILE,
        "VCS_HOME": str(cfg.VCS.parent.parent), "VCS_ARCH_OVERRIDE": "linux"}
    results = {}
    for axis, mode in cfg.AXES.items():
        point = output / axis
        point.mkdir()
        build = args.reuse_builds.resolve() / axis if args.reuse_builds else point
        if not args.reuse_builds:
            command([str(cfg.VCS), "-full64", "-sverilog", "-timescale=1ns/1ps",
                 "+vcs+initreg+random", f"+define+M2217_SCHEDULE_MODE={mode}",
                 "-debug_access+r", "-assert", "svaext", "-lca",
                 *map(str, sources), str(PROBE), "-top",
                 "tb_m2217_m2018_tsbg_matched_native_saif_power", "-o", str(build / "simv")],
                    build, env, build / "compile.log")
        for window in cfg.STRATA:
            saif = point / f"{window}_state.saif"
            log = point / f"{window}.log"
            command([str(build / "simv"), f"+M2217_STRATUM={window}", "-no_save", "-ucli",
                     "-i", str(Path(__file__).with_name("m2247_state_probe.ucli.tcl"))],
                    build, {**env, "M2247_STATE_SAIF": str(saif)}, log)
            text = saif.read_text()
            if not all(f"state_q\\[{bit}\\]" in text for bit in range(4)):
                raise RuntimeError("Plain-vector probe still missing indexed state bits")
            duration = float(re.search(r"\(DURATION ([0-9.]+)\)", text).group(1))
            expected = float(re.search(r"duration_ns=([0-9.]+)", log.read_text()).group(1)) * 1000
            if duration != expected or re.search(r"\(TX (?!0\))[0-9]+\)", text):
                raise RuntimeError("Probe window contains preload or unknown state")
            results[f"{axis}/{window}"] = str(saif)
            print("Captured indexed state:", axis, window, flush=True)
    (output / "result.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
