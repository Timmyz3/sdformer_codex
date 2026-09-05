#!/usr/bin/env python3
"""One small Synopsys regression, optionally queued behind the active power run.

No hashes, approval chain, or destructive cleanup. Each invocation gets a
normal temporary build directory; stdout contains its path and final status.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

HW = Path(__file__).resolve().parents[2]
VCS = Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs")
POWER = HW / "results/m2242_tsbg_power_continue_20260905/progress.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--after-power", action="store_true")
    args = p.parse_args()
    if args.after_power:
        print("Queued behind M2242 DC/PTPX, not starting another EDA job.", flush=True)
        while json.loads(POWER.read_text())["status"] == "RUNNING":
            time.sleep(15)
    # Read-only collision check, using executable names rather than matching
    # this Python runner's command line against its own strings.
    ps = subprocess.check_output(["ps", "-u", str(os.getuid()), "-o", "comm="], text=True)
    if any(name.strip() in {"dc_shell", "snps_shell", "common_shell_exe", "common_shell_ex", "pt_shell", "simv"}
           for name in ps.splitlines()):
        raise RuntimeError("EDA queue occupied; leave this regression pending")
    out = Path(tempfile.mkdtemp(prefix="m2243_borrowed_vcs_", dir=HW / "results"))
    print("Output:", out, flush=True)
    env = {**os.environ, "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
        "VCS_HOME": str(VCS.parent.parent), "VCS_ARCH_OVERRIDE": "linux",
        "SNPSLMD_LICENSE_FILE": "27030@ic.ismd-nemo", "LM_LICENSE_FILE": "/opt/synopsys/Synopsys.dat"}
    sources = [HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
        HW / "rtl_m2243/m2243_c2_borrowed_weight_consumers.sv",
        HW / "tb_m2243/tb_m2243_borrowed_weight_consumers.sv"]
    commands = [[str(VCS), "-full64", "-sverilog", "-assert", "svaext",
        "-timescale=1ns/1ps", "-top", "tb_m2243_borrowed_weight_consumers",
        *map(str, sources), "-o", str(out / "simv")],
        [str(out / "simv"), "+MASKED=0"], [str(out / "simv"), "+MASKED=1"]]
    for name, cmd in zip(("compile", "sim_full", "sim_masked"), commands):
        with (out / f"{name}.log").open("w") as log:
            subprocess.run(cmd, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=600, check=True)
    passes = []
    for name in ("sim_full", "sim_masked"):
        log = (out / f"{name}.log").read_text()
        match = re.search(r"PASS_M2243_M803_BORROWED_CONSUMERS[^\n]*", log)
        if match is None or re.search(r"(?:Error|Fatal):|Assertion.*fail", log, re.I):
            raise RuntimeError("VCS regression failed: " + str(out / f"{name}.log"))
        passes.append(match.group())
    result = {"status": "PASS", "scope": "M803 plus per-beat borrowed-payload controller; directed only",
        "pass_lines": passes, "full_scheduler_integrated": False,
        "cycle_speedup_measured": False, "area_or_power_measured": False}
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
