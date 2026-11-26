#!/usr/bin/env python3
"""Serial dedicated-GPU E0 diagnosis and engineering smoke after E1 succeeds."""
import datetime
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/data/flow_a/runs/round32_20261021")
ARGS = Path("/data/flow_a/code/research_20261021/e0_ordinary/LAUNCH_ARGS_V2.json")


def main():
    receipt = ROOT / "e0_first_launch_v1.json"
    out = {"status": "WAIT_E1", "jobs": [], "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with receipt.open("x") as f:
        json.dump(out, f, indent=2)
    stop = time.monotonic() + 1800
    while True:
        assert time.monotonic() < stop, "Bounded wait for E1"
        r = json.loads((ROOT / "e1_v1/results.json").read_text())
        assert r["status"] != "FAILED", "E1 failed; do not start next GPU task"
        probe = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                               capture_output=True, text=True, check=True)
        if r["status"] == "LOCALIZATION80_AND4_REFERENCE_COMPLETE" and not probe.stdout.strip():
            break
        time.sleep(5)
    args = json.loads(ARGS.read_text())
    env = dict(os.environ, OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES="0")
    for task in ("diagnostic", "smoke"):
        probe = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                               capture_output=True, text=True, check=True)
        assert not probe.stdout.strip(), "GPU occupied; never compete or kill other work"
        command = ["timeout", "--signal=TERM", "--kill-after=30s", "1250"] + args[task] + ["--execute"]
        start = time.monotonic()
        out["status"] = "RUNNING_" + task
        receipt.write_text(json.dumps(out, indent=2) + "\n")
        with (ROOT / ("e0_" + task + "_v1.log")).open("x") as log:
            run = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        out["jobs"].append({"task": task, "returncode": run.returncode,
                            "elapsed_seconds": time.monotonic() - start, "command": command})
        receipt.write_text(json.dumps(out, indent=2) + "\n")
        if run.returncode:
            out["status"] = "FAILED_" + task
            receipt.write_text(json.dumps(out, indent=2) + "\n")
            raise SystemExit(run.returncode)
    out["status"] = "DIAGNOSTIC_AND_SMOKE_COMPLETE_FORMAL_NOT_STARTED"
    receipt.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
