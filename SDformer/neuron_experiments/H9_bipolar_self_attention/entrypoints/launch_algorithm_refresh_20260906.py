#!/usr/bin/env python3
"""Detach a finite queue and keep its Markdown result table up to date."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import run_algorithm_refresh_20260906 as queue


def report():
    summary = queue.ROOT / "summary.json"
    rows = json.loads(summary.read_text()) if summary.exists() else {}
    current_path = queue.ROOT / "current.json"
    current = json.loads(current_path.read_text()) if current_path.exists() else {}
    lines = ["# September algorithm refresh results", "",
             "All arms: frozen C12 ep34 parent, five fresh-optimizer epochs, full DSEC train.",
             "Only standard valid825 rows are included; smoke numbers are excluded.", "",
             f"Current status: `{current.get('status', 'starting')}`; phase: `{current.get('phase', '')}`.",
             f"Current run: `{Path(current.get('folder', '')).name}`.", "",
             "| Mode | Epoch | AEE | AAE-2D | AE-3D | Fl (%) | Spikes (G) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for mode in queue.MODES:
        if mode not in rows:
            lines.append(f"| {mode} | pending | - | - | - | - | - |")
        for row in rows.get(mode, []):
            lines.append(f"| {mode} | {row['epoch']} | {row['AEE']:.6f} | "
                         f"{row['AAE_2D']:.6f} | {row['AE_3D']:.6f} | "
                         f"{row['Fl_percent']:.4f} | {row['spikes_g']:.4f} |")
    lines += ["", "No automatic promotion. Compare distill versus augment and candidates versus control.",
              "New weights do not inherit ep34 RTL/capture evidence.", ""]
    path = queue.ROOT / "report.md"
    temp = path.with_suffix(".tmp")
    temp.write_text("\n".join(lines))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervise", action="store_true")
    args = parser.parse_args()
    if not args.supervise:
        queue.verify_identity()
        acceptance = json.loads((queue.ROOT / "smoke_acceptance.json").read_text())
        if not acceptance["accepted"] or acceptance["manifest_sha256"] != queue.sha(queue.MANIFEST):
            raise RuntimeError("smoke not admitted")
        with (queue.ROOT / "launch.attempt.json").open("x") as stream:
            json.dump({"manifest_sha256": queue.sha(queue.MANIFEST)}, stream)
        with (queue.ROOT / "supervisor.log").open("x") as stream:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--supervise"],
                cwd=queue.REPO, env=queue.environment(), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        queue.write_json(queue.ROOT / "supervisor.json", {"pid": child.pid,
                         "launcher_sha256": queue.sha(Path(__file__))})
        print(f"Supervisor PID={child.pid}; status={queue.ROOT / 'status.log'}", flush=True)
        return
    child = subprocess.Popen([sys.executable, "-u", str(Path(queue.__file__)), "--run"],
                             cwd=queue.REPO, env=queue.environment())
    while child.poll() is None:
        report()
        time.sleep(30)
    report()
    raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
