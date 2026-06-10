#!/usr/bin/env python3
"""Watcher: wait for J62a full30 to finish, then launch H56a autopilot.

The autopilot runs λ→LR→target_rate sweep, then promotes the best config
to full30 automatically.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

J62A_PID = 2782567
J62A_DIR = "/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/j62a_full30_20260527_191500"
AUTOPILOT_SCRIPT = "/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/entrypoints/run_h56a_autopilot.py"
LOG_DIR = Path("/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/results/h56a_watcher_logs")


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"watcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    print(f"=== H56a watcher ===")
    print(f"Watching PID {J62A_PID} (J62a full30)")
    print(f"Log: {log_path}")
    print()

    # Wait for J62a to finish
    check_interval = 60  # seconds
    while True:
        if not is_running(J62A_PID):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] J62a PID {J62A_PID} finished")
            break
        # Check latest epoch
        ckpt_dir = Path(J62A_DIR)
        if ckpt_dir.exists():
            epochs = sorted(
                int(p.stem.replace("checkpoint_epoch", ""))
                for p in ckpt_dir.glob("checkpoint_epoch*.pth")
            )
            latest = epochs[-1] if epochs else "?"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] J62a still running, latest epoch={latest}")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] J62a still running")
        time.sleep(check_interval)

    # Brief cooldown to let GPU memory free
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Cooling down 30s...")
    time.sleep(30)

    # Launch H56a autopilot
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Launching H56a autopilot...")
    cmd = (
        f"source /opt/conda/etc/profile.d/conda.sh && "
        f"conda activate sdformerflow && "
        f"export SDFORMER_USE_MLFLOW=0 && "
        f"nohup python -u {AUTOPILOT_SCRIPT} "
        f"> {LOG_DIR}/autopilot_stdout.log 2>&1 &"
    )
    os.system(cmd)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Autopilot launched")
    print(f"  stdout: {LOG_DIR}/autopilot_stdout.log")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
