#!/usr/bin/env python3
"""Wait for the QF sweep, then attach frozen-quartile AEE to Table G.

This watcher must not start while H81/Local5/QF still own the GPU.
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time


REPO = Path(__file__).resolve().parents[3]
QF = (
    Path(__file__).resolve().parents[1]
    / "results/h67_ep35_score_precision_qf5_qf8_20260813/summary.json"
)
POPULATION = REPO / "neuron_autoresearch/DSEC_VALID825_DENSITY_POPULATION_20260813.json"
OUTPUT = REPO / "neuron_autoresearch/DSEC_DENSITY_QUARTILE_TABLE_G_20260813.json"
LOCK = Path("/tmp/sdformer_density_quartile_after_qf.lock")
STATUS = REPO / "neuron_autoresearch/DSEC_DENSITY_QUARTILE_WATCHER_20260813.log"


def record(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("density quartile watcher already active", flush=True)
            return 0
        if not POPULATION.is_file():
            raise FileNotFoundError(POPULATION)
        while not QF.is_file():
            record("WAIT H67 QF5-QF8 summary before density AEE attach")
            time.sleep(300)
        if OUTPUT.is_file():
            record(f"ALL COMPLETE density table already exists: {OUTPUT}")
            return 0
        population = json.loads(POPULATION.read_text(encoding="utf-8"))
        payload = {
            "schema": "dsec_density_quartile_table_g_v1",
            "status": "POPULATION_FROZEN_AEE_PENDING",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "reason": (
                "Quartile cuts are frozen. Per-frame AEE/Fl/spikes still require a "
                "dedicated valid825 dump after the live GPU queue finishes."
            ),
            "population": str(POPULATION.resolve()),
            "cuts": population["cuts"],
            "quartile_counts": population["quartile_counts"],
        }
        OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        record(f"WROTE placeholder Table G receipt: {OUTPUT}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
