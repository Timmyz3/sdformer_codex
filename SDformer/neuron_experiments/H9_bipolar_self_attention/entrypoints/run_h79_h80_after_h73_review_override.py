"""Run H79 and H80 after the reviewed H73 training and evaluation finish."""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
RESULTS = EXP / "results"
H73_STATUS = RESULTS / "h73_only_after_date_review_status.log"
STATUS = RESULTS / "h79_h80_after_h73_review_override_status.log"
PY = Path("/opt/conda/envs/sdformerflow/bin/python")
H73_MARKER = "ALL COMPLETE H73 only:"


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    record(f"WAIT H73 strict-load/valid825 completion: {H73_STATUS}")
    while not H73_STATUS.exists() or H73_MARKER not in H73_STATUS.read_text(
        encoding="utf-8", errors="ignore"
    ):
        time.sleep(300)

    command = [
        str(PY),
        "-u",
        str(EXP / "entrypoints/run_h73_h80_bs4acc2_queue.py"),
        "--ids",
        "H79",
        "H80",
    ]
    env = os.environ.copy()
    env.update({
        "SDFORMER_USE_MLFLOW": "0",
        "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
        "SDFORMER_SNN_BACKEND": "cupy",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    queue_log = RESULTS / "h79_h80_after_h73_review_override.log"
    record(f"START independent H79 -> H80 queue: {' '.join(command)}")
    with queue_log.open("a", encoding="utf-8") as handle:
        proc = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    record(f"END independent H79 -> H80 queue: exit_code={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"H79/H80 queue failed; log={queue_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
