#!/usr/bin/env python3
"""Run the remaining DATE algorithm evidence queue without official DSEC upload."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
ROOT = EXP / "results/date_algorithm_evidence_queue_20260821"
STATUS = ROOT / "status.log"
LOCK = Path("/tmp/sdformer_date_algorithm_evidence_queue_20260821.lock")
D3_RANKING = (
    EXP
    / "results/dsec_fullres_w15_H88_local5_a3s_ft5_short_20260818/"
    "profile_ranking_valid825.md"
)
H81_CONFIG = EXP / "configs/generated/dsec_fullres_w15_H81_nomotion_bb1e4_ft40.yml"
H81_CHECKPOINT = (
    EXP
    / "results/dsec_fullres_w15_H81_nomotion_bb1e4_ft40_20260811/"
    "checkpoint_epoch29.pth"
)
H81_PROFILE = EXP / "results/h81_ep29_score_pair_profile20_20260821"
H81_TRACE = H81_PROFILE / "bit_trace"
H67_TRACE = (
    REPO
    / "hw_autoresearch_nts07/results/"
    "h67_ep35_multisample100_t450_real_rtl_bit_trace"
)
PAIR_RESULT = REPO / "neuron_autoresearch/H67_H81_SCORE_PAIR_PROFILE_20260821.json"


def record(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "SDFORMER_USE_MLFLOW": "0",
            "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
            "SDFORMER_SNN_BACKEND": "cupy",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return env


def run(command: list[str], log: Path, label: str) -> None:
    record(f"START {label}: {' '.join(command)}")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=REPO,
            env=environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    record(f"END {label}: exit_code={result.returncode}")
    if result.returncode:
        raise RuntimeError(f"{label} failed; see {log}")


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("DATE algorithm evidence queue already active", flush=True)
            return 0
        while not D3_RANKING.is_file():
            record("WAIT D3 standard valid825")
            time.sleep(60)
        profile_json = H81_PROFILE / "nts11_hardware_p0_profile.json"
        if not profile_json.is_file() or not (H81_TRACE / "manifest.json").is_file():
            run(
                [
                    sys.executable,
                    "-u",
                    str(EXP / "entrypoints/profile_nts11_hardware_p0.py"),
                    "--config",
                    str(H81_CONFIG),
                    "--checkpoint",
                    str(H81_CHECKPOINT),
                    "--output-dir",
                    str(H81_PROFILE),
                    "--samples",
                    "20",
                    "--num-workers",
                    "0",
                    "--bit-trace-dir",
                    str(H81_TRACE),
                    "--bit-trace-samples",
                    "20",
                    "--bit-trace-windows",
                    "1",
                    "--bit-trace-all-blocks",
                ],
                H81_PROFILE / "console.log",
                "H81 ep29 matched score-pair profile20",
            )
        if not PAIR_RESULT.is_file():
            run(
                [
                    sys.executable,
                    "-u",
                    str(EXP / "entrypoints/analyze_h67_h81_score_pairs_20260821.py"),
                    "--h67-trace-dir",
                    str(H67_TRACE),
                    "--h81-trace-dir",
                    str(H81_TRACE),
                    "--sample-limit",
                    "20",
                    "--output",
                    str(PAIR_RESULT),
                ],
                ROOT / "score_pair_analysis.log",
                "H67/H81 classified score-pair analysis",
            )
        run(
            [
                sys.executable,
                "-u",
                str(EXP / "entrypoints/make_date_fullres_factorial_controls_20260821.py"),
            ],
            ROOT / "factorial_config_generation.log",
            "generate factorial configs",
        )
        run(
            [
                sys.executable,
                "-u",
                str(EXP / "entrypoints/run_date_fullres_factorial_controls_20260821.py"),
            ],
            ROOT / "factorial_queue.log",
            "DATE full-resolution factorial queue",
        )
        record("ALL COMPLETE DATE algorithm evidence queue; official DSEC skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
