#!/usr/bin/env python3
"""Run the full-resolution H81 no-motion control after direct MVSEC completes."""

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
CONFIG = EXP / "configs/generated/dsec_fullres_w15_H81_nomotion_bb1e4_ft40.yml"
SOURCE = (
    EXP
    / "results/h81_allbinary_all12_h60_nomotion_equalbudget_w720_fastlr_full30_bs8_full30_20260717_setsid/checkpoint_epoch19.pth"
)
ROOT = EXP / "results/dsec_fullres_w15_H81_nomotion_bb1e4_ft40_20260811"
STATUS = ROOT / "status.log"
PIPELINE_DONE = EXP / "results/mvsec_cicc_nb0_h67_local5_comparison_20260811.json"
LOCK = Path("/tmp/sdformer_dsec_fullres_h81_nomotion.lock")
EVAL_EPOCHS = (29, 34, 39)


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


def audit_load() -> None:
    text = (ROOT / "train.log").read_text(encoding="utf-8", errors="replace")
    required = (
        "installed ATLIFTernaryPSN before load: 105 modules",
        "installed Shiftmax attention: 12 modules",
        "checkpoint_overlay_keys=210, missing=0, unexpected=0",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"H81 fullres load audit failed: {missing}")


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("H81 fullres watcher already active", flush=True)
            return 0

        for required in (CONFIG, SOURCE):
            if not required.is_file():
                raise FileNotFoundError(required)

        while not PIPELINE_DONE.is_file():
            record("WAIT direct-MVSEC NB0/H67/Local5 comparison")
            time.sleep(300)

        ranking = ROOT / "profile_ranking_valid825.md"
        if ranking.is_file():
            record("ALL COMPLETE H81 no-motion fullres40 control")
            return 0

        final_checkpoint = ROOT / "checkpoint_epoch39.pth"
        if not final_checkpoint.is_file():
            run(
                [
                    sys.executable,
                    "-u",
                    str(EXP / "entrypoints/train.py"),
                    "--config",
                    str(CONFIG),
                    "--prev_runid",
                    str(SOURCE),
                    "--save_path",
                    str(ROOT / "checkpoint_epoch{}.pth"),
                    "--finetune",
                    "1",
                ],
                ROOT / "train.log",
                "H81 no-motion DSEC fullres40",
            )
        audit_load()

        epoch_args = [item for epoch in EVAL_EPOCHS for item in ("--epoch", str(epoch))]
        run(
            [
                sys.executable,
                "-u",
                str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
                "--config",
                str(CONFIG),
                "--run-dir",
                str(ROOT),
                "--ranking-mode",
                "aee",
                *epoch_args,
            ],
            ROOT / "valid825.log",
            "H81 no-motion DSEC fullres Valid825 epochs 29/34/39",
        )
        record("ALL COMPLETE H81 no-motion fullres40 control")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
