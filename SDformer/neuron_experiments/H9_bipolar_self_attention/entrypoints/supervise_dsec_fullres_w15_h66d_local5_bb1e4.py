"""Restart the Local-5 fullres pipeline after recoverable process failures."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import run_dsec_fullres_w15_h66d_local5_bb1e4_full_pipeline as pipeline


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
PIPELINE = EXP / "entrypoints/run_dsec_fullres_w15_h66d_local5_bb1e4_full_pipeline.py"
ROOT = EXP / "results/dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805"
STATUS = ROOT / "supervisor.log"
PIPELINE_LOG = ROOT / "launcher.log"
LOCK = ROOT / "supervisor.lock"
FINAL_MARKER = "ALL COMPLETE Local-5 bb1e4 train/eval/deploy/post-G0 pipeline"
PYTHON = "/opt/conda/envs/sdformerflow/bin/python"


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def complete() -> bool:
    pipeline_status = ROOT / "status.log"
    if not pipeline_status.is_file() or FINAL_MARKER not in pipeline_status.read_text(
        encoding="utf-8", errors="replace"
    ):
        return False
    try:
        pipeline.validate_checkpoint_contract()
        for epoch in pipeline.EVAL_EPOCHS:
            checkpoint = ROOT / f"checkpoint_epoch{epoch}.pth"
            pipeline.validate_eval_profile_contract(
                ROOT / f"standard_valid825/epoch{epoch}/spike_profile.json",
                checkpoint,
            )
        rank1 = pipeline.best_epoch()
        checkpoint = ROOT / f"checkpoint_epoch{rank1}.pth"
        pipeline.validate_deploy_summary_contract(checkpoint)
        pipeline.validate_profile_acceptance(
            checkpoint
        )
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False
    return True


def pipeline_pids() -> list[int]:
    found: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        try:
            process_cwd = (entry / "cwd").resolve()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for item in argv:
            argument = item.decode(errors="replace")
            if Path(argument).name != PIPELINE.name:
                continue
            candidate = Path(argument)
            if not candidate.is_absolute():
                candidate = process_cwd / candidate
            try:
                matches = candidate.resolve() == PIPELINE
            except (OSError, RuntimeError):
                matches = False
            if matches:
                found.append(int(entry.name))
                break
    return found


def start_pipeline() -> int:
    env = os.environ.copy()
    env.update(
        {
            "SDFORMER_USE_MLFLOW": "0",
            "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
            "SDFORMER_SNN_BACKEND": "cupy",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    with PIPELINE_LOG.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [PYTHON, "-u", str(PIPELINE)],
            cwd=REPO,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    return process.pid


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            record("SKIP supervisor lock already held")
            return 0

        restarts = 0
        while not complete():
            pids = pipeline_pids()
            if pids:
                time.sleep(120)
                continue
            if restarts >= 5:
                raise RuntimeError("Local-5 pipeline exceeded five automatic restarts")
            restarts += 1
            pid = start_pipeline()
            record(f"RESTART pipeline attempt={restarts} pid={pid}")
            time.sleep(120)
        record(f"ALL COMPLETE supervisor restarts={restarts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
