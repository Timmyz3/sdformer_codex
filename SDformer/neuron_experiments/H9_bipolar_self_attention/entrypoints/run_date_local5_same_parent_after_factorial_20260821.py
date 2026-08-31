#!/usr/bin/env python3
"""Run Local5 same-parent control after the five-cell factorial completes."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
GENERATED = EXP / "configs/generated"
MANIFEST = GENERATED / "date_fullres_local5_same_parent_control_20260821.json"
FACTORIAL_DONE = EXP / "results/date_fullres_factorial_controls_20260821/summary.json"
ROOT = EXP / "results/date_fullres_local5_same_parent_control_20260821"
STATUS = ROOT / "status.log"
LOCK = Path("/tmp/sdformer_date_local5_same_parent_control_20260821.lock")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            print("Local5 same-parent watcher already active", flush=True)
            return 0
        while not FACTORIAL_DONE.is_file():
            record("WAIT five-cell same-parent factorial summary")
            time.sleep(300)
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        config = Path(manifest["config"])
        parent = Path(manifest["parent_checkpoint"])
        if sha256(config) != manifest["config_sha256"]:
            raise RuntimeError("Local5 control config SHA changed")
        if sha256(parent) != manifest["parent_checkpoint_sha256"]:
            raise RuntimeError("Local5 control parent SHA changed")
        ranking = ROOT / "profile_ranking_valid825.md"
        checkpoint = ROOT / "checkpoint_epoch9.pth"
        if not checkpoint.is_file():
            run(
                [
                    sys.executable,
                    "-u",
                    str(EXP / "entrypoints/train.py"),
                    "--config",
                    str(config),
                    "--prev_runid",
                    str(parent),
                    "--save_path",
                    str(ROOT / "checkpoint_epoch{}.pth"),
                    "--finetune",
                    "1",
                ],
                ROOT / "train.log",
                "train C20 all-binary Local5",
            )
        text = (ROOT / "train.log").read_text(encoding="utf-8", errors="replace")
        required = (
            "installed ATLIFTernaryPSN before load: 105 modules",
            "installed attention before load: 12 modules",
            "checkpoint_overlay_keys=0, missing=210, unexpected=0",
            "installed Shiftmax attention: 12 modules",
        )
        missing = [marker for marker in required if marker not in text]
        if missing:
            raise RuntimeError(f"C20 initial load audit failed: {missing}")
        if not ranking.is_file():
            run(
                [
                    sys.executable,
                    "-u",
                    str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
                    "--config",
                    str(config),
                    "--run-dir",
                    str(ROOT),
                    "--ranking-mode",
                    "aee",
                    "--epoch",
                    "4",
                    "--epoch",
                    "9",
                ],
                ROOT / "valid825.log",
                "valid825 C20 epochs 4/9",
            )
        record("ALL COMPLETE C20 Local5 same-parent control")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
