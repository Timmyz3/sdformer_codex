#!/usr/bin/env python3
"""Fine-tune H86 Member-Delta Class File from H81 ep29, per the frozen
H86_MEMBER_DELTA_CLASS_FILE_CONTRACT_20260818.json.

Fail-closed: verifies the frozen operator/config/tests SHAs before launching.
Does not write hardware docs. Does not touch the DATE model (H67).
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
CONFIG = EXP / "configs/generated/dsec_fullres_w15_H86_member_delta_class_file_ft15.yml"
INIT = (
    EXP
    / "results/dsec_fullres_w15_H81_nomotion_bb1e4_ft40_20260811/checkpoint_epoch29.pth"
)
ROOT = EXP / "results/dsec_fullres_w15_H86_member_delta_class_file_ft15_20260818"
LOCK = Path("/tmp/sdformer_h86_member_delta.lock")
STATUS = ROOT / "status.log"
CONTRACT = (
    REPO
    / "neuron_autoresearch/H86_MEMBER_DELTA_CLASS_FILE_CONTRACT_20260818.json"
)
PY = Path("/opt/conda/envs/sdformerflow/bin/python")
OPERATOR = EXP / "overlay/models/STSwinNet_SNN/bsa_attention.py"
TESTS = EXP / "tests/test_h86_member_delta_class_file.py"

FROZEN = {
    "operator_py": OPERATOR,
    "config": CONFIG,
    "tests": TESTS,
}


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


def verify_contract() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("status") != "OPERATOR_FROZEN_WAITING_FOR_H82_GPU":
        raise RuntimeError(f"H86 contract status unexpected: {contract.get('status')}")
    parent_sha = contract["parent"]["checkpoint_sha256"]
    actual_parent = sha256(INIT)
    if actual_parent != parent_sha:
        raise RuntimeError(
            f"H86 parent checkpoint SHA mismatch: frozen={parent_sha[:16]} "
            f"actual={actual_parent[:16]}"
        )
    artifacts = contract["artifacts"]
    for name, path in FROZEN.items():
        expected = artifacts[name]["sha256"]
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"H86 frozen SHA mismatch for {name}: "
                f"frozen={expected[:16]} actual={actual[:16]}"
            )
        record(f"verified {name} sha={actual[:16]}")
    record(f"verified parent sha={actual_parent[:16]}")


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("H86 trainer already active", flush=True)
            return 0
        if not CONFIG.is_file() or not INIT.is_file():
            raise FileNotFoundError("H86 config or H81 init missing")
        verify_contract()
        if (ROOT / "checkpoint_epoch14.pth").is_file():
            record("ALL COMPLETE H86 ft15 already has epoch14")
            return 0
        env = os.environ.copy()
        env.update(
            {
                "SDFORMER_USE_MLFLOW": "0",
                "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
                "SDFORMER_SNN_BACKEND": "cupy",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        command = [
            str(PY),
            "-u",
            str(EXP / "entrypoints/train.py"),
            "--config",
            str(CONFIG),
            "--prev_runid",
            str(INIT),
            "--save_path",
            str(ROOT / "checkpoint_epoch{}.pth"),
            "--finetune",
            "1",
        ]
        record("START " + " ".join(command))
        log = ROOT / "train.log"
        with log.open("w", encoding="utf-8") as log_handle:
            log_handle.write("$ " + " ".join(command) + "\n")
            log_handle.write(f"# contract={CONTRACT}\n")
            log_handle.flush()
            proc = subprocess.run(
                command,
                cwd=REPO,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            log_handle.write(f"\n[h86-ft15] exit_code={proc.returncode}\n")
        if proc.returncode:
            raise RuntimeError(f"H86 train failed; log={log}")
        record("ALL COMPLETE H86 ft15")
        return 0


if __name__ == "__main__":
    sys.exit(main())
