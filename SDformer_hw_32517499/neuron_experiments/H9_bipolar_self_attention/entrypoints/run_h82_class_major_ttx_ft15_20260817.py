#!/usr/bin/env python3
"""Fine-tune H82 Class-Major TTX from H81 ep29. Does not write hardware docs."""

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
CONFIG = EXP / "configs/generated/dsec_fullres_w15_H82_class_major_ttx_ft15.yml"
INIT = (
    EXP
    / "results/dsec_fullres_w15_H81_nomotion_bb1e4_ft40_20260811/checkpoint_epoch29.pth"
)
ROOT = EXP / "results/dsec_fullres_w15_H82_class_major_ttx_ft15_20260817"
LOCK = Path("/tmp/sdformer_h82_class_major_ttx.lock")
STATUS = ROOT / "status.log"
CONTRACT = REPO / "neuron_autoresearch/H82_CLASS_MAJOR_TTX_OPERATOR_CONTRACT_20260817.json"
PY = Path("/opt/conda/envs/sdformerflow/bin/python")
OPERATOR = EXP / "overlay/models/STSwinNet_SNN/bsa_attention.py"


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


def freeze_contract() -> None:
    payload = {
        "schema": "h82_class_major_ttx_operator_contract_v1",
        "status": "OPERATOR_FROZEN_TRAINING",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "c8": ["C8.3_class_major_projection", "C8.1_class_stability_regularizer"],
        "forbidden": [
            "Motion-XOR",
            "Local5 stencil",
            "multiplicity-weighted equivalent Shiftmax",
            "Prosperity/Bishop/FuseMax/FLAT/FusionArch/Ditto paste",
        ],
        "parent": {
            "line": "H81_no_motion",
            "checkpoint": str(INIT),
            "checkpoint_sha256": sha256(INIT),
        },
        "artifacts": {
            "operator_py": {"path": str(OPERATOR), "sha256": sha256(OPERATOR)},
            "config": {"path": str(CONFIG), "sha256": sha256(CONFIG)},
            "contract_md": {
                "path": str(CONTRACT.with_suffix(".md")),
                "sha256": sha256(CONTRACT.with_suffix(".md")),
            },
        },
        "isa": "Class File (class_id, multiplicity, temporal_mask, gate_c); K expand after class Shiftmax",
    }
    CONTRACT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    record(f"FROZE operator contract {CONTRACT} sha={sha256(CONTRACT)}")


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("H82 trainer already active", flush=True)
            return 0
        if not CONFIG.is_file() or not INIT.is_file():
            raise FileNotFoundError("H82 config or H81 init missing")
        freeze_contract()
        if (ROOT / "checkpoint_epoch14.pth").is_file():
            record("ALL COMPLETE H82 ft15 already has epoch14")
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
            log_handle.flush()
            proc = subprocess.run(
                command,
                cwd=REPO,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            log_handle.write(f"\n[h82-ft15] exit_code={proc.returncode}\n")
        if proc.returncode:
            raise RuntimeError(f"H82 train failed; log={log}")
        record("ALL COMPLETE H82 ft15")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
