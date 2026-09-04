#!/usr/bin/env python3
"""Hash H82 force_save checkpoints. Refuses GPU AEE while ft15 is alive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(
    "/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/"
    "H9_bipolar_self_attention/results/dsec_fullres_w15_H82_class_major_ttx_ft15_20260817"
)
LOCK = Path("/tmp/sdformer_h82_class_major_ttx.lock")
TRAIN_PID = Path("/proc/2133687")
OUT = Path(__file__).resolve().parents[1] / "results" / "h82_class_file_isa_20260817"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(ROOT.glob("checkpoint_epoch*.pth"))
    train_alive = TRAIN_PID.exists() or LOCK.exists()
    payload = {
        "schema": "h82_ft15_ckpt_receipt_v1",
        "train_alive": train_alive,
        "aee_eval": "REFUSED_WHILE_TRAIN_OWNS_GPU" if train_alive else "gpu_free_not_started",
        "checkpoints": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in ckpts
        ],
        "note": "AEE/hardware-order must wait until this ft15 releases the A800.",
    }
    (OUT / "ep4_receipt.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
