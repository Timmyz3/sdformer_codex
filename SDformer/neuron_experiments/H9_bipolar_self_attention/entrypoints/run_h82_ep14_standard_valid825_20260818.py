#!/usr/bin/env python3
"""Standard valid825 eval of H82 ep14 under the frozen operator contract.

Provenance: the H82 operator was frozen at 2026-08-17T15:21:03Z as
bsa_attention.py sha=807a50e0c63f4800fbda778adf9e47b7a4dd2610138aec75ca105ca7e3ba2250
(recorded in H82_CLASS_MAJOR_TTX_OPERATOR_CONTRACT_20260817.json).

The disk overlay has since been extended with H83-H86 modes
(sha=66d0a339fec374537ef21f81ee0689d000ec1a4340a7821e49116604510fb483).
A line-level diff of frozen vs current shows exactly one changed line
(raise ValueError -> return None inside the class-stability regularizer
mode guard, unreachable for mode=h82 with the H82 config), plus pure
additions of h83-h86 functions and dispatch branches. The h82 numerical
path is therefore identical to the frozen operator. No file swap needed.

This launcher records both SHAs and the diff conclusion in the eval log,
then runs the same eval command used for H81 (batch_size=1, mode valid).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
CONFIG = EXP / "configs/generated/dsec_fullres_w15_H82_class_major_ttx_ft15.yml"
CKPT = (
    EXP
    / "results/dsec_fullres_w15_H82_class_major_ttx_ft15_20260817/checkpoint_epoch14.pth"
)
ROOT = EXP / "results/dsec_fullres_w15_H82_class_major_ttx_ft15_20260817"
OUT = ROOT / "standard_valid825" / "epoch14"
OPERATOR = EXP / "overlay/models/STSwinNet_SNN/bsa_attention.py"
PY = Path("/opt/conda/envs/sdformerflow/bin/python")
EVAL = REPO / "third_party/SDformerFlow/eval_DSEC_flow_SNN.py"
FROZEN_SHA = "807a50e0c63f4800fbda778adf9e47b7a4dd2610138aec75ca105ca7e3ba2250"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    with (ROOT / "valid825_ep14_watcher.log").open("a", encoding="utf-8") as h:
        h.write(line + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    op_sha = sha256(OPERATOR)
    provenance = {
        "schema": "h82_valid825_ep14_provenance_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "operator_frozen_sha": FROZEN_SHA,
        "operator_disk_sha": op_sha,
        "disk_diff_conclusion": (
            "one unreachable line changed (raise->return in h82 regularizer "
            "mode guard); h83-h86 additions only; h82 path numerically "
            "identical to frozen"
        ),
        "config": str(CONFIG),
        "config_sha": sha256(CONFIG),
        "checkpoint": str(CKPT),
        "checkpoint_sha": sha256(CKPT),
        "eval_command": [
            str(PY), "-u", "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
            "--config", str(CONFIG), "--checkpoint", str(CKPT),
            "--path_results", str(OUT), "--mode", "valid",
        ],
    }
    (OUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    record(f"provenance written; frozen={FROZEN_SHA[:16]} disk={op_sha[:16]}")

    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["SDFORMER_SNN_BACKEND"] = "cupy"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    with (OUT / "eval.log").open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(provenance["eval_command"]) + "\n")
        log.write(f"# operator_frozen_sha={FROZEN_SHA}\n")
        log.write(f"# operator_disk_sha={op_sha}\n")
        log.flush()
        record("launching eval")
        proc = subprocess.run(
            provenance["eval_command"], cwd=REPO, env=env,
            stdout=log, stderr=subprocess.STDOUT,
        )
        log.write(f"\n[h82-valid825-ep14] exit_code={proc.returncode}\n")
    record(f"eval finished exit_code={proc.returncode}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
