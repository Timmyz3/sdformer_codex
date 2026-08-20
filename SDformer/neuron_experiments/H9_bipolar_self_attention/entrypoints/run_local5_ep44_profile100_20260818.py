#!/usr/bin/env python3
"""Local5 fullres ep44: 100-sample aggregate per-operator activity profile.

Closes gap P1-4 (CLAUDE_DATE_EXPERIMENT_GAPS_20260818.md): the hardware side
needs a full-encoder operator activity fraction (ATLIF/attention/FFN/residual/
decoder) for the full-network Amdahl estimate. H67 ep35 is covered by the P0-3
profile100 run (same profiler, ordered trace, per-block/stage/operator CSVs);
this run provides the same structure for Local5 ep44 (the accuracy rank-1).

Protocol: profile_nts11_hardware_p0.py, first 100 frames of the DSEC local
valid list, batch=1, no bit trace (P0-3 covers the H67 raw-trace side).
Evidence tier: [prof].
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
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
OPERATOR = EXP / "overlay/models/STSwinNet_SNN/bsa_attention.py"
PY = Path("/opt/conda/envs/sdformerflow/bin/python")
PROFILER = Path(__file__).with_name("profile_nts11_hardware_p0.py")
CONFIG = GEN / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_hardware_order_q7q17_deploy.yml"
CKPT = RESULTS / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/checkpoint_epoch44.pth"
OUT = RESULTS / "local5_fullres_ep44_t450_profile100_20260818"
SAMPLES = 100
LOG = OUT / "status.log"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if sha256(OPERATOR) != "66d0a339fec374537ef21f81ee0689d000ec1a4340a7821e49116604510fb483":
        raise RuntimeError("operator disk SHA changed; abort")
    record(f"config sha: {sha256(CONFIG)[:16]}... checkpoint sha: {sha256(CKPT)[:16]}...")
    command = [
        str(PY), "-u", str(PROFILER),
        "--config", str(CONFIG),
        "--checkpoint", str(CKPT),
        "--output-dir", str(OUT),
        "--samples", str(SAMPLES),
        "--ordered-trace",
    ]
    record("$ " + " ".join(command))
    env = os.environ.copy()
    env.update({
        "SDFORMER_USE_MLFLOW": "0",
        "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
        "SDFORMER_SNN_BACKEND": "cupy",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    proc = subprocess.run(command, cwd=REPO, env=env)
    if proc.returncode != 0:
        record(f"[FAIL] profiler exit={proc.returncode}")
        return 1
    profile = OUT / "nts11_hardware_p0_profile.json"
    summary = {
        "schema": "local5_ep44_profile100_activity_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "samples": SAMPLES,
        "profile": str(profile),
        "profile_sha256": sha256(profile),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    record(f"[DONE] {summary['profile']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
