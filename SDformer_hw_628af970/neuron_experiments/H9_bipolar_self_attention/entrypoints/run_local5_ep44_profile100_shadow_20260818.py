#!/usr/bin/env python3
"""Local5 fullres ep44: 100-sample aggregate per-operator activity profile.

SHADOW variant (2026-08-18): the disk overlay bsa_attention.py was modified by
the concurrent D1 (T>2 temporal operator) agent at 18:31 UTC (disk SHA is now
a8e94f56..., NOT the frozen 66d0a339). P1-4 must stay on the frozen overlay to
remain comparable with the H67 ep35 profile (P0-3, 66d0a339) and the Local5
provenance chain. This launcher therefore runs a byte-identical copy of the
profiler/trace_writer (SHAs 5f21c8d7/75c91340 verified below) from a shadow tree
in /tmp whose overlay/ contains the pristine 66d0a339 bsa_attention.py (taken
from /tmp/bsa_attention_pristine_20260818.py, the copy saved by the D1 agent
itself before its edits). The disk overlay file is NOT touched.

Closes gap P1-4 (CLAUDE_DATE_EXPERIMENT_GAPS_20260818.md): per-operator
full-encoder activity share for the hw-side Amdahl input.
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
SHADOW = Path("/tmp/p1_4_shadow/neuron_experiments/H9_bipolar_self_attention")
PY = Path("/opt/conda/envs/sdformerflow/bin/python")
PROFILER = SHADOW / "entrypoints/profile_nts11_hardware_p0.py"
SHADOW_OPERATOR = SHADOW / "overlay/models/STSwinNet_SNN/bsa_attention.py"
DISK_OPERATOR = EXP / "overlay/models/STSwinNet_SNN/bsa_attention.py"
CONFIG = GEN / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_hardware_order_q7q17_deploy.yml"
CKPT = RESULTS / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/checkpoint_epoch44.pth"
OUT = RESULTS / "local5_fullres_ep44_t450_profile100_20260818"
EXPECTED_PROFILER_SHA = "5f21c8d7eae27e251edfc07d9cfa3a75307893d1d2a5e1a522b1fa027c4bdf22"
EXPECTED_WRITER_SHA = "75c9134061aa06c8050389cbaac0a80a7956911cda0f8ce7b4144ba40ab3f58e"
EXPECTED_OPERATOR_SHA = "66d0a339fec374537ef21f81ee0689d000ec1a4340a7821e49116604510fb483"
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
    disk_sha = sha256(DISK_OPERATOR)
    record(f"[WARN] disk overlay SHA (modified by D1 agent): {disk_sha[:16]}... (expected frozen 66d0a339; shadow used instead)")
    if sha256(SHADOW_OPERATOR) != EXPECTED_OPERATOR_SHA:
        raise RuntimeError("shadow overlay SHA mismatch; abort")
    if sha256(PROFILER) != EXPECTED_PROFILER_SHA:
        raise RuntimeError("shadow profiler SHA mismatch; abort")
    if sha256(SHADOW / "entrypoints/h67_bit_trace.py") != EXPECTED_WRITER_SHA:
        raise RuntimeError("shadow trace_writer SHA mismatch; abort")
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
        "shadow_run": {
            "reason": "disk overlay modified by D1 agent at 18:31Z; frozen 66d0a339 used via /tmp shadow tree",
            "disk_overlay_sha256": disk_sha,
            "shadow_overlay_sha256": EXPECTED_OPERATOR_SHA,
        },
        "profile": str(profile),
        "profile_sha256": sha256(profile),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    record(f"[DONE] {summary['profile']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
