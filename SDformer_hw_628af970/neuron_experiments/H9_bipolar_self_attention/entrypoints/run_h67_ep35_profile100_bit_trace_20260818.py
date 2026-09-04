#!/usr/bin/env python3
"""H67 fullres ep35: 100-sample aggregate profile + raw Q/K bit trace export.

Closes gap P0-3 (CLAUDE_DATE_EXPERIMENT_GAPS_20260818.md): hw docs/399 requires
the algorithm side to export layered raw Q/K bit traces on the DATE mainline
checkpoint so the hardware side can run Fixed2S/RQTB2S 100-sample real RTL
replay (currently sealed at 10 samples: PASS_SEALED_COMPONENT_RTL_NOT_ENCODER).

Toolchain SHAs verified identical to the 2026-08-15 multisample10 run:
  profiler     = 5f21c8d7eae27e251edfc07d9cfa3a75307893d1d2a5e1a522b1fa027c4bdf22
  trace_writer = 75c9134061aa06c8050389cbaac0a80a7956911cda0f8ce7b4144ba40ab3f58e
Config/checkpoint identical to the multisample10 run_context:
  config     = dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml (8be3f7bb)
  checkpoint = checkpoint_epoch35.pth (4f33e086)
Population: first 100 frames of the DSEC local valid list (same as the frozen
valid825 density population), 1 window per sample, all 12 blocks.

Evidence tier: [prof]/[model] trace export; RTL replay belongs to the hw side.
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
HW_RESULTS = REPO / "hw_autoresearch_nts07/results"
OPERATOR = EXP / "overlay/models/STSwinNet_SNN/bsa_attention.py"
PY = Path("/opt/conda/envs/sdformerflow/bin/python")
PROFILER = Path(__file__).with_name("profile_nts11_hardware_p0.py")
CONFIG = GEN / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml"
CKPT = RESULTS / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth"
OUT = RESULTS / "h67_fullres_ep35_t450_profile100_20260818"
BIT_TRACE_DIR = HW_RESULTS / "h67_ep35_multisample100_t450_real_rtl_bit_trace"
EXPECTED_PROFILER_SHA = "5f21c8d7eae27e251edfc07d9cfa3a75307893d1d2a5e1a522b1fa027c4bdf22"
EXPECTED_WRITER_SHA = "75c9134061aa06c8050389cbaac0a80a7956911cda0f8ce7b4144ba40ab3f58e"
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
    prof_sha = sha256(PROFILER)
    writer_sha = sha256(Path(__file__).with_name("h67_bit_trace.py"))
    record(f"profiler sha ok: {prof_sha == EXPECTED_PROFILER_SHA} ({prof_sha[:16]}...)")
    record(f"trace_writer sha ok: {writer_sha == EXPECTED_WRITER_SHA} ({writer_sha[:16]}...)")
    record(f"config sha: {sha256(CONFIG)[:16]}...  checkpoint sha: {sha256(CKPT)[:16]}...")
    record(f"bit-trace-dir: {BIT_TRACE_DIR}")

    command = [
        str(PY), "-u", str(PROFILER),
        "--config", str(CONFIG),
        "--checkpoint", str(CKPT),
        "--output-dir", str(OUT),
        "--samples", str(SAMPLES),
        "--ordered-trace",
        "--bit-trace-dir", str(BIT_TRACE_DIR),
        "--bit-trace-samples", str(SAMPLES),
        "--bit-trace-windows", "1",
        "--bit-trace-all-blocks",
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

    manifest = BIT_TRACE_DIR / "manifest.json"
    if not manifest.exists():
        record("[FAIL] bit trace manifest missing")
        return 1
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    records = len(meta.get("records", []))
    if records != SAMPLES * 12:
        record(f"[FAIL] expected {SAMPLES * 12} records, got {records}")
        return 1
    npz_count = len(list(BIT_TRACE_DIR.glob("sample*_*.npz")))
    profile = OUT / "nts11_hardware_p0_profile.json"
    summary = {
        "schema": "h67_ep35_profile100_bit_trace_export_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "samples": SAMPLES,
        "bit_trace_records": records,
        "npz_count": npz_count,
        "bit_trace_manifest": str(manifest),
        "bit_trace_manifest_sha256": sha256(manifest),
        "profile": str(profile),
        "profile_sha256": sha256(profile) if profile.exists() else None,
        "toolchain": {"profiler": prof_sha, "trace_writer": writer_sha},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    record(f"[DONE] records={records} npz={npz_count}; summary={OUT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
