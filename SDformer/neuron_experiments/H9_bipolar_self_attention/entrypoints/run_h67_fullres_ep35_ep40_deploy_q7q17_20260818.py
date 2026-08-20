#!/usr/bin/env python3
"""H67 fullres (DATE mainline) ep35/ep40 hardware-order + dyadic numeric valid825.

Closes gap P0-1 (CLAUDE_DATE_EXPERIMENT_GAPS_20260818.md): the DATE mainline
checkpoint (H67 Motion-TTX fullres ep35) has float standard_valid825 but no
hardware-order / dyadic numeric evaluation. Local5 ep44 already has the same
pair (deploy_valid825/{hardware_order_q7q17,dyadic_q7q17}/epoch44, 2026-08-15).

Protocol: eval_DSEC_flow_SNN.py --mode valid, batch_size=1, 825 frames /
18 sequences, same as the four-line standard protocol and the Local5 deploy
precedent. hardware_order = Q7 score + Q8 LUT integer Shiftmax + Q1.7 RNE gate;
dyadic = Q7 score + float exp2 Shiftmax + Q1.7 RNE gate.

Operator provenance: disk overlay bsa_attention.py sha=66d0a339... is the
H83-H86 stacked version. The H67 h60/Motion numeric path used here is the same
disk version that produced the four-line ledger numbers (8-10 standard eval,
8-15 multisample10 trace) and the H82 analysis documented h83-h86 as pure
additions. No file swap is performed.
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
ROOT = RESULTS / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805"
OPERATOR = EXP / "overlay/models/STSwinNet_SNN/bsa_attention.py"
PY = Path("/opt/conda/envs/sdformerflow/bin/python")
EVAL = REPO / "third_party/SDformerFlow/eval_DSEC_flow_SNN.py"
DISK_OP_SHA = "66d0a339fec374537ef21f81ee0689d000ec1a4340a7821e49116604510fb483"
LOG = ROOT / "deploy_valid825_q7q17_20260818_watcher.log"

CASES = (
    # (kind, deploy config, epoch, expected quant contract)
    ("hardware_order", GEN / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml", 35),
    ("hardware_order", GEN / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml", 40),
    ("dyadic", GEN / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_dyadic_q7q17_deploy.yml", 35),
    ("dyadic", GEN / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_dyadic_q7q17_deploy.yml", 40),
)


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


def run_eval(config: Path, checkpoint: Path, output: Path) -> dict:
    profile = output / "spike_profile.json"
    if profile.exists():
        record(f"[reuse] profile exists: {profile}")
        return json.loads(profile.read_text(encoding="utf-8"))
    env = os.environ.copy()
    env.update({
        "SDFORMER_USE_MLFLOW": "0",
        "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
        "SDFORMER_SNN_BACKEND": "cupy",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(PY), "-u", str(EVAL),
        "--config", str(config), "--checkpoint", str(checkpoint),
        "--path_results", str(output), "--mode", "valid",
    ]
    record("$ " + " ".join(command))
    with (output / "eval.log").open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        proc = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"eval failed: {output / 'eval.log'}")
    return json.loads(profile.read_text(encoding="utf-8"))


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    op_sha = sha256(OPERATOR)
    if op_sha != DISK_OP_SHA:
        raise RuntimeError(f"operator disk SHA changed: {op_sha}")
    record(f"operator disk sha ok: {op_sha[:16]}... (H83-H86 stacked, unchanged)")

    results = []
    for kind, config, epoch in CASES:
        checkpoint = ROOT / f"checkpoint_epoch{epoch}.pth"
        output = ROOT / "deploy_valid825" / f"{kind}_q7q17" / f"epoch{epoch}"
        if not checkpoint.exists():
            record(f"[skip] missing checkpoint {checkpoint}")
            continue
        profile = run_eval(config, checkpoint, output)
        metrics = profile.get("metrics", {})
        row = {
            "kind": kind,
            "epoch": epoch,
            "config": str(config),
            "config_sha256": sha256(config),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "AEE": metrics.get("AEE"),
            "AAE": metrics.get("AAE"),
            "AAE_Benchmark": metrics.get("AAE_Benchmark"),
            "DSEC_Fl": metrics.get("DSEC_Fl"),
            "total_spikes_g": profile.get("total_spikes", 0) / 1e9,
            "energy_uj": profile.get("energy_uj"),
            "samples": profile.get("samples"),
            "profile": str(output / "spike_profile.json"),
            "profile_sha256": sha256(output / "spike_profile.json"),
            "overlay_missing": profile.get("overlay_missing"),
            "overlay_unexpected": profile.get("overlay_unexpected"),
        }
        results.append(row)
        record(
            f"[done] {kind} ep{epoch}: AEE={row['AEE']} AAE={row['AAE']} "
            f"Fl={row['DSEC_Fl']} spikes={row['total_spikes_g']:.3f}G "
            f"energy={row['energy_uj']}uJ"
        )

    summary = {
        "schema": "h67_fullres_ep35_ep40_deploy_q7q17_valid825_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "population": "DSEC local valid825, 825 frames / 18 sequences, batch=1",
        "operator_disk_sha": op_sha,
        "operator_note": "H83-H86 stacked; h60/Motion path unchanged since 8-10/8-15 receipts",
        "results": results,
    }
    summary_path = RESULTS / "h67_fullres_ep35_ep40_deploy_q7q17_valid825_20260818.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    record(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
