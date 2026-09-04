#!/usr/bin/env python3
"""Run one fail-closed ep44 Q7 Local5 gate-cardinality calibration step."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import torch
import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
GENERATED = EXP / "configs/generated"
RESULTS = EXP / "results"
SOURCE_CONFIG = (
    GENERATED / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50.yml"
)
DEPLOY_CONFIG = GENERATED / (
    "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_"
    "hardware_order_q7q17_deploy.yml"
)
SOURCE_ROOT = RESULTS / (
    "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812"
)
SOURCE_LABEL = 44
PROXY_MODE = "mean_collapse"
EXPERIMENT = "dsec_fullres_w15_local5_ep44_gatecard_q7_calibration1"
CONFIG = GENERATED / f"{EXPERIMENT}.yml"
ROOT = RESULTS / f"{EXPERIMENT}_20260815"
STATUS = ROOT / "status.log"
LOG = ROOT / "train.log"
RECEIPT = ROOT / "calibration_receipt.json"
LOCK = Path("/tmp/sdformer_local5_ep44_gatecard_calibration.lock")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def build_config() -> dict:
    source = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    deploy = yaml.safe_load(DEPLOY_CONFIG.read_text(encoding="utf-8"))
    source["experiment"] = EXPERIMENT
    source_attention = source.setdefault("bsa_attention", {})
    deploy_attention = deploy["bsa_attention"]
    hardware_keys = (
        "hardware_quant_enabled",
        "hardware_mu_pow2_shift",
        "hardware_score_step",
        "hardware_score_min",
        "hardware_score_max",
        "hardware_gate_step",
        "hardware_gate_min",
        "hardware_gate_max",
        "hardware_mask_invalid_candidates",
        "hardware_rtl_shiftmax_enabled",
    )
    for key in hardware_keys:
        source_attention[key] = deploy_attention[key]
    source_attention["source_gate_cardinality_regularization_weight"] = 1.0
    source_attention["source_gate_cardinality_proxy_mode"] = PROXY_MODE
    source_attention["source_gate_cardinality_log_interval_steps"] = 1

    source["loader"]["n_epochs"] = SOURCE_LABEL + 2
    source["loader"]["persistent_workers"] = False
    source["test"]["sample"] = 2
    runtime = source.setdefault("runtime", {})
    runtime.update(
        {
            "max_train_steps": 1,
            "skip_save": True,
            "skip_state_save": True,
            "force_save_epochs": [],
            "state_save_epochs": [],
            "save_only_force_epochs": False,
            "gate_cardinality_calibration_only": True,
            "gate_cardinality_source_checkpoint_label": SOURCE_LABEL,
        }
    )
    source["note"] = (
        "One-step, no-save calibration from Local5 ep44 under frozen Q7/Q1.7 "
        "hardware-order semantics. Weight=1 measures the unweighted source-local "
        "gate-cardinality proxy scale; this is not a trained candidate or paper result."
    )
    return source


def prepare() -> None:
    for path in (SOURCE_CONFIG, DEPLOY_CONFIG):
        if not path.is_file():
            raise FileNotFoundError(path)
    ROOT.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(build_config(), sort_keys=False, width=100)
    if CONFIG.is_file() and CONFIG.read_text(encoding="utf-8") != rendered:
        raise RuntimeError(f"calibration config drift: {CONFIG}")
    if not CONFIG.is_file():
        CONFIG.write_text(rendered, encoding="utf-8")

    source_model = SOURCE_ROOT / f"checkpoint_epoch{SOURCE_LABEL}.pth"
    source_state = SOURCE_ROOT / f"checkpoint_epoch{SOURCE_LABEL}_state_dict.pth"
    for path in (source_model, source_state):
        if not path.is_file():
            raise FileNotFoundError(path)
    staged_model = ROOT / source_model.name
    staged_state = ROOT / source_state.name
    for source, staged in ((source_model, staged_model), (source_state, staged_state)):
        if not staged.exists():
            os.link(source, staged)
        if staged.stat().st_ino != source.stat().st_ino:
            raise RuntimeError(f"staged checkpoint is not the expected hardlink: {staged}")

    state = torch.load(staged_state, map_location="cpu", weights_only=False)
    if int(state.get("epoch", -1)) != SOURCE_LABEL:
        raise RuntimeError("ep44 calibration state identity mismatch")
    record("PREPARED ep44 Q7 gate-cardinality one-step calibration")


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


def run() -> None:
    command = [
        sys.executable,
        "-u",
        str(EXP / "entrypoints/train.py"),
        "--config",
        str(CONFIG),
        "--prev_runid",
        str(ROOT / f"checkpoint_epoch{SOURCE_LABEL}.pth"),
        "--save_path",
        str(ROOT / "checkpoint_epoch{}.pth"),
        "--finetune",
        "1",
        "--resume",
        "1",
    ]
    record("START " + " ".join(command))
    with LOG.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=REPO,
            env=environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    record(f"END exit_code={result.returncode}")
    if result.returncode:
        raise RuntimeError(f"calibration failed; inspect {LOG}")


def seal() -> None:
    text = LOG.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"\[H9-GC\] step 1: flow_loss=([0-9.eE+-]+), "
        r"unweighted_proxy=([0-9.eE+-]+), weighted_penalty=([0-9.eE+-]+)",
        text,
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one H9-GC calibration line, found {len(matches)}")
    flow_loss, proxy, penalty = map(float, matches[0])
    checks = {
        "one proxy measurement": len(matches) == 1,
        "finite positive flow loss": flow_loss > 0.0,
        "finite nonnegative proxy": proxy >= 0.0,
        "weight-one identity": abs(proxy - penalty) <= max(1e-8, abs(proxy) * 1e-6),
        "one-step early stop": "stopping train epoch early at max_train_steps=1" in text,
        "no checkpoint written": not any(ROOT.glob("checkpoint_epoch45.pth")),
        "no error": "Traceback (most recent call last)" not in text,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"calibration seal failed: {failed}")

    target_ratios = (0.001, 0.005, 0.01)
    lambdas = {
        f"flow_loss_ratio_{ratio:g}": (ratio * flow_loss / proxy if proxy > 0.0 else None)
        for ratio in target_ratios
    }
    receipt = {
        "schema": "local5_ep44_gate_cardinality_calibration_v1",
        "status": "PASS",
        "evidence_level": "one_real_fullres_training_batch_scale_measurement",
        "claim_boundary": (
            "No trained candidate, AEE, RTL, cycle, energy, encoder, or PPA claim."
        ),
        "source_checkpoint_label": SOURCE_LABEL,
        "proxy_mode": PROXY_MODE,
        "source_checkpoint_sha256": sha256(
            SOURCE_ROOT / f"checkpoint_epoch{SOURCE_LABEL}.pth"
        ),
        "source_config_sha256": sha256(SOURCE_CONFIG),
        "deploy_config_sha256": sha256(DEPLOY_CONFIG),
        "calibration_config": str(CONFIG.resolve()),
        "calibration_config_sha256": sha256(CONFIG),
        "train_log": str(LOG.resolve()),
        "train_log_sha256": sha256(LOG),
        "flow_loss": flow_loss,
        "unweighted_gate_cardinality_proxy": proxy,
        "weight_one_penalty": penalty,
        "candidate_lambdas": lambdas,
        "checks": checks,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    record("PASS ep44 Q7 gate-cardinality scale calibration")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="execute the one-step GPU calibration")
    args = parser.parse_args()
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("gate-cardinality calibration is already active") from exc
        prepare()
        if not args.run:
            record("PREPARE ONLY; pass --run to execute")
            return 0
        if RECEIPT.is_file():
            record("ALL COMPLETE receipt already exists")
            return 0
        run()
        seal()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
