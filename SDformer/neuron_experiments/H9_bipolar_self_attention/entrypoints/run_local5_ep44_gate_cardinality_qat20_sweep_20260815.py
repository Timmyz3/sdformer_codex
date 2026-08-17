#!/usr/bin/env python3
"""Run a bounded four-branch ep44 Q7 gate-cardinality QAT screen."""

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
CALIBRATION_ROOT = RESULTS / "dsec_fullres_w15_local5_ep44_gatecard_q7_calibration1_20260815"
CALIBRATION = CALIBRATION_ROOT / "calibration_receipt.json"
SOURCE_ROOT = RESULTS / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812"
SOURCE_LABEL = 44
ROOT = RESULTS / "local5_ep44_gatecard_qat20_sweep_20260815"
STATUS = ROOT / "status.log"
SUMMARY = ROOT / "summary.json"
LOCK = Path("/tmp/sdformer_local5_ep44_gatecard_qat20_sweep.lock")
STEPS = 20
EXPERIMENT_PREFIX = "dsec_fullres_w15_local5_ep44_gatecard"


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


def load_calibration() -> dict:
    raw = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if raw.get("status") != "PASS":
        raise RuntimeError("gate-cardinality calibration did not pass")
    return raw


def branch_specs(calibration: dict) -> list[tuple[str, float, float]]:
    candidates = calibration["candidate_lambdas"]
    return [
        ("control", 0.0, 0.0),
        ("r001", 0.001, float(candidates["flow_loss_ratio_0.001"])),
        ("r005", 0.005, float(candidates["flow_loss_ratio_0.005"])),
        ("r010", 0.010, float(candidates["flow_loss_ratio_0.01"])),
    ]


def prepare_branch(name: str, ratio: float, weight: float) -> tuple[Path, Path]:
    config = yaml.safe_load(
        Path(json.loads(CALIBRATION.read_text(encoding="utf-8"))["calibration_config"])
        .read_text(encoding="utf-8")
    )
    experiment = f"{EXPERIMENT_PREFIX}_qat{STEPS}_{name}"
    config["experiment"] = experiment
    attention = config["bsa_attention"]
    attention["source_gate_cardinality_regularization_weight"] = weight
    attention["source_gate_cardinality_log_interval_steps"] = 5 if weight > 0.0 else 0
    config["loader"]["n_epochs"] = SOURCE_LABEL + 2
    config["test"]["sample"] = 40
    runtime = config["runtime"]
    runtime.update(
        {
            "max_train_steps": STEPS,
            "skip_save": False,
            "skip_state_save": True,
            "force_save_epochs": [SOURCE_LABEL + 1],
            "state_save_epochs": [],
            "save_only_force_epochs": True,
            "gate_cardinality_calibration_only": False,
            "gate_cardinality_short_sweep": True,
            "gate_cardinality_target_flow_loss_ratio": ratio,
        }
    )
    config["note"] = (
        f"{STEPS}-step Q7 hardware-order Local5 gate-cardinality screen from ep44; "
        f"target initial loss ratio={ratio}, lambda={weight}. This is a bounded "
        "go/no-go screen, not a converged model or paper result."
    )

    config_path = GENERATED / f"{experiment}.yml"
    rendered = yaml.safe_dump(config, sort_keys=False, width=100)
    if config_path.is_file() and config_path.read_text(encoding="utf-8") != rendered:
        raise RuntimeError(f"sweep config drift: {config_path}")
    if not config_path.is_file():
        config_path.write_text(rendered, encoding="utf-8")

    branch = ROOT / name
    branch.mkdir(parents=True, exist_ok=True)
    for suffix in (".pth", "_state_dict.pth"):
        source = SOURCE_ROOT / f"checkpoint_epoch{SOURCE_LABEL}{suffix}"
        staged = branch / source.name
        if not staged.exists():
            os.link(source, staged)
        if staged.stat().st_ino != source.stat().st_ino:
            raise RuntimeError(f"staged checkpoint identity mismatch: {staged}")
    state = torch.load(
        branch / f"checkpoint_epoch{SOURCE_LABEL}_state_dict.pth",
        map_location="cpu",
        weights_only=False,
    )
    if int(state.get("epoch", -1)) != SOURCE_LABEL:
        raise RuntimeError(f"{name}: source state is not ep44")
    return config_path, branch


def run_branch(name: str, config: Path, branch: Path) -> None:
    checkpoint = branch / f"checkpoint_epoch{SOURCE_LABEL + 1}.pth"
    log = branch / "train.log"
    if checkpoint.is_file():
        record(f"SKIP {name}: candidate checkpoint already exists")
        return
    if log.exists() and log.stat().st_size:
        raise RuntimeError(f"{name}: partial log exists without checkpoint: {log}")
    command = [
        sys.executable,
        "-u",
        str(EXP / "entrypoints/train.py"),
        "--config",
        str(config),
        "--prev_runid",
        str(branch / f"checkpoint_epoch{SOURCE_LABEL}.pth"),
        "--save_path",
        str(branch / "checkpoint_epoch{}.pth"),
        "--finetune",
        "1",
        "--resume",
        "1",
    ]
    record(f"START {name}: {' '.join(command)}")
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=REPO,
            env=environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    record(f"END {name}: exit_code={result.returncode}")
    if result.returncode:
        raise RuntimeError(f"{name} failed; inspect {log}")


def parse_branch(name: str, ratio: float, weight: float, config: Path, branch: Path) -> dict:
    log = branch / "train.log"
    checkpoint = branch / f"checkpoint_epoch{SOURCE_LABEL + 1}.pth"
    text = log.read_text(encoding="utf-8", errors="replace")
    train_matches = re.findall(r"Epoch loss = ([0-9.eE+-]+)", text)
    valid_matches = re.findall(r"Epoch loss \(Validation\): ([0-9.eE+-]+)", text)
    proxy_matches = re.findall(
        r"\[H9-GC\] step (\d+): flow_loss=([0-9.eE+-]+), "
        r"unweighted_proxy=([0-9.eE+-]+), weighted_penalty=([0-9.eE+-]+)",
        text,
    )
    checks = {
        "checkpoint exists": checkpoint.is_file(),
        "one train loss": len(train_matches) == 1,
        "one validation loss": len(valid_matches) == 1,
        "twenty-step early stop": f"max_train_steps={STEPS}" in text,
        "load audit clean": (
            "checkpoint_overlay_keys=210, missing=0, unexpected=0" in text
        ),
        "no traceback": "Traceback (most recent call last)" not in text,
        "proxy cadence": len(proxy_matches) == (STEPS // 5 if weight > 0.0 else 0),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"{name} seal failed: {failed}")
    proxies = [
        {
            "step": int(step),
            "flow_loss": float(flow_loss),
            "unweighted_proxy": float(proxy),
            "weighted_penalty": float(penalty),
        }
        for step, flow_loss, proxy, penalty in proxy_matches
    ]
    return {
        "name": name,
        "target_initial_loss_ratio": ratio,
        "lambda": weight,
        "train_loss": float(train_matches[0]),
        "validation_loss": float(valid_matches[0]),
        "proxy_samples": proxies,
        "proxy_first": proxies[0]["unweighted_proxy"] if proxies else None,
        "proxy_last": proxies[-1]["unweighted_proxy"] if proxies else None,
        "proxy_change_fraction": (
            (proxies[-1]["unweighted_proxy"] / proxies[0]["unweighted_proxy"] - 1.0)
            if proxies and proxies[0]["unweighted_proxy"] > 0.0
            else None
        ),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "config": str(config.resolve()),
        "config_sha256": sha256(config),
        "train_log": str(log.resolve()),
        "train_log_sha256": sha256(log),
        "checks": checks,
    }


def seal(rows: list[dict], calibration: dict) -> None:
    control = next(row for row in rows if row["name"] == "control")
    for row in rows:
        row["validation_loss_delta_vs_control"] = (
            row["validation_loss"] / control["validation_loss"] - 1.0
        )
    output = {
        "schema": f"local5_ep44_gate_cardinality_qat{STEPS}_sweep_v1",
        "status": "HOLD_FIXED_TRACE_REQUIRED",
        "evidence_level": f"{len(rows)}_branch_{STEPS}_step_fullres_q7_training_screen",
        "claim_boundary": (
            "Not converged; no AEE, exact gate-cardinality distribution, RTL, cycle, "
            "energy, encoder, or PPA claim."
        ),
        "source_checkpoint_label": SOURCE_LABEL,
        "source_checkpoint_sha256": sha256(
            SOURCE_ROOT / f"checkpoint_epoch{SOURCE_LABEL}.pth"
        ),
        "calibration_receipt": str(CALIBRATION.resolve()),
        "calibration_receipt_sha256": sha256(CALIBRATION),
        "calibration_flow_loss": calibration["flow_loss"],
        "calibration_proxy": calibration["unweighted_gate_cardinality_proxy"],
        "steps": STEPS,
        "selection_rule": (
            "No branch selection from training-step proxy samples because each logged "
            "step sees a different batch. Compare candidate checkpoints on one fixed "
            "ordered trace before any go/no-go decision."
        ),
        "selected_branch": None,
        "fixed_trace_probe_priority": [
            row["name"] for row in reversed(rows) if row["name"] != "control"
        ],
        "branches": rows,
    }
    SUMMARY.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    record(f"SEALED status={output['status']} selected={output['selected_branch']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reseal-only", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("gate-cardinality QAT20 sweep is already active") from exc
        calibration = load_calibration()
        prepared = []
        for name, ratio, weight in branch_specs(calibration):
            config, branch = prepare_branch(name, ratio, weight)
            prepared.append((name, ratio, weight, config, branch))
        record(f"PREPARED {len(prepared)}-branch QAT{STEPS} sweep")
        if args.prepare_only:
            return 0
        if SUMMARY.is_file() and not args.reseal_only:
            record("ALL COMPLETE summary already exists")
            return 0
        rows = []
        for name, ratio, weight, config, branch in prepared:
            if not args.reseal_only:
                run_branch(name, config, branch)
            rows.append(parse_branch(name, ratio, weight, config, branch))
        seal(rows, calibration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
