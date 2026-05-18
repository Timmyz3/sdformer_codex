"""Autonomous I16 sweep for balanced negative ATLIF trigger scales.

The script generates H9a-compatible configs, runs short continuation tests,
selects a candidate, launches a full run, and profiles the resulting checkpoint.
It intentionally calls the existing H9 entrypoints so baseline code stays
untouched.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import os
import re
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


ROOT = repo_root()
EXP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXP_ROOT / "generated_configs"
RESULTS_DIR = EXP_ROOT / "results"
BASE_CONFIG = ROOT / "neuron_experiments/H9_bipolar_self_attention/configs/h9a_shiftmax_compat_h8m_full.yml"
H9_TRAIN = ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py"
H9_PROFILE = ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_sops.py"
H9_OVERLAY = ROOT / "neuron_experiments/H9_bipolar_self_attention/overlay"
BASELINE_CKPT = ROOT / "experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_base_config() -> dict[str, Any]:
    with BASE_CONFIG.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def make_config(
    *,
    tag: str,
    negative_scale: float,
    lambda_ang: float,
    max_train_steps: int,
    n_epochs: int,
) -> Path:
    cfg = deepcopy(load_base_config())
    cfg["experiment"] = tag
    cfg.setdefault("runtime", {})
    cfg["runtime"]["max_train_steps"] = int(max_train_steps)
    cfg["runtime"]["skip_state_save"] = True
    cfg["runtime"]["skip_save"] = False
    cfg["runtime"]["use_mlflow_model_logging"] = False
    cfg.setdefault("atlif_ternary_psn", {})
    cfg["atlif_ternary_psn"]["negative_threshold_scale"] = float(negative_scale)
    cfg["atlif_ternary_psn"]["log_interval_steps"] = 0
    cfg.setdefault("loss", {})
    cfg["loss"]["lambda_ang"] = float(lambda_ang)
    cfg["loss"]["use_angular_loss"] = bool(lambda_ang != 0.0)
    cfg.setdefault("loader", {})
    cfg["loader"]["n_epochs"] = int(n_epochs)
    cfg["loader"]["batch_size"] = 8
    cfg["loader"]["n_workers"] = 8
    cfg["loader"]["persistent_workers"] = True
    cfg["loader"]["prefetch_factor"] = 4
    cfg["loader"]["pin_memory"] = False
    cfg.setdefault("test", {})
    cfg["test"]["sample"] = 10
    cfg["test"]["n_valid"] = 1
    path = CONFIG_DIR / f"{tag}.yml"
    write_yaml(path, cfg)
    return path


def run_logged(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return process.returncode


def parse_neuron_summary(log_path: Path) -> dict[str, float]:
    pattern = re.compile(r"ATLIFTernaryPSN summary: (\{.*\})")
    last: dict[str, Any] | None = None
    if not log_path.exists():
        return {}
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                try:
                    last = ast.literal_eval(match.group(1))
                except (SyntaxError, ValueError):
                    continue
    if not last:
        return {}
    return {key: float(value) for key, value in last.items() if isinstance(value, (int, float))}


def load_profile(profile_dir: Path) -> dict[str, Any]:
    summary_path = profile_dir / "sops_summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def train_case(tag: str, config: Path, run_dir: Path, prev_runid: Path = BASELINE_CKPT) -> dict[str, Any]:
    log_path = run_dir / "train.log"
    save_pattern = run_dir / "checkpoint_epoch{}.pth"
    code = run_logged(
        [
            sys.executable,
            str(H9_TRAIN),
            "--config",
            str(config),
            "--prev_runid",
            str(prev_runid),
            "--save_path",
            str(save_pattern),
        ],
        log_path,
    )
    ckpt = run_dir / "checkpoint_epoch0.pth"
    return {
        "tag": tag,
        "returncode": code,
        "log": str(log_path),
        "checkpoint": str(ckpt),
        "neuron": parse_neuron_summary(log_path),
    }


def profile_case(tag: str, config: Path, checkpoint: Path, output_dir: Path, samples: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "profile.log"
    code = run_logged(
        [
            sys.executable,
            str(H9_PROFILE),
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--overlay",
            str(H9_OVERLAY),
            "--output-dir",
            str(output_dir),
            "--num-samples",
            str(samples),
            "--batch-size",
            "1",
            "--num-workers",
            "4",
            "--metric",
            "AEE",
            "--metric",
            "AAE",
        ],
        log_path,
    )
    profile = load_profile(output_dir)
    profile["returncode"] = code
    profile["log"] = str(log_path)
    profile["tag"] = tag
    return profile


def score_case(row: dict[str, Any]) -> float:
    metrics = row.get("metrics", {})
    aee = 99.0 if metrics.get("AEE") is None else float(metrics.get("AEE"))
    aae = 99.0 if metrics.get("AAE") is None else float(metrics.get("AAE"))
    firing = 1.0 if row.get("global_firing_rate") is None else float(row.get("global_firing_rate"))
    sops_g = 99.0 if row.get("estimated_total_sops") is None else float(row.get("estimated_total_sops")) / 1.0e9
    neg = 0.0 if row.get("neg_mean") is None else float(row.get("neg_mean"))
    pos = 0.0 if row.get("pos_mean") is None else float(row.get("pos_mean"))
    # This project is aiming for a sparsity story, so dense "accuracy-only"
    # short runs should not win the full-run slot.
    neg_bonus = min(neg, 0.01) * 20.0
    imbalance_penalty = max(0.0, (pos / max(neg, 1.0e-6)) - 200.0) * 0.0005
    dense_penalty = max(0.0, sops_g - 4.5) * 0.8 + max(0.0, firing - 0.11) * 10.0
    return aee + 0.04 * aae + 0.35 * sops_g + 4.0 * firing + dense_penalty + imbalance_penalty - neg_bonus


def flatten_result(
    *,
    tag: str,
    phase: str,
    negative_scale: float,
    lambda_ang: float,
    train: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    neuron = train.get("neuron", {})
    metrics = profile.get("metrics", {})
    row = {
        "tag": tag,
        "phase": phase,
        "negative_scale": negative_scale,
        "lambda_ang": lambda_ang,
        "train_returncode": train.get("returncode"),
        "profile_returncode": profile.get("returncode"),
        "checkpoint": train.get("checkpoint"),
        "train_log": train.get("log"),
        "profile_log": profile.get("log"),
        "AEE": metrics.get("AEE"),
        "AAE": metrics.get("AAE"),
        "sops_g": None
        if profile.get("estimated_total_sops") is None
        else float(profile["estimated_total_sops"]) / 1.0e9,
        "firing": profile.get("global_firing_rate"),
        "threshold_mean": neuron.get("threshold_mean"),
        "activity_mean": neuron.get("activity_mean"),
        "pos_mean": neuron.get("pos_mean"),
        "neg_mean": neuron.get("neg_mean"),
    }
    row["score"] = score_case(
        {
            "metrics": {"AEE": row["AEE"], "AAE": row["AAE"]},
            "estimated_total_sops": 0.0 if row["sops_g"] is None else row["sops_g"] * 1.0e9,
            "global_firing_rate": 1.0 if row["firing"] is None else row["firing"],
            "pos_mean": 0.0 if row["pos_mean"] is None else row["pos_mean"],
            "neg_mean": 0.0 if row["neg_mean"] is None else row["neg_mean"],
        }
    )
    return row


def write_trajectory(rows: list[dict[str, Any]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "trajectory.csv"
    fields = [
        "tag",
        "phase",
        "negative_scale",
        "lambda_ang",
        "score",
        "AEE",
        "AAE",
        "sops_g",
        "firing",
        "threshold_mean",
        "activity_mean",
        "pos_mean",
        "neg_mean",
        "checkpoint",
        "train_log",
        "profile_log",
        "train_returncode",
        "profile_returncode",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def finite_row(row: dict[str, Any]) -> bool:
    return (
        row.get("train_returncode") == 0
        and row.get("profile_returncode") == 0
        and row.get("AEE") is not None
        and row.get("AAE") is not None
        and math.isfinite(float(row["score"]))
    )


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    run_stamp = now_tag()

    phase1: list[tuple[str, float, float]] = [
        ("i16a_neg1_noang_guard120", 1.0, 0.0),
        ("i16b_neg2_noang_guard120", 2.0, 0.0),
        ("i16c_neg4_noang_guard120", 4.0, 0.0),
        ("i16d_neg8_noang_guard120", 8.0, 0.0),
    ]
    for tag, neg_scale, lambda_ang in phase1:
        cfg = make_config(
            tag=tag,
            negative_scale=neg_scale,
            lambda_ang=lambda_ang,
            max_train_steps=120,
            n_epochs=1,
        )
        run_dir = RESULTS_DIR / f"{tag}_{run_stamp}"
        train = train_case(tag, cfg, run_dir)
        profile = profile_case(tag, cfg, Path(train["checkpoint"]), run_dir / "profile_valid10", samples=10)
        rows.append(
            flatten_result(
                tag=tag,
                phase="phase1_neg_scale",
                negative_scale=neg_scale,
                lambda_ang=lambda_ang,
                train=train,
                profile=profile,
            )
        )
        write_trajectory(rows)

    viable = [row for row in rows if finite_row(row)]
    viable.sort(key=lambda item: float(item["score"]))
    promoted = viable[:2] if len(viable) >= 2 else viable

    for base_row in promoted:
        neg_scale = float(base_row["negative_scale"])
        tag = f"i16e_neg{neg_scale:g}_ang01_guard120"
        cfg = make_config(
            tag=tag,
            negative_scale=neg_scale,
            lambda_ang=0.1,
            max_train_steps=120,
            n_epochs=1,
        )
        run_dir = RESULTS_DIR / f"{tag}_{run_stamp}"
        train = train_case(tag, cfg, run_dir)
        profile = profile_case(tag, cfg, Path(train["checkpoint"]), run_dir / "profile_valid10", samples=10)
        rows.append(
            flatten_result(
                tag=tag,
                phase="phase2_angular",
                negative_scale=neg_scale,
                lambda_ang=0.1,
                train=train,
                profile=profile,
            )
        )
        write_trajectory(rows)

    viable = [row for row in rows if finite_row(row)]
    viable.sort(key=lambda item: float(item["score"]))
    if not viable:
        raise RuntimeError("No viable I16 short-run candidate completed.")

    best = viable[0]
    full_tag = f"i16_full_neg{float(best['negative_scale']):g}_ang{float(best['lambda_ang']):g}_{run_stamp}"
    full_cfg = make_config(
        tag=full_tag,
        negative_scale=float(best["negative_scale"]),
        lambda_ang=float(best["lambda_ang"]),
        max_train_steps=0,
        n_epochs=30,
    )
    full_dir = RESULTS_DIR / full_tag
    full_train = train_case(full_tag, full_cfg, full_dir)
    full_ckpt = full_dir / "checkpoint_epoch29.pth"
    if not full_ckpt.exists():
        full_ckpt = Path(full_train["checkpoint"])
    full_profile = profile_case(full_tag, full_cfg, full_ckpt, full_dir / "profile_epoch29_valid40", samples=40)
    rows.append(
        flatten_result(
            tag=full_tag,
            phase="full_selected",
            negative_scale=float(best["negative_scale"]),
            lambda_ang=float(best["lambda_ang"]),
            train=full_train,
            profile=full_profile,
        )
    )
    write_trajectory(rows)

    summary = {
        "selected_short_candidate": best,
        "full_tag": full_tag,
        "full_config": str(full_cfg),
        "full_dir": str(full_dir),
        "full_profile": full_profile,
        "trajectory_csv": str(RESULTS_DIR / "trajectory.csv"),
        "updated_at": now_tag(),
    }
    with (RESULTS_DIR / "autopilot_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
