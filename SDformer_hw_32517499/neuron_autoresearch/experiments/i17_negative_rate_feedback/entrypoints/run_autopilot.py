"""I17 autonomous sweep for keeping ATLIF ternary negative spikes alive."""

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


def make_config(case: dict[str, Any], *, max_train_steps: int, n_epochs: int) -> Path:
    cfg = deepcopy(load_base_config())
    cfg["experiment"] = case["tag"]
    cfg.setdefault("runtime", {})
    cfg["runtime"]["max_train_steps"] = int(max_train_steps)
    cfg["runtime"]["skip_state_save"] = True
    cfg["runtime"]["skip_save"] = False
    cfg["runtime"]["use_mlflow_model_logging"] = False
    cfg.setdefault("atlif_ternary_psn", {})
    cfg["atlif_ternary_psn"]["negative_threshold_scale"] = float(case["negative_scale"])
    cfg["atlif_ternary_psn"]["negative_target_rate"] = float(case["negative_target_rate"])
    cfg["atlif_ternary_psn"]["negative_target_eta"] = float(case["negative_target_eta"])
    cfg["atlif_ternary_psn"]["negative_scale_min"] = float(case["negative_scale_min"])
    cfg["atlif_ternary_psn"]["negative_scale_max"] = float(case["negative_scale_max"])
    cfg["atlif_ternary_psn"]["log_interval_steps"] = 0
    cfg.setdefault("loss", {})
    cfg["loss"]["lambda_ang"] = float(case.get("lambda_ang", 0.0))
    cfg["loss"]["use_angular_loss"] = bool(float(case.get("lambda_ang", 0.0)) != 0.0)
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
    path = CONFIG_DIR / f"{case['tag']}.yml"
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
    path = profile_dir / "sops_summary.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def train_case(case: dict[str, Any], config: Path, run_dir: Path, *, n_epochs: int) -> dict[str, Any]:
    log_path = run_dir / "train.log"
    save_pattern = run_dir / "checkpoint_epoch{}.pth"
    code = run_logged(
        [
            sys.executable,
            str(H9_TRAIN),
            "--config",
            str(config),
            "--prev_runid",
            str(BASELINE_CKPT),
            "--save_path",
            str(save_pattern),
        ],
        log_path,
    )
    last_ckpt = run_dir / f"checkpoint_epoch{int(n_epochs) - 1}.pth"
    if not last_ckpt.exists():
        last_ckpt = run_dir / "checkpoint_epoch0.pth"
    return {
        "tag": case["tag"],
        "returncode": code,
        "log": str(log_path),
        "checkpoint": str(last_ckpt),
        "neuron": parse_neuron_summary(log_path),
    }


def profile_case(case: dict[str, Any], config: Path, checkpoint: Path, output_dir: Path, *, samples: int) -> dict[str, Any]:
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
    profile["tag"] = case["tag"]
    return profile


def score_row(row: dict[str, Any]) -> float:
    aee = 99.0 if row.get("AEE") is None else float(row["AEE"])
    aae = 99.0 if row.get("AAE") is None else float(row["AAE"])
    sops_g = 99.0 if row.get("sops_g") is None else float(row["sops_g"])
    firing = 1.0 if row.get("firing") is None else float(row["firing"])
    neg = 0.0 if row.get("neg_mean") is None else float(row["neg_mean"])
    neg_low_penalty = max(0.0, 0.0025 - neg) * 120.0
    neg_dense_penalty = max(0.0, neg - 0.04) * 35.0
    sparse_penalty = max(0.0, sops_g - 3.6) * 0.7 + max(0.0, firing - 0.09) * 7.0
    bad_accuracy_penalty = max(0.0, aee - 2.0) * 2.5 + max(0.0, aae - 10.0) * 0.08
    return (
        aee
        + 0.04 * aae
        + 0.35 * sops_g
        + 3.0 * firing
        + neg_low_penalty
        + neg_dense_penalty
        + sparse_penalty
        + bad_accuracy_penalty
    )


def flatten(case: dict[str, Any], phase: str, train: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    neuron = train.get("neuron", {})
    metrics = profile.get("metrics", {})
    row = {
        "tag": case["tag"],
        "phase": phase,
        "negative_scale": case["negative_scale"],
        "negative_target_rate": case["negative_target_rate"],
        "negative_target_eta": case["negative_target_eta"],
        "negative_scale_min": case["negative_scale_min"],
        "negative_scale_max": case["negative_scale_max"],
        "lambda_ang": case.get("lambda_ang", 0.0),
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
        "negative_scale_mean": neuron.get("negative_scale_mean"),
        "negative_scale_min_observed": neuron.get("negative_scale_min"),
        "negative_scale_max_observed": neuron.get("negative_scale_max"),
        "checkpoint": train.get("checkpoint"),
        "train_log": train.get("log"),
        "profile_log": profile.get("log"),
        "train_returncode": train.get("returncode"),
        "profile_returncode": profile.get("returncode"),
    }
    row["score"] = score_row(row)
    return row


def write_trajectory(rows: list[dict[str, Any]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "tag",
        "phase",
        "score",
        "negative_scale",
        "negative_target_rate",
        "negative_target_eta",
        "negative_scale_min",
        "negative_scale_max",
        "lambda_ang",
        "AEE",
        "AAE",
        "sops_g",
        "firing",
        "threshold_mean",
        "activity_mean",
        "pos_mean",
        "neg_mean",
        "negative_scale_mean",
        "negative_scale_min_observed",
        "negative_scale_max_observed",
        "checkpoint",
        "train_log",
        "profile_log",
        "train_returncode",
        "profile_returncode",
    ]
    with (RESULTS_DIR / "trajectory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def viable(row: dict[str, Any]) -> bool:
    return (
        row.get("train_returncode") == 0
        and row.get("profile_returncode") == 0
        and row.get("AEE") is not None
        and row.get("AAE") is not None
        and math.isfinite(float(row["score"]))
    )


def run_case(
    case: dict[str, Any],
    phase: str,
    run_stamp: str,
    *,
    max_train_steps: int,
    n_epochs: int,
    profile_samples: int,
) -> dict[str, Any]:
    cfg = make_config(case, max_train_steps=max_train_steps, n_epochs=n_epochs)
    run_dir = RESULTS_DIR / f"{case['tag']}_{run_stamp}"
    train = train_case(case, cfg, run_dir, n_epochs=n_epochs)
    profile = profile_case(case, cfg, Path(train["checkpoint"]), run_dir / f"profile_valid{profile_samples}", samples=profile_samples)
    return flatten(case, phase, train, profile)


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    run_stamp = now_tag()

    guards = [
        {
            "tag": "i17a_t003_eta5_scale30_guard180",
            "negative_scale": 30.0,
            "negative_target_rate": 0.003,
            "negative_target_eta": 5.0,
            "negative_scale_min": 8.0,
            "negative_scale_max": 60.0,
            "lambda_ang": 0.0,
        },
        {
            "tag": "i17b_t005_eta5_scale30_guard180",
            "negative_scale": 30.0,
            "negative_target_rate": 0.005,
            "negative_target_eta": 5.0,
            "negative_scale_min": 8.0,
            "negative_scale_max": 60.0,
            "lambda_ang": 0.0,
        },
        {
            "tag": "i17c_t003_eta10_scale30_guard180",
            "negative_scale": 30.0,
            "negative_target_rate": 0.003,
            "negative_target_eta": 10.0,
            "negative_scale_min": 8.0,
            "negative_scale_max": 60.0,
            "lambda_ang": 0.0,
        },
        {
            "tag": "i17d_t005_eta10_scale30_guard180",
            "negative_scale": 30.0,
            "negative_target_rate": 0.005,
            "negative_target_eta": 10.0,
            "negative_scale_min": 8.0,
            "negative_scale_max": 60.0,
            "lambda_ang": 0.0,
        },
        {
            "tag": "i17e_t005_eta5_scale30_ang01_guard180",
            "negative_scale": 30.0,
            "negative_target_rate": 0.005,
            "negative_target_eta": 5.0,
            "negative_scale_min": 8.0,
            "negative_scale_max": 60.0,
            "lambda_ang": 0.1,
        },
    ]

    for case in guards:
        rows.append(run_case(case, "phase1_guard", run_stamp, max_train_steps=180, n_epochs=1, profile_samples=10))
        write_trajectory(rows)

    candidates = [row for row in rows if viable(row)]
    candidates.sort(key=lambda item: float(item["score"]))
    promoted_rows = candidates[:2] if len(candidates) >= 2 else candidates
    medium_rows: list[dict[str, Any]] = []
    for row in promoted_rows:
        case = {
            "tag": row["tag"].replace("_guard180", "_medium3"),
            "negative_scale": float(row["negative_scale"]),
            "negative_target_rate": float(row["negative_target_rate"]),
            "negative_target_eta": float(row["negative_target_eta"]),
            "negative_scale_min": float(row["negative_scale_min"]),
            "negative_scale_max": float(row["negative_scale_max"]),
            "lambda_ang": float(row["lambda_ang"]),
        }
        medium = run_case(case, "phase2_medium3", run_stamp, max_train_steps=0, n_epochs=3, profile_samples=10)
        rows.append(medium)
        medium_rows.append(medium)
        write_trajectory(rows)

    final_candidates = [row for row in medium_rows if viable(row) and float(row.get("neg_mean") or 0.0) >= 0.002]
    if not final_candidates:
        final_candidates = [row for row in medium_rows if viable(row)]
    if not final_candidates:
        raise RuntimeError("No viable I17 medium candidate completed.")
    final_candidates.sort(key=lambda item: float(item["score"]))
    best = final_candidates[0]
    full_case = {
        "tag": f"i17_full_{best['tag']}_{run_stamp}",
        "negative_scale": float(best["negative_scale"]),
        "negative_target_rate": float(best["negative_target_rate"]),
        "negative_target_eta": float(best["negative_target_eta"]),
        "negative_scale_min": float(best["negative_scale_min"]),
        "negative_scale_max": float(best["negative_scale_max"]),
        "lambda_ang": float(best["lambda_ang"]),
    }
    full = run_case(full_case, "phase3_full", run_stamp, max_train_steps=0, n_epochs=30, profile_samples=40)
    rows.append(full)
    write_trajectory(rows)

    summary = {
        "selected_medium_candidate": best,
        "full_result": full,
        "trajectory_csv": str(RESULTS_DIR / "trajectory.csv"),
        "updated_at": now_tag(),
    }
    with (RESULTS_DIR / "autopilot_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
