"""Resume NTS-10 after short test:补 valid40 → promote full30 → standard valid825."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = EXP_ROOT / "results"
BASELINE = REPO_ROOT / "experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
REDESIGN_MD = REPO_ROOT / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
TAG = "nts10_blocks"

H9A_AEE = 1.5044
H9A_AAE = 7.6365
H9A_SOPS_G = 3.0847
BASELINE_FIRING = 0.084961


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def append_status(path: Path, message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(command: list[str], log_path: Path, *, check: bool = True) -> int:
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["SDFORMER_SNN_BACKEND"] = "cupy"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        proc = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[nts10-resume] exit_code={proc.returncode}\n")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}; log={log_path}")
    return int(proc.returncode)


def latest_short_dir() -> Path:
    dirs = sorted(RESULTS_DIR.glob(f"{TAG}_20*"), key=lambda item: item.stat().st_mtime)
    if not dirs:
        raise FileNotFoundError(f"no rapid screen dir for tag={TAG}")
    return dirs[-1]


def candidate_score(aee: float, aae: float, sops_g: float, firing: float) -> float:
    score = aee + 0.025 * aae
    score += 0.30 * max(0.0, sops_g - H9A_SOPS_G)
    score += 0.90 * max(0.0, aee - 1.58)
    score += 0.08 * max(0.0, aae - 7.90)
    score += 0.25 * max(0.0, sops_g - 3.50)
    score += 1.2 * max(0.0, firing - BASELINE_FIRING)
    if sops_g <= 3.25:
        score -= 0.05
    if aee <= H9A_AEE and aae <= H9A_AAE:
        score -= 0.04
    return score


def gate_reason(row: dict[str, Any], promote_aee: float, promote_aae: float, promote_sops_g: float) -> str:
    if row["AEE"] > promote_aee:
        return f"AEE>{promote_aee}"
    if row["AAE"] > promote_aae:
        return f"AAE>{promote_aae}"
    if row["SOPs_G"] > promote_sops_g:
        return f"SOPs>{promote_sops_g}G"
    return "pass"


def profile_checkpoint(config_path: Path, checkpoint: Path, out_dir: Path, samples: int) -> dict[str, Any] | None:
    command = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/profile_sops.py"),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(out_dir),
        "--split",
        "valid",
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
    ]
    start = time.time()
    exit_code = run(command, out_dir / "profile.log")
    summary_path = out_dir / "sops_summary.json"
    if exit_code != 0 or not summary_path.exists():
        return None
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    aee = float(metrics.get("AEE", math.inf))
    aae = float(metrics.get("AAE", math.inf))
    sops_g = float(data.get("estimated_total_sops", math.inf)) / 1e9
    firing = float(data.get("global_firing_rate", math.inf))
    row: dict[str, Any] = {
        "name": f"{config_path.stem}_valid{samples}",
        "steps": 1224,
        "samples": samples,
        "AEE": aee,
        "AAE": aae,
        "PE1": float(metrics.get("AEE_PE1", math.inf)),
        "PE2": float(metrics.get("AEE_PE2", math.inf)),
        "PE3": float(metrics.get("AEE_PE3", metrics.get("AEE_outliers", math.inf))),
        "SOPs_G": sops_g,
        "firing": firing,
        "profile_seconds": time.time() - start,
        "summary": str(summary_path),
        "stage": "confirm" if samples >= 40 else "screen",
    }
    row["gate"] = gate_reason(row, 1.75, 16.0, 6.0)
    row["score"] = candidate_score(aee, aae, sops_g, firing)
    return row


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_tables(rows: list[dict[str, Any]], out_root: Path) -> None:
    rows = sorted(
        rows,
        key=lambda item: (
            item.get("stage") != "confirm",
            item.get("gate") != "pass",
            float(item.get("score", math.inf)),
        ),
    )
    fields = [
        "name", "stage", "gate", "steps", "samples", "AEE", "AAE", "PE1", "PE2", "PE3",
        "SOPs_G", "firing", "score", "train_seconds", "profile_seconds", "summary",
    ]
    csv_path = out_root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    md_path = out_root / "summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# 快速短训筛选汇总\n\n")
        handle.write("补跑 valid40 后更新。\n\n")
        handle.write("| rank | name | stage | gate | samples | AEE | AAE | SOPs(G) | score |\n")
        handle.write("|---:|---|---|---|---:|---:|---:|---:|---:|\n")
        for rank, row in enumerate(rows, 1):
            handle.write(
                f"| {rank} | {row['name']} | {row['stage']} | {row['gate']} | {row['samples']} | "
                f"{float(row['AEE']):.4f} | {float(row['AAE']):.4f} | {float(row['SOPs_G']):.4f} | "
                f"{float(row['score']):.4f} |\n"
            )


def parse_promote_log(log_path: Path) -> tuple[Path, Path]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    config_match = re.findall(r"full config: (.+)", text)
    run_match = re.findall(r"full run dir: (.+)", text)
    if not config_match or not run_match:
        raise RuntimeError(f"could not parse full config/run dir from {log_path}")
    return Path(config_match[-1].strip()), Path(run_match[-1].strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-dir", type=Path, default=None)
    args = parser.parse_args()

    short_dir = args.short_dir or latest_short_dir()
    driver_dir = RESULTS_DIR / f"nts10_blocks_resume_{stamp()}"
    driver_dir.mkdir(parents=True, exist_ok=True)
    status = driver_dir / "status.log"
    append_status(status, f"short_dir={short_dir}")

    rows = read_csv_rows(short_dir / "summary.csv")
    screen_rows = [row for row in rows if row.get("stage") == "screen"]
    append_status(status, f"screen rows={len(screen_rows)}")

    for row in screen_rows:
        base_name = str(row["name"])
        if base_name.endswith("_valid40"):
            continue
        config_path = short_dir / "configs" / f"{base_name}.yml"
        checkpoint = short_dir / "runs" / base_name / "checkpoint_epoch0.pth"
        out_dir = short_dir / "profiles" / f"{base_name}_valid40"
        if (out_dir / "sops_summary.json").exists():
            append_status(status, f"skip existing valid40 {base_name}")
            continue
        if not config_path.exists() or not checkpoint.exists():
            append_status(status, f"missing assets for {base_name}")
            continue
        append_status(status, f"valid40 start {base_name}")
        promoted = profile_checkpoint(config_path, checkpoint, out_dir, 40)
        if promoted is None:
            append_status(status, f"valid40 failed {base_name}")
            continue
        promoted["train_seconds"] = row.get("train_seconds", "")
        rows.append({key: str(value) if not isinstance(value, (int, float)) else value for key, value in promoted.items()})
        append_status(
            status,
            f"valid40 done {base_name} AEE={promoted['AEE']:.4f} AAE={promoted['AAE']:.4f} gate={promoted['gate']}",
        )
        write_tables(rows, short_dir)

    confirm_pass = [row for row in rows if row.get("stage") == "confirm" and row.get("gate") == "pass"]
    append_status(status, f"confirm pass rows={len(confirm_pass)}")
    if not confirm_pass:
        append_status(status, "no valid40 pass candidate; stop before full30")
        return 1

    promote_log = driver_dir / "promote_full.log"
    promote_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/promote_best_rapid_screen.py"),
        "--tag",
        TAG,
        "--prev-runid",
        str(BASELINE),
        "--batch-size",
        "6",
        "--workers",
        "8",
        "--prefetch-factor",
        "4",
        "--pin-memory",
        "--epochs",
        "30",
        "--profile-samples",
        "40",
        "--profile-epoch",
        "0",
        "--profile-epoch",
        "9",
        "--profile-epoch",
        "19",
        "--profile-epoch",
        "29",
        "--log",
        str(promote_log),
    ]
    append_status(status, "promotion/full training start")
    run(promote_cmd, driver_dir / "promote_driver.log")
    full_config, run_dir = parse_promote_log(promote_log)
    append_status(status, f"full training done config={full_config} run_dir={run_dir}")

    waiter_log = run_dir / "standard_valid825_waiter" / "wait.log"
    waiter_log.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-u",
            str(EXP_ROOT / "entrypoints/wait_full_then_run_standard_valid825.py"),
            "--config",
            str(full_config),
            "--run-dir",
            str(run_dir),
            "--epoch",
            "19",
            "--epoch",
            "24",
            "--epoch",
            "29",
        ],
        waiter_log,
    )
    append_status(status, "standard valid825 done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())