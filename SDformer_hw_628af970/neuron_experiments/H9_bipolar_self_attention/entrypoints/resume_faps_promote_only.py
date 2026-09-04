"""Resume FAPS from valid40 summary: promote → full30 → standard eval."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = EXP_ROOT / "results"
BASELINE = REPO_ROOT / "experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
REDESIGN_MD = REPO_ROOT / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
SHORT_DIR = RESULTS_DIR / "faps_short_20260607_023751"
TAG = "faps_short"

PROMOTE_AEE = 3.0
PROMOTE_AAE = 40.0
PROMOTE_SOPS_G = 6.0


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
        handle.write(f"\n[resume-faps-promote] exit_code={proc.returncode}\n")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}; log={log_path}")
    return int(proc.returncode)


def gate_reason(row: dict[str, Any]) -> str:
    if float(row["AEE"]) > PROMOTE_AEE:
        return f"AEE>{PROMOTE_AEE}"
    if float(row["AAE"]) > PROMOTE_AAE:
        return f"AAE>{PROMOTE_AAE}"
    if float(row["SOPs_G"]) > PROMOTE_SOPS_G:
        return f"SOPs>{PROMOTE_SOPS_G}G"
    return "pass"


def refresh_summary_gates() -> None:
    csv_path = SHORT_DIR / "summary.csv"
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if str(row.get("name", "")).endswith("_valid40"):
            row["stage"] = "confirm"
            row["gate"] = gate_reason(row)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_promote_log(log_path: Path) -> tuple[Path, Path]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    config_match = re.findall(r"full config: (.+)", text)
    run_match = re.findall(r"full run dir: (.+)", text)
    if not config_match or not run_match:
        raise RuntimeError(f"could not parse full config/run dir from {log_path}")
    return Path(config_match[-1].strip()), Path(run_match[-1].strip())


def standard_eval(config: Path, run_dir: Path, epochs: list[int], status: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in epochs:
        checkpoint = run_dir / f"checkpoint_epoch{epoch}.pth"
        if not checkpoint.exists():
            append_status(status, f"standard eval skip missing {checkpoint.name}")
            continue
        out_dir = run_dir / "standard_valid825" / f"epoch{epoch}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / "spike_profile.json").exists():
            (out_dir / "spike_profile.json").unlink()
        append_status(status, f"standard eval epoch{epoch} start")
        run(
            [
                sys.executable,
                "-u",
                "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--path_results",
                str(out_dir),
                "--mode",
                "valid",
            ],
            out_dir / "eval.log",
        )
        data = json.loads((out_dir / "spike_profile.json").read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        row = {
            "epoch": epoch,
            "AEE": float(metrics.get("AEE", "nan")),
            "AAE": float(metrics.get("AAE", "nan")),
            "spikes_g": float(data.get("total_spikes", 0.0) or 0.0) / 1e9,
            "firing": float(data.get("global_firing_rate", 0.0) or 0.0),
            "energy_uj": float(data.get("energy_uj", 0.0) or 0.0),
        }
        rows.append(row)
        append_status(status, f"standard eval epoch{epoch} AEE={row['AEE']:.4f} AAE={row['AAE']:.4f}")
    return rows


def append_md(run_dir: Path, full_config: Path, rows: list[dict[str, Any]], valid40_best: dict[str, Any]) -> None:
    with REDESIGN_MD.open("a", encoding="utf-8") as handle:
        handle.write("\n\n## 三十六、FAPS 短测 → 全量标准 valid825\n\n")
        handle.write(f"- 时间：`{datetime.now().isoformat(timespec='seconds')}`\n")
        handle.write(f"- 短测目录：`{SHORT_DIR}`\n")
        handle.write(
            f"- 短测最优 valid40：`{valid40_best['name']}` "
            f"AEE={float(valid40_best['AEE']):.4f} AAE={float(valid40_best['AAE']):.4f}\n"
        )
        handle.write(f"- 全量配置：`{full_config}`\n")
        handle.write(f"- 全量目录：`{run_dir}`\n\n")
        handle.write("| epoch | AEE | AAE | total_spikes | firing | energy_uj |\n")
        handle.write("|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['epoch']} | {row['AEE']:.4f} | {row['AAE']:.4f} | "
                f"{row['spikes_g']:.4f}G | {row['firing'] * 100:.4f}% | {row['energy_uj']:.2f} |\n"
            )


def main() -> int:
    driver_dir = RESULTS_DIR / f"faps_promote_{stamp()}"
    driver_dir.mkdir(parents=True, exist_ok=True)
    status = driver_dir / "status.log"
    refresh_summary_gates()
    append_status(status, "summary gates refreshed with AEE<=3.0")

    promote_log = driver_dir / "promote_full.log"
    run(
        [
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
        ],
        driver_dir / "promote_driver.log",
    )
    full_config, run_dir = parse_promote_log(promote_log)
    append_status(status, f"full30 done run_dir={run_dir}")

    with (SHORT_DIR / "summary.csv").open("r", encoding="utf-8") as handle:
        valid40 = [r for r in csv.DictReader(handle) if str(r.get("name", "")).endswith("_valid40")]
    best = min(valid40, key=lambda r: float(r["AEE"]))

    rows = standard_eval(full_config, run_dir, [19, 24, 27, 29], status)
    append_md(run_dir, full_config, rows, best)
    append_status(status, "FAPS promote+eval complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())