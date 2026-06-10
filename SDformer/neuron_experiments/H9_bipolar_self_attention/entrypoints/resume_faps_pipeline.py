"""Resume FAPS pipeline after short test: valid40 confirm → full30 → standard eval."""

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

import yaml


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
        handle.write(f"\n[resume-faps] exit_code={proc.returncode}\n")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}; log={log_path}")
    return int(proc.returncode)


def gate_reason(row: dict[str, Any]) -> str:
    if row["AEE"] > PROMOTE_AEE:
        return f"AEE>{PROMOTE_AEE}"
    if row["AAE"] > PROMOTE_AAE:
        return f"AAE>{PROMOTE_AAE}"
    if row["SOPs_G"] > PROMOTE_SOPS_G:
        return f"SOPs>{PROMOTE_SOPS_G}G"
    return "pass"


def profile_valid40(name: str, config_path: Path, checkpoint: Path, out_dir: Path) -> dict[str, Any] | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
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
            "40",
            "--batch-size",
            "1",
            "--num-workers",
            "4",
            "--metric",
            "AEE",
            "--metric",
            "AAE",
        ],
        out_dir / "profile.log",
    )
    summary_path = out_dir / "sops_summary.json"
    if not summary_path.exists():
        return None
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    return {
        "name": f"{name}_valid40",
        "stage": "confirm",
        "gate": "",
        "steps": 360,
        "samples": 40,
        "AEE": float(metrics.get("AEE", "nan")),
        "AAE": float(metrics.get("AAE", "nan")),
        "PE1": float(metrics.get("AEE_PE1", "nan")),
        "PE2": float(metrics.get("AEE_PE2", "nan")),
        "PE3": float(metrics.get("AEE_PE3", metrics.get("AEE_outliers", "nan"))),
        "SOPs_G": float(data.get("estimated_total_sops", 0.0) or 0.0) / 1e9,
        "firing": float(data.get("global_firing_rate", 0.0) or 0.0),
        "summary": str(summary_path),
        "root": str(SHORT_DIR),
    }


def update_summary(valid40_rows: list[dict[str, Any]]) -> None:
    for row in valid40_rows:
        row["gate"] = gate_reason(row)
    valid40_rows.sort(key=lambda item: item["AEE"] + 0.025 * item["AAE"])

    csv_path = SHORT_DIR / "summary.csv"
    existing: list[dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))

    merged = {str(row["name"]): row for row in existing}
    for row in valid40_rows:
        merged[str(row["name"])] = row
    all_rows = list(merged.values())
    all_rows.sort(
        key=lambda item: (
            0 if item.get("stage") == "confirm" and item.get("gate") == "pass" else 1,
            float(item.get("AEE", 999)),
        )
    )

    fieldnames = list(existing[0].keys()) if existing else list(valid40_rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    md_path = SHORT_DIR / "summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# FAPS 快速短训筛选汇总（补跑 valid40）\n\n")
        handle.write("| rank | name | stage | gate | AEE | AAE | SOPs(G) | firing |\n")
        handle.write("|---:|---|---|---|---:|---:|---:|---:|\n")
        for rank, row in enumerate(all_rows, 1):
            handle.write(
                f"| {rank} | {row['name']} | {row.get('stage','')} | {row.get('gate','')} | "
                f"{float(row['AEE']):.4f} | {float(row['AAE']):.4f} | "
                f"{float(row['SOPs_G']):.4f} | {float(row['firing']):.5f} |\n"
            )


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
        stale = out_dir / "spike_profile.json"
        if stale.exists():
            stale.unlink()
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
        append_status(status, f"standard eval epoch{epoch} done AEE={row['AEE']:.4f} AAE={row['AAE']:.4f}")
    return rows


def append_md(run_dir: Path, full_config: Path, rows: list[dict[str, Any]]) -> None:
    with REDESIGN_MD.open("a", encoding="utf-8") as handle:
        handle.write("\n\n## 三十六、FAPS 短测 → 全量标准 valid825（resume 追加）\n\n")
        handle.write(f"- 时间：`{datetime.now().isoformat(timespec='seconds')}`\n")
        handle.write(f"- 短测目录：`{SHORT_DIR}`\n")
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
    driver_dir = RESULTS_DIR / f"faps_resume_{stamp()}"
    driver_dir.mkdir(parents=True, exist_ok=True)
    status = driver_dir / "status.log"
    append_status(status, f"resume from {SHORT_DIR}")

    valid40_rows: list[dict[str, Any]] = []
    runs_dir = SHORT_DIR / "runs"
    for run_path in sorted(runs_dir.iterdir()):
        if not run_path.is_dir():
            continue
        name = run_path.name
        config_path = SHORT_DIR / "configs" / f"{name}.yml"
        checkpoint = run_path / "checkpoint_epoch0.pth"
        if not config_path.exists() or not checkpoint.exists():
            append_status(status, f"skip missing assets for {name}")
            continue
        out_dir = SHORT_DIR / "profiles" / f"{name}_valid40"
        append_status(status, f"valid40 profile start {name}")
        row = profile_valid40(name, config_path, checkpoint, out_dir)
        if row is None:
            append_status(status, f"valid40 profile failed {name}")
            continue
        valid40_rows.append(row)
        append_status(status, f"valid40 done {name} AEE={row['AEE']:.4f} AAE={row['AAE']:.4f} gate={gate_reason(row)}")

    if not valid40_rows:
        raise RuntimeError("no valid40 profiles produced")

    update_summary(valid40_rows)
    append_status(status, f"summary updated with {len(valid40_rows)} valid40 rows")

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
    append_status(status, f"full30 done config={full_config} run_dir={run_dir}")

    rows = standard_eval(full_config, run_dir, [19, 24, 27, 29], status)
    append_md(run_dir, full_config, rows)
    append_status(status, "FAPS resume pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())