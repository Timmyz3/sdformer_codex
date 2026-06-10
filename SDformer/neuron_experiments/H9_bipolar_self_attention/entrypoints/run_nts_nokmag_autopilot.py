"""NTS-00 autopilot: pure TX+SC (no K_mag) short-test → full30 → standard valid825."""

from __future__ import annotations

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
TAG = "nts_nokmag_short"

CONFIGS = [
    "generated/nts00a_mu005_std_s360.yml",
    "generated/nts00b_mu010_std_s360.yml",
    "generated/nts00c_mu015_std_s360.yml",
    "generated/nts00d_mu005_fast_s360.yml",
    "generated/nts00e_mu010_fast_s360.yml",
    "generated/nts00f_mu005_slow_s360.yml",
    "generated/nts00g_mu010_bs6_s360.yml",
    "generated/nts00h_mu005_sched_s360.yml",
]


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
        handle.write(f"\n[nts-nokmag-autopilot] exit_code={proc.returncode}\n")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}; log={log_path}")
    return int(proc.returncode)


def latest_short_dir() -> Path:
    dirs = sorted(RESULTS_DIR.glob(f"{TAG}_20*"), key=lambda item: item.stat().st_mtime)
    if not dirs:
        raise FileNotFoundError(f"no rapid screen dir for tag={TAG}")
    return dirs[-1]


def parse_promote_log(log_path: Path) -> tuple[Path, Path]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    config_match = re.findall(r"full config: (.+)", text)
    run_match = re.findall(r"full run dir: (.+)", text)
    if not config_match or not run_match:
        raise RuntimeError(f"could not parse full config/run dir from {log_path}")
    return Path(config_match[-1].strip()), Path(run_match[-1].strip())


def metric_float(metrics: dict[str, Any], key: str) -> float:
    return float(metrics.get(key, "nan"))


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
        summary = out_dir / "spike_profile.json"
        if not summary.exists():
            append_status(status, f"standard eval epoch{epoch} missing spike_profile.json")
            continue
        data = json.loads(summary.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        dense = float(data.get("dense_flops", 0.0) or 0.0)
        effective = float(data.get("effective_flops", 0.0) or 0.0)
        row = {
            "epoch": epoch,
            "AEE": metric_float(metrics, "AEE"),
            "AAE": metric_float(metrics, "AAE"),
            "PE1": metric_float(metrics, "AEE_PE1"),
            "PE2": metric_float(metrics, "AEE_PE2"),
            "outlier": metric_float(metrics, "AEE_outliers"),
            "spikes_g": float(data.get("total_spikes", 0.0) or 0.0) / 1e9,
            "firing": float(data.get("global_firing_rate", 0.0) or 0.0),
            "effective_g": effective / 1e9,
            "sparsity": 1.0 - effective / dense if dense else 0.0,
            "energy_uj": float(data.get("energy_uj", 0.0) or 0.0),
        }
        rows.append(row)
        append_status(status, f"standard eval epoch{epoch} done AEE={row['AEE']:.4f} AAE={row['AAE']:.4f}")
    return rows


def append_md(short_dir: Path, promote_log: Path, full_config: Path, run_dir: Path, rows: list[dict[str, Any]]) -> None:
    with REDESIGN_MD.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### NTS-00 纯 TX+SC（无 K_mag）自动短测与全量结果（自动追加）\n\n")
        handle.write(f"- 时间：`{datetime.now().isoformat(timespec='seconds')}`\n")
        handle.write(f"- 短测目录：`{short_dir}`\n")
        handle.write(f"- promotion log：`{promote_log}`\n")
        handle.write(f"- 全量配置：`{full_config}`\n")
        handle.write(f"- 全量目录：`{run_dir}`\n")
        handle.write("- 方法：NTS pure TX+SC score fusion，`k_magnitude_alpha=0`\n")
        handle.write("- 标准推理：`eval_DSEC_flow_SNN.py`，full valid825，CuPy backend。\n\n")
        handle.write("| epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |\n")
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['epoch']} | {row['AEE']:.4f} | {row['AAE']:.4f} | "
                f"{row['PE1']:.4f} | {row['PE2']:.4f} | {row['outlier']:.4f} | "
                f"{row['spikes_g']:.4f}G | {row['firing'] * 100:.4f}% | {row['energy_uj']:.2f} |\n"
            )
        if rows:
            best = min(rows, key=lambda item: item["AEE"])
            handle.write(
                f"\n当前自动判断：精度最佳 epoch{best['epoch']}，AEE `{best['AEE']:.4f}`、AAE `{best['AAE']:.4f}`。\n"
            )


def main() -> int:
    driver_dir = RESULTS_DIR / f"nts_nokmag_autopilot_{stamp()}"
    driver_dir.mkdir(parents=True, exist_ok=True)
    status = driver_dir / "status.log"
    append_status(status, f"driver_dir={driver_dir}")
    append_status(status, f"baseline={BASELINE}")

    run([sys.executable, str(EXP_ROOT / "entrypoints/make_nts_nokmag_configs.py")], driver_dir / "make_configs.log")
    run([sys.executable, "-m", "py_compile", str(EXP_ROOT / "entrypoints/make_nts_nokmag_configs.py")], driver_dir / "py_compile_make.log")
    run([sys.executable, "-m", "py_compile", str(EXP_ROOT / "entrypoints/run_nts_nokmag_autopilot.py")], driver_dir / "py_compile_autopilot.log")

    rapid_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/rapid_screen.py"),
        "--steps",
        "360",
        "--prev-runid",
        str(BASELINE),
        "--batch-size",
        "8",
        "--workers",
        "8",
        "--prefetch-factor",
        "4",
        "--pin-memory",
        "--amp",
        "--valid-samples",
        "10",
        "--promote-samples",
        "40",
        "--promote-aee",
        "3.00",
        "--promote-aae",
        "40.0",
        "--promote-sops-g",
        "6.0",
        "--max-zero-neg-modules",
        "20",
        "--max-worst-pos-neg-ratio",
        "100000000",
        "--tag",
        TAG,
    ]
    for config in CONFIGS:
        rapid_cmd.extend(["--config", config])
    append_status(status, "rapid screen start")
    run(rapid_cmd, driver_dir / "rapid_screen.log")
    short_dir = latest_short_dir()
    append_status(status, f"rapid screen done short_dir={short_dir}")

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

    rows = standard_eval(full_config, run_dir, [19, 24, 28, 29], status)
    append_md(short_dir, promote_log, full_config, run_dir, rows)
    append_status(status, "NTS-00 nokmag autopilot complete; md appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())