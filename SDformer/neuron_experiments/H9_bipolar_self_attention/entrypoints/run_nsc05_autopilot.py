"""NSC-05 autopilot: short-screen confidence SC, then full valid825 eval."""

from __future__ import annotations

import csv
import json
import math
import os
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
TAG = "nsc05_sc_conf_blend_short"

CONFIGS = [
    "generated/nsc05a_h56m_mu05_l06_tr03_steps360.yml",
    "generated/nsc05b_conf_mu04_l05_tr03_steps360.yml",
    "generated/nsc05c_conf_mu04_l06_tr03_steps360.yml",
    "generated/nsc05d_conf_mu06_l05_tr03_steps360.yml",
    "generated/nsc05e_conf_mu04_l05_tr02_steps360.yml",
    "generated/nsc05f_conf_mu04_l05_notr_steps360.yml",
    "generated/nsc05g_conf_active_mu04_l05_tr03_steps360.yml",
    "generated/nsc05h_conf_kmod_mu04_l05_tr03_steps360.yml",
    "generated/nsc05i_conf_mu04_l05_tr03_slowlr_steps360.yml",
    "generated/nsc05j_conf_mu04_l05_tr03_clamp_steps360.yml",
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
        handle.write(f"\n[nsc05-autopilot] exit_code={proc.returncode}\n")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}; log={log_path}")
    return int(proc.returncode)


def latest_short_dir() -> Path:
    dirs = sorted(RESULTS_DIR.glob(f"{TAG}_20*"), key=lambda item: item.stat().st_mtime)
    if not dirs:
        raise FileNotFoundError(f"no rapid screen dir for tag={TAG}")
    return dirs[-1]


def _float(row: dict[str, str], key: str, default: float = math.inf) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def select_best(short_dir: Path) -> dict[str, Any]:
    summary = short_dir / "summary.csv"
    if not summary.exists():
        raise FileNotFoundError(summary)
    rows: list[dict[str, Any]] = []
    with summary.open("r", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            row["AEE"] = _float(raw, "AEE")
            row["AAE"] = _float(raw, "AAE")
            row["SOPs_G"] = _float(raw, "SOPs_G")
            row["firing"] = _float(raw, "firing")
            row["score"] = _float(raw, "score")
            row["samples"] = int(_float(raw, "samples", 0.0))
            rows.append(row)
    valid40 = [
        row
        for row in rows
        if row.get("stage") == "confirm" and row.get("gate") == "pass" and row["samples"] >= 40
    ]
    pool = valid40 or rows
    if not pool:
        raise RuntimeError(f"no rows in {summary}")
    return min(
        pool,
        key=lambda row: (
            row.get("stage") != "confirm",
            row.get("gate") != "pass",
            row["score"],
            row["AEE"] + 0.025 * row["AAE"] + 0.30 * max(0.0, row["SOPs_G"] - 3.1),
        ),
    )


def row_paths(short_dir: Path, row: dict[str, Any]) -> tuple[Path, Path, Path]:
    name = str(row["name"])
    if name.endswith("_valid40"):
        name = name[: -len("_valid40")]
    cfg = short_dir / "configs" / f"{name}.yml"
    run_dir = short_dir / "runs" / name
    checkpoints = sorted(run_dir.glob("checkpoint_epoch*.pth"))
    if not cfg.exists() or not checkpoints:
        raise FileNotFoundError(f"missing config/checkpoint for selected row {name}")
    return cfg, run_dir, checkpoints[-1]


def metric_float(metrics: dict[str, Any], key: str) -> float:
    return float(metrics.get(key, "nan"))


def standard_eval(config: Path, checkpoint: Path, out_dir: Path, status: Path) -> dict[str, Any]:
    profile = out_dir / "spike_profile.json"
    if profile.exists():
        profile.unlink()
    append_status(status, f"standard eval selected short checkpoint start: {checkpoint}")
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
    data = json.loads(profile.read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})
    dense = float(data.get("dense_flops", 0.0) or 0.0)
    effective = float(data.get("effective_flops", 0.0) or 0.0)
    return {
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


def append_md(short_dir: Path, selected: dict[str, Any], config: Path, checkpoint: Path, out_dir: Path, result: dict[str, Any]) -> None:
    with REDESIGN_MD.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### 31.14 NSC-05 confidence-aware SC 短测与标准推理（自动追加）\n\n")
        handle.write(f"- 时间：`{datetime.now().isoformat(timespec='seconds')}`\n")
        handle.write(f"- 短测目录：`{short_dir}`\n")
        handle.write(f"- 选中短测：`{selected['name']}`，valid40 AEE `{selected['AEE']:.4f}`，AAE `{selected['AAE']:.4f}`，SOPs `{selected['SOPs_G']:.4f}G`\n")
        handle.write(f"- 配置：`{config}`\n")
        handle.write(f"- checkpoint：`{checkpoint}`\n")
        handle.write(f"- 标准推理目录：`{out_dir}`\n")
        handle.write("- 推理：`eval_DSEC_flow_SNN.py` full valid825，CuPy backend。\n\n")
        handle.write("| AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | effective_flops | sparsity | energy_uj |\n")
        handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        handle.write(
            f"| {result['AEE']:.4f} | {result['AAE']:.4f} | {result['PE1']:.4f} | "
            f"{result['PE2']:.4f} | {result['outlier']:.4f} | {result['spikes_g']:.4f}G | "
            f"{result['firing'] * 100:.4f}% | {result['effective_g']:.4f}G | "
            f"{result['sparsity'] * 100:.2f}% | {result['energy_uj']:.2f} |\n"
        )
        handle.write(
            "\n判断口径：若 full valid825 AEE 不能接近 `1.55` 或 spikes 不能压到 `34G` 附近，"
            "SC/NSC 线不应继续抢主线资源；若短测 full valid825 接近 NTX-04/NTX-01，则再排 full30。\n"
        )


def main() -> int:
    driver_dir = RESULTS_DIR / f"nsc05_autopilot_{stamp()}"
    driver_dir.mkdir(parents=True, exist_ok=True)
    status = driver_dir / "status.log"
    append_status(status, f"driver_dir={driver_dir}")
    append_status(status, f"baseline={BASELINE}")

    run([sys.executable, str(EXP_ROOT / "entrypoints/make_nsc05_sc_conf_blend_configs.py")], driver_dir / "make_configs.log")
    run([sys.executable, "-m", "py_compile", str(EXP_ROOT / "entrypoints/make_nsc05_sc_conf_blend_configs.py")], driver_dir / "py_compile_make.log")
    run([sys.executable, "-m", "py_compile", str(EXP_ROOT / "entrypoints/run_nsc05_autopilot.py")], driver_dir / "py_compile_autopilot.log")

    rapid_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/rapid_screen.py"),
        "--steps",
        "360",
        "--prev-runid",
        str(BASELINE),
        "--batch-size",
        "4",
        "--workers",
        "4",
        "--prefetch-factor",
        "2",
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
    selected = select_best(short_dir)
    config, _, checkpoint = row_paths(short_dir, selected)
    append_status(status, f"selected {selected['name']} config={config} checkpoint={checkpoint}")
    out_dir = driver_dir / "standard_valid825_selected"
    result = standard_eval(config, checkpoint, out_dir, status)
    append_status(status, f"standard eval done AEE={result['AEE']:.4f} AAE={result['AAE']:.4f}")
    append_md(short_dir, selected, config, checkpoint, out_dir, result)
    append_status(status, "NSC-05 autopilot complete; md appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
