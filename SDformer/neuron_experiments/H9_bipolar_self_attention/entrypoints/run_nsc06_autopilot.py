"""NSC-06 autopilot: TX+SC hybrid short screen, then launch one full30."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = EXP_ROOT / "results"
CONFIG_DIR = EXP_ROOT / "configs"
BASELINE = REPO_ROOT / "experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
REDESIGN_MD = REPO_ROOT / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
TAG = "nsc06_tx_sc_hybrid_short"

CONFIGS = [
    "generated/nsc06a_h57_tx_control_all_mu0_steps360.yml",
    "generated/nsc06b_h57_all_mu010_l03_steps360.yml",
    "generated/nsc06c_h57_all_mu020_l03_steps360.yml",
    "generated/nsc06d_h57_s2_mu020_l03_steps360.yml",
    "generated/nsc06e_h57_s02_mu015_l03_steps360.yml",
    "generated/nsc06f_h57_s012_mu015_l04_steps360.yml",
    "generated/nsc06g_h57_s2_conf_mu020_l03_steps360.yml",
    "generated/nsc06h_h56r_s2_alpha025_l03_steps360.yml",
    "generated/nsc06i_h57_s23_mu010_l03_steps360.yml",
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
        handle.write(f"\n[nsc06-autopilot] exit_code={proc.returncode}\n")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}; log={log_path}")
    return int(proc.returncode)


def latest_short_dir() -> Path:
    dirs = sorted(RESULTS_DIR.glob(f"{TAG}_20*"), key=lambda item: item.stat().st_mtime)
    if not dirs:
        raise FileNotFoundError(f"no rapid screen dir for tag={TAG}")
    return dirs[-1]


def _float(row: dict[str, str], key: str, default: float = float("inf")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def load_rows(short_dir: Path) -> list[dict[str, Any]]:
    summary = short_dir / "summary.csv"
    rows: list[dict[str, Any]] = []
    with summary.open("r", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for key in ("AEE", "AAE", "SOPs_G", "firing", "score"):
                row[key] = _float(raw, key)
            row["samples"] = int(_float(raw, "samples", 0.0))
            rows.append(row)
    return rows


def variant_from_row_name(name: str) -> str:
    if name.endswith("_valid40"):
        name = name[: -len("_valid40")]
    while name.endswith("_steps360"):
        name = name[: -len("_steps360")]
    return name


def select_for_full(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    pool = [
        row
        for row in rows
        if row.get("stage") == "confirm" and row.get("gate") == "pass" and row.get("samples", 0) >= 40
    ] or rows
    pool = sorted(pool, key=lambda row: (row.get("gate") != "pass", row["score"], row["AEE"]))
    best = pool[0]
    non_control = [row for row in pool if "tx_control" not in str(row.get("name", ""))]
    if non_control:
        candidate = non_control[0]
        close_enough = (
            candidate["AEE"] <= best["AEE"] + 0.12
            and candidate["AAE"] <= best["AAE"] + 1.20
            and candidate["SOPs_G"] <= best["SOPs_G"] + 0.35
            and candidate["AEE"] <= 1.95
        )
        if "tx_control" in str(best.get("name", "")) and close_enough:
            return candidate, "selected best non-control close to TX control"
    return best, "selected best ranked row"


def launch_full30(variant: str, driver_dir: Path, status: Path) -> tuple[Path, Path, int]:
    full_config = CONFIG_DIR / "generated" / f"{variant}_full30.yml"
    if not full_config.exists():
        raise FileNotFoundError(full_config)
    run_dir = RESULTS_DIR / f"{variant}_auto_full_bs8_{stamp()}_setsid"
    run_dir.mkdir(parents=True, exist_ok=True)
    train_log = run_dir / "train.log"
    command = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/train.py"),
        "--config",
        str(full_config),
        "--prev_runid",
        str(BASELINE),
        "--save_path",
        str(run_dir / "checkpoint_epoch{}.pth"),
    ]
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["SDFORMER_SNN_BACKEND"] = "cupy"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with train_log.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (run_dir / "train.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
    append_status(status, f"launched full30 variant={variant} pid={proc.pid} run_dir={run_dir}")
    (driver_dir / "selected_full30.txt").write_text(
        f"variant={variant}\nconfig={full_config}\nrun_dir={run_dir}\npid={proc.pid}\n",
        encoding="utf-8",
    )
    return full_config, run_dir, int(proc.pid)


def append_md(short_dir: Path, rows: list[dict[str, Any]], selected: dict[str, Any], reason: str, full_config: Path, run_dir: Path, pid: int) -> None:
    top = [
        row
        for row in sorted(rows, key=lambda item: (item.get("stage") != "confirm", item.get("gate") != "pass", item["score"]))
        if row.get("stage") == "confirm"
    ][:6]
    with REDESIGN_MD.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### 31.15 NSC-06 TX+SC hybrid attention 短测与 full30 启动（自动追加）\n\n")
        handle.write(f"- 时间：`{datetime.now().isoformat(timespec='seconds')}`\n")
        handle.write(f"- 短测目录：`{short_dir}`\n")
        handle.write("- 新增 mode：`tx_sc_residual_selector_shiftmax/h57`，TX carrier/gate 为主，SC agree/disagree 小比例 residual。\n")
        handle.write(f"- 选择理由：`{reason}`\n")
        handle.write(f"- 选中短测：`{selected['name']}`，AEE `{selected['AEE']:.4f}`，AAE `{selected['AAE']:.4f}`，SOPs `{selected['SOPs_G']:.4f}G`\n")
        handle.write(f"- full30 配置：`{full_config}`\n")
        handle.write(f"- full30 目录：`{run_dir}`\n")
        handle.write(f"- full30 PID：`{pid}`\n\n")
        handle.write("| rank | variant | stage | AEE | AAE | SOPs | firing | score |\n")
        handle.write("|---:|---|---|---:|---:|---:|---:|---:|\n")
        for rank, row in enumerate(top, 1):
            handle.write(
                f"| {rank} | `{row['name']}` | {row.get('stage', '')} | "
                f"{row['AEE']:.4f} | {row['AAE']:.4f} | {row['SOPs_G']:.4f}G | "
                f"{row['firing'] * 100:.3f}% | {row['score']:.4f} |\n"
            )
        handle.write(
            "\n后续标准化口径：full30 完成后优先对 epoch `19/24/28/29` 使用 "
            "`eval_DSEC_flow_SNN.py` 跑 full valid825，并检查 `checkpoint_overlay_keys`、"
            "`missing`、`unexpected` 与安装模块数。\n"
        )


def main() -> int:
    driver_dir = RESULTS_DIR / f"nsc06_autopilot_{stamp()}"
    driver_dir.mkdir(parents=True, exist_ok=True)
    status = driver_dir / "status.log"
    append_status(status, f"driver_dir={driver_dir}")
    append_status(status, f"baseline={BASELINE}")

    run([sys.executable, str(EXP_ROOT / "entrypoints/make_nsc06_tx_sc_hybrid_configs.py")], driver_dir / "make_configs.log")
    run([sys.executable, "-m", "py_compile", str(EXP_ROOT / "entrypoints/make_nsc06_tx_sc_hybrid_configs.py")], driver_dir / "py_compile_make.log")
    run([sys.executable, "-m", "py_compile", str(EXP_ROOT / "entrypoints/run_nsc06_autopilot.py")], driver_dir / "py_compile_autopilot.log")

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
        "2.05",
        "--promote-aae",
        "30.0",
        "--promote-sops-g",
        "4.4",
        "--max-zero-neg-modules",
        "40",
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
    rows = load_rows(short_dir)
    selected, reason = select_for_full(rows)
    variant = variant_from_row_name(str(selected["name"]))
    append_status(status, f"selected {selected['name']} -> {variant}; {reason}")
    full_config, run_dir, pid = launch_full30(variant, driver_dir, status)
    append_md(short_dir, rows, selected, reason, full_config, run_dir, pid)
    append_status(status, "NSC-06 autopilot complete; full30 launched and md appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
