"""Run H50/H51/H52 short screens, then launch the best H50 full run."""

from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATED_DIR = EXP_ROOT / "configs" / "generated"
RESULTS_DIR = EXP_ROOT / "results"
BASELINE_CKPT = REPO_ROOT / "experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def append_status(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] {text.rstrip()}\n")


def run_logged(command: list[str], log_path: Path, status_path: Path, label: str) -> int:
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    append_status(status_path, f"{label} start: {' '.join(command)}")
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        code = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        log.write(f"\n[{label}] exit_code={code}\n")
    append_status(status_path, f"{label} exit={code}")
    return int(code)


def newest_screen_dir() -> Path:
    dirs = sorted(RESULTS_DIR.glob("h50_h51_h52_short_*"), key=lambda path: path.stat().st_mtime)
    if not dirs:
        raise FileNotFoundError("No h50_h51_h52_short_* result directory found")
    return dirs[-1]


def finite_float(value: Any, default: float = math.inf) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def choose_h50_config(screen_dir: Path) -> Path:
    summary = screen_dir / "summary.csv"
    if not summary.exists():
        return GENERATED_DIR / "h50a_h49_layered_precision.yml"
    rows: list[dict[str, Any]] = []
    with summary.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row.get("name", "")
            if not name.startswith("h50"):
                continue
            if "_valid40" not in name and row.get("stage") != "confirm":
                continue
            aee = finite_float(row.get("AEE"))
            aae = finite_float(row.get("AAE"))
            sops = finite_float(row.get("SOPs_G"))
            firing = finite_float(row.get("firing"))
            if not all(math.isfinite(item) for item in (aee, aae, sops, firing)):
                continue
            row["_pick_score"] = aee + 0.035 * aae + 0.25 * max(0.0, sops - 3.20)
            rows.append(row)
    if not rows:
        return GENERATED_DIR / "h50a_h49_layered_precision.yml"
    rows.sort(key=lambda item: finite_float(item.get("_pick_score")))
    stem = rows[0]["name"].split("_steps", 1)[0]
    return GENERATED_DIR / f"{stem}.yml"


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def make_full_config(selected: Path, run_dir: Path) -> Path:
    cfg = read_yaml(selected)
    cfg["experiment"] = f"{selected.stem}_full30_{run_dir.name.rsplit('_', 1)[-1]}"
    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 30
    loader["batch_size"] = 8
    loader["n_workers"] = 8
    loader["pin_memory"] = True
    loader["persistent_workers"] = True
    loader["prefetch_factor"] = 4
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = list(range(30))
    runtime["use_mlflow_model_logging"] = False
    cfg.setdefault("optimizer", {})["use_amp"] = True
    note = cfg.get("note", "")
    cfg["note"] = f"{note}\nH50 autopilot full run selected after H50/H51/H52 short screening."
    out = GENERATED_DIR / f"{cfg['experiment']}.yml"
    write_yaml(out, cfg)
    return out


def launch_full(selected: Path, status: Path) -> int:
    run_dir = RESULTS_DIR / f"{selected.stem}_full30_{stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = make_full_config(selected, run_dir)
    (run_dir / "selected_base_config.txt").write_text(str(selected) + "\n", encoding="utf-8")
    (run_dir / "full_config.txt").write_text(str(config) + "\n", encoding="utf-8")
    append_status(status, f"H50 full selected_config={selected}")
    append_status(status, f"H50 full run_dir={run_dir}")

    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    train_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/train.py"),
        "--config",
        str(config),
        "--prev_runid",
        str(BASELINE_CKPT),
        "--save_path",
        str(run_dir / "checkpoint_epoch{}.pth"),
    ]
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(train_cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(train_cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        (run_dir / "run.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
        append_status(status, f"H50 full train pid={proc.pid}")
        code = proc.wait()
        log.write(f"\n[h50_full_train] exit_code={code}\n")
    append_status(status, f"H50 full train exit={code}")
    if code != 0:
        return int(code)

    profile_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/profile_checkpoints.py"),
        "--config",
        str(config),
        "--run-dir",
        str(run_dir),
        "--output-root",
        str(run_dir / "profiles"),
        "--samples",
        "40",
        "--epoch",
        "26",
        "--epoch",
        "29",
    ]
    return run_logged(profile_cmd, run_dir / "profile.log", status, "h50_full_profile")


def main() -> int:
    driver_dir = RESULTS_DIR / f"h50_h51_h52_autopilot_{stamp()}"
    driver_dir.mkdir(parents=True, exist_ok=True)
    status = driver_dir / "status.log"
    append_status(status, "autopilot start")

    make_code = run_logged(
        [sys.executable, "-u", str(EXP_ROOT / "entrypoints/make_h50_h51_h52_configs.py")],
        driver_dir / "make_configs.log",
        status,
        "make_configs",
    )
    if make_code != 0:
        return make_code

    configs = [
        "generated/h50a_h49_layered_precision.yml",
        "generated/h50b_h49_layered_balanced.yml",
        "generated/h50c_h49_layered_sparse.yml",
        "generated/h51a_dual_channel_balanced.yml",
        "generated/h51b_dual_channel_precision.yml",
        "generated/h52a_kasv_a2os2a_shiftmax.yml",
    ]
    rapid_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/rapid_screen.py"),
        "--steps",
        "360",
        "--batch-size",
        "8",
        "--workers",
        "8",
        "--pin-memory",
        "--amp",
        "--valid-samples",
        "10",
        "--promote-samples",
        "40",
        "--promote-aee",
        "2.30",
        "--promote-aae",
        "13.00",
        "--promote-sops-g",
        "4.20",
        "--max-zero-neg-modules",
        "999",
        "--max-worst-pos-neg-ratio",
        "999",
        "--parallel",
        "2",
        "--tag",
        "h50_h51_h52_short",
    ]
    for config in configs:
        rapid_cmd.extend(["--config", config])
    rapid_code = run_logged(rapid_cmd, driver_dir / "rapid_screen.log", status, "rapid_screen")
    if rapid_code != 0:
        append_status(status, "rapid_screen returned nonzero; falling back to H50a full")
    screen_dir = newest_screen_dir()
    selected = choose_h50_config(screen_dir)
    append_status(status, f"short_screen_dir={screen_dir}")
    append_status(status, f"selected_h50={selected}")
    return launch_full(selected, status)


if __name__ == "__main__":
    raise SystemExit(main())
