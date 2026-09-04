"""NTS-11ac autopilot: verify -> short screen (1224) -> full30 -> standard valid825."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = EXP_ROOT / "results"
BASELINE = REPO_ROOT / "experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
MAKE_CFG = EXP_ROOT / "entrypoints/make_nts11_secondary_ac_config.py"
VERIFY = EXP_ROOT / "entrypoints/verify_nts11_chain.py"
RAPID = EXP_ROOT / "entrypoints/rapid_screen.py"
TRAIN = EXP_ROOT / "entrypoints/train.py"
STD_EVAL = EXP_ROOT / "entrypoints/run_h9_standard_valid825_eval.py"
CFG = EXP_ROOT / "configs/generated/nts11ac_hw_h60_s23_sn2qbin_fastlr_freeze816_warm720_full30.yml"
TAG = "nts11ac_autopilot"
SHORT_STEPS = 1224


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
    # Do not prepend overlay/entrypoints here; train.py manages import order itself.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        proc = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[nts11ac-autopilot] exit_code={proc.returncode}\n")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): log={log_path}")
    return int(proc.returncode)


def latest_short_dir() -> Path:
    dirs = sorted(RESULTS_DIR.glob(f"{TAG}_20*"), key=lambda item: item.stat().st_mtime)
    if not dirs:
        raise FileNotFoundError(f"no rapid screen dir for tag={TAG}")
    return dirs[-1]


def read_short_metrics(short_dir: Path) -> dict:
    summary = short_dir / "summary.csv"
    if not summary.exists():
        return {}
    lines = summary.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return {}
    header = [part.strip() for part in lines[0].split(",")]
    values = [part.strip() for part in lines[1].split(",")]
    return dict(zip(header, values))


def main() -> int:
    driver_dir = RESULTS_DIR / f"{TAG}_{stamp()}"
    status = driver_dir / "status.log"
    append_status(status, f"driver_dir={driver_dir}")

    if CFG.is_file():
        append_status(status, f"step1: reuse existing config {CFG}")
    else:
        append_status(status, "step1: regenerate nts11ac config")
        run([sys.executable, str(MAKE_CFG)], driver_dir / "make_config.log")

    append_status(status, "step2: verify_nts11_chain")
    run([sys.executable, str(VERIFY), str(CFG)], driver_dir / "verify.log")

    append_status(status, f"step3: rapid_screen short test ({SHORT_STEPS} steps)")
    run(
        [
            sys.executable,
            "-u",
            str(RAPID),
            "--config",
            str(CFG),
            "--steps",
            str(SHORT_STEPS),
            "--prev-runid",
            str(BASELINE),
            "--batch-size",
            "8",
            "--valid-samples",
            "10",
            "--confirm-steps",
            str(SHORT_STEPS),
            "--promote-samples",
            "40",
            "--promote-aee",
            "4.5",
            "--promote-aae",
            "80.0",
            "--promote-sops-g",
            "2.5",
            "--workers",
            "8",
            "--prefetch-factor",
            "4",
            "--tag",
            TAG,
        ],
        driver_dir / "rapid_screen.log",
    )

    short_dir = latest_short_dir()
    short_metrics = read_short_metrics(short_dir)
    append_status(status, f"short_test_done dir={short_dir} metrics={json.dumps(short_metrics, ensure_ascii=False)}")

    run_stamp = stamp()
    run_dir = EXP_ROOT / "results" / f"nts11ac_full30_bs8_{run_stamp}_setsid"
    run_dir.mkdir(parents=True, exist_ok=True)
    (driver_dir / "full_run_dir.txt").write_text(str(run_dir) + "\n", encoding="utf-8")

    append_status(status, f"step4: full30 training -> {run_dir}")
    run(
        [
            sys.executable,
            "-u",
            str(TRAIN),
            "--config",
            str(CFG),
            "--prev_runid",
            str(BASELINE),
            "--save_path",
            str(run_dir / "checkpoint_epoch{}.pth"),
        ],
        run_dir / "train.log",
    )

    append_status(status, "step5: standard valid825 eval (epochs 9,14,19,24,28,29)")
    run(
        [
            sys.executable,
            "-u",
            str(STD_EVAL),
            "--config",
            str(CFG),
            "--run-dir",
            str(run_dir),
            "--epoch",
            "9",
            "--epoch",
            "14",
            "--epoch",
            "19",
            "--epoch",
            "24",
            "--epoch",
            "28",
            "--epoch",
            "29",
        ],
        run_dir / "standard_valid825_eval.log",
        check=False,
    )

    append_status(status, f"DONE run_dir={run_dir} ranking={run_dir / 'profile_ranking_valid825.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())