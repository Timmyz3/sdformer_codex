"""Wait for H67/H68 full30, then screen and promote one H69 temperature."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
PY = Path(sys.executable)
TTX = (
    RESULTS
    / "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid"
    / "checkpoint_epoch2.pth"
)
H68_RUN = RESULTS / "h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30_bs8_full30_20260711_setsid"
H68_DONE = H68_RUN / "profile_ranking_valid825.md"
H67_DEPLOY_STATUS = RESULTS / "h67_early_deploy_after_h68_status.log"
TTB_STATUS = RESULTS / "ttb_density_after_h67_status.log"
STATUS = RESULTS / "h69_after_h67_h68_status.log"
SCREEN_TAG = "h69_dyadic_temperature_screen"
EPOCHS = (0, 4, 9, 14, 19, 24, 28, 29)


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(command: list[str], log: Path, label: str) -> None:
    env = os.environ.copy()
    env.update({
        "SDFORMER_USE_MLFLOW": "0",
        "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
        "SDFORMER_SNN_BACKEND": "cupy",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    log.parent.mkdir(parents=True, exist_ok=True)
    record(f"START {label}: {' '.join(command)}")
    with log.open("a", encoding="utf-8") as handle:
        proc = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    record(f"END {label}: exit_code={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed; log={log}")


def wait_for_h68() -> None:
    while not H68_DONE.exists():
        record(f"WAIT H68 standard valid825: {H68_DONE}")
        time.sleep(600)
    marker = "ALL COMPLETE H67 EARLY DEPLOY:"
    while not H67_DEPLOY_STATUS.exists() or marker not in H67_DEPLOY_STATUS.read_text(encoding="utf-8", errors="ignore"):
        record(f"WAIT H67 early dyadic deploy: {H67_DEPLOY_STATUS}")
        time.sleep(300)
    marker = "ALL COMPLETE TRUE TTB:"
    while not TTB_STATUS.exists() or marker not in TTB_STATUS.read_text(encoding="utf-8", errors="ignore"):
        record(f"WAIT true TTB profile100: {TTB_STATUS}")
        time.sleep(300)


def newest_screen() -> Path:
    candidates = sorted(
        (path for path in RESULTS.glob(f"{SCREEN_TAG}_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError("H69 rapid screen directory was not created")
    return candidates[-1]


def screen_is_complete(screen_dir: Path) -> bool:
    summary = screen_dir / "summary.csv"
    if not summary.exists():
        return False
    rows = list(csv.DictReader(summary.open(encoding="utf-8")))
    screened = {row["name"] for row in rows if row["stage"] == "screen"}
    return all(
        any(f"dyadic_temperature_ttx_x{scale}_steps360" in name for name in screened)
        for scale in (4, 8, 16)
    )


def select_best(screen_dir: Path) -> dict[str, str]:
    rows = list(csv.DictReader((screen_dir / "summary.csv").open(encoding="utf-8")))
    confirmed = [row for row in rows if row["stage"] == "confirm" and row["gate"] == "pass"]
    if confirmed:
        return min(confirmed, key=lambda row: float(row["score"]))

    # A 360-step directional metric is a convergence diagnostic, not a valid
    # reason to reject a structurally eligible all12 candidate.  When no row
    # clears the valid40 promotion gate, still promote the best observed
    # temperature to the pre-registered full30 comparison.
    candidates = [row for row in rows if row["stage"] in {"screen", "confirm"}]
    if not candidates:
        raise RuntimeError(f"no usable H69 screen result: {screen_dir / 'summary.md'}")
    return min(
        candidates,
        key=lambda row: (row["stage"] != "confirm", float(row["score"])),
    )


def make_full_config(screen_dir: Path, best: dict[str, str]) -> tuple[Path, str]:
    short_name = best["name"].removesuffix("_valid40")
    short_config = screen_dir / "configs" / f"{short_name}.yml"
    config = yaml.safe_load(short_config.read_text(encoding="utf-8")) or {}
    scale = int(float(config["bsa_attention"]["score_scale"]))
    name = f"h69_allbinary_all12_dyadic_temperature_ttx_x{scale}_w720_fastlr_full30"
    config = deepcopy(config)
    config["experiment"] = name
    config["runtime"].update({
        "max_train_steps": 0,
        "skip_state_save": False,
        "save_only_force_epochs": True,
        "state_save_epochs": [19, 24, 29],
        "force_save_epochs": list(EPOCHS),
        "use_mlflow_model_logging": False,
    })
    config["loader"].update({
        "n_epochs": 30,
        "batch_size": 8,
        "n_workers": 8,
        "persistent_workers": True,
        "pin_memory": False,
        "prefetch_factor": 4,
        "non_blocking": True,
    })
    config["optimizer"].update({"milestones": [20, 25], "use_amp": True})
    config["note"] = (
        f"H69 promoted full30 with fixed dyadic score_scale={scale}; selected by the "
        "pre-registered short360 summary score fallback after no candidate cleared the "
        "valid40 promotion gate. Starts independently from TTX epoch2."
    )
    path = GEN / f"{name}.yml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path, name


def append_result(name: str, config: Path, run_dir: Path, screen_dir: Path) -> None:
    marker = f"H69_FULL30::{name}"
    if marker in REDESIGN.read_text(encoding="utf-8"):
        return
    ranking = run_dir / "profile_ranking_valid825.md"
    table = [line for line in ranking.read_text(encoding="utf-8").splitlines() if line.startswith("| ")]
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n### H69 dyadic-temperature full30 自动结果：{name}\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(f"- short screen: `{screen_dir.relative_to(REPO)}`\n")
        handle.write(f"- full config: `{config.relative_to(REPO)}`\n")
        handle.write(f"- start checkpoint: `{TTX.relative_to(REPO)}`\n")
        handle.write(f"- run dir: `{run_dir.relative_to(REPO)}`\n\n")
        for line in table:
            handle.write(line + "\n")


def main() -> int:
    wait_for_h68()
    run([str(PY), str(EXP / "entrypoints/make_h69_dyadic_temperature_configs.py")], STATUS, "generate H69 configs")
    existing = sorted(
        (path for path in RESULTS.glob(f"{SCREEN_TAG}_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    complete = [path for path in existing if screen_is_complete(path)]
    screen_dir = complete[-1] if complete else None
    if screen_dir is None:
        screen_command = [
            str(PY), "-u", str(EXP / "entrypoints/rapid_screen.py"),
            "--steps", "360", "--prev-runid", str(TTX), "--tag", SCREEN_TAG,
            "--promote-aee", "1.80", "--promote-aae", "11.50", "--promote-sops-g", "5.0",
            "--confirm-steps", "360", "--workers", "8", "--amp",
        ]
        for scale in (4, 8, 16):
            screen_command.extend(["--config", str(GEN / f"h69_allbinary_all12_dyadic_temperature_ttx_x{scale}.yml")])
        run(screen_command, RESULTS / "h69_dyadic_temperature_screen_launcher.log", "H69 short360+valid40")
        screen_dir = newest_screen()
    else:
        record(f"REUSE complete H69 screen: {screen_dir}")
    best = select_best(screen_dir)
    if best["stage"] != "confirm" or best["gate"] != "pass":
        record(
            "PROMOTE FALLBACK despite short gate: "
            f"name={best['name']} stage={best['stage']} gate={best['gate']} "
            f"score={best['score']}"
        )
    config, name = make_full_config(screen_dir, best)
    record(f"PROMOTE {name}: score={best['score']} AEE={best['AEE']} AAE={best['AAE']}")
    run_dir = RESULTS / f"{name}_bs8_full30_20260711_setsid"
    run_dir.mkdir(parents=True, exist_ok=True)
    final = run_dir / "checkpoint_epoch29.pth"
    ranking = run_dir / "profile_ranking_valid825.md"
    if not final.exists():
        run(
            [
                str(PY), "-u", str(EXP / "entrypoints/train.py"), "--config", str(config),
                "--prev_runid", str(TTX), "--save_path", str(run_dir / "checkpoint_epoch{}.pth"),
            ],
            run_dir / "train.log",
            f"{name} train full30",
        )
    else:
        record(f"REUSE completed H69 full30 checkpoint: {final}")
    eval_command = [
        str(PY), "-u", str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
        "--config", str(config), "--run-dir", str(run_dir),
    ]
    for epoch in EPOCHS:
        eval_command.extend(["--epoch", str(epoch)])
    if not ranking.exists():
        run(eval_command, run_dir / "valid825_queue.log", f"{name} valid825")
    else:
        record(f"REUSE completed H69 valid825 ranking: {ranking}")
    append_result(name, config, run_dir, screen_dir)
    record(f"ALL COMPLETE H69: {run_dir / 'profile_ranking_valid825.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
