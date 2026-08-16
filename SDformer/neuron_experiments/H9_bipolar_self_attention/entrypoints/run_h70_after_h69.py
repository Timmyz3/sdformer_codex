"""Wait for H69, then run H70 health check, full30, and valid825."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


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
H69_STATUS = RESULTS / "h69_after_h67_h68_status.log"
STATUS = RESULTS / "h70_after_h69_status.log"
NAME = "h70_allbinary_all12_event_selective_ttx_maxshift3_w720_fastlr_full30"
RUN_DIR = RESULTS / f"{NAME}_bs8_full30_20260711_setsid"
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


def wait_for_h69() -> None:
    while True:
        if H69_STATUS.exists() and "ALL COMPLETE H69:" in H69_STATUS.read_text(encoding="utf-8", errors="ignore"):
            return
        record(f"WAIT H69 full30: {H69_STATUS}")
        time.sleep(600)


def append_result(config: Path) -> None:
    marker = "H70_EVENT_SELECTIVE_FULL30_RESULT"
    if marker in REDESIGN.read_text(encoding="utf-8"):
        return
    ranking = RUN_DIR / "profile_ranking_valid825.md"
    table = [line for line in ranking.read_text(encoding="utf-8").splitlines() if line.startswith("| ")]
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### H70 Event-Selective TTX full30 自动结果\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(f"- config: `{config.relative_to(REPO)}`\n")
        handle.write(f"- start checkpoint: `{TTX.relative_to(REPO)}`\n")
        handle.write(f"- run dir: `{RUN_DIR.relative_to(REPO)}`\n\n")
        for line in table:
            handle.write(line + "\n")


def main() -> int:
    wait_for_h69()
    run([str(PY), str(EXP / "entrypoints/make_h70_event_selective_ttx_configs.py")], STATUS, "generate H70 configs")
    smoke = GEN / "h70_allbinary_all12_event_selective_ttx_maxshift3_s360.yml"
    status_text = STATUS.read_text(encoding="utf-8", errors="ignore") if STATUS.exists() else ""
    if "END H70 implementation health check: exit_code=0" not in status_text:
        run(
            [
                str(PY), "-u", str(EXP / "entrypoints/rapid_screen.py"),
                "--config", str(smoke), "--steps", "360", "--prev-runid", str(TTX),
                "--tag", "h70_event_selective_health", "--promote-aee", "3.0",
                "--promote-aae", "40.0", "--promote-sops-g", "10.0", "--amp",
            ],
            RESULTS / "h70_event_selective_health_launcher.log",
            "H70 implementation health check",
        )
    else:
        record("REUSE completed H70 implementation health check")
    config = GEN / f"{NAME}.yml"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    final = RUN_DIR / "checkpoint_epoch29.pth"
    ranking = RUN_DIR / "profile_ranking_valid825.md"
    if not final.exists():
        run(
            [
                str(PY), "-u", str(EXP / "entrypoints/train.py"), "--config", str(config),
                "--prev_runid", str(TTX), "--save_path", str(RUN_DIR / "checkpoint_epoch{}.pth"),
            ],
            RUN_DIR / "train.log",
            "H70 train full30",
        )
    else:
        record(f"REUSE completed H70 full30 checkpoint: {final}")
    eval_command = [
        str(PY), "-u", str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
        "--config", str(config), "--run-dir", str(RUN_DIR),
    ]
    for epoch in EPOCHS:
        eval_command.extend(["--epoch", str(epoch)])
    if not ranking.exists():
        run(eval_command, RUN_DIR / "valid825_queue.log", "H70 valid825")
    else:
        record(f"REUSE completed H70 valid825 ranking: {ranking}")
    append_result(config)
    record(f"ALL COMPLETE H70: {RUN_DIR / 'profile_ranking_valid825.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
