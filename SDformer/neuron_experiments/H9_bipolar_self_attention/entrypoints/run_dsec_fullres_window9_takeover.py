"""Take over the DSEC fullres window9 series:

1) Wait for a safe H67 force-save checkpoint (prefer ep9+).
2) Stop the old Codex train/queue processes.
3) Run NB0 formal multi-ckpt valid825 first.
4) Resume H67 FT from the latest saved checkpoint with --resume.
5) Formal-eval H67, then train+eval H66d.

Does not modify crop experiments; only orchestrates fullres runs.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
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
STATUS = RESULTS / "dsec_fullres_window9_takeover_status.log"
RUN_TAG = "20260726"
FORMAL_EPOCHS = [0, 4, 9, 14, 19, 24, 28, 29]

NB0 = {
    "id": "NB0",
    "config": GEN / "dsec_fullres_w9_nb0_ep59_ft30.yml",
    "run_dir": RESULTS / f"dsec_fullres_w9_nb0_ep59_ft30_bs2_{RUN_TAG}",
}
H67 = {
    "id": "H67",
    "config": GEN / "dsec_fullres_w9_h67_motion_ep19_ft30.yml",
    "run_dir": RESULTS / f"dsec_fullres_w9_h67_motion_ep19_ft30_bs2_{RUN_TAG}",
    "crop_start": EXP
    / (
        "results/h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_"
        "bs8_full30_20260711_setsid/checkpoint_epoch19.pth"
    ),
}
H66D = {
    "id": "H66d",
    "config": GEN / "dsec_fullres_w9_h66d_local5_ep29_ft30.yml",
    "run_dir": RESULTS / f"dsec_fullres_w9_h66d_local5_ep29_ft30_bs2_{RUN_TAG}",
    "crop_start": EXP
    / (
        "results/h66d_allbinary_all12_lr_ttx_w720_fastlr_full30_"
        "bs8_full30_20260712_setsid/checkpoint_epoch29.pth"
    ),
}


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "SDFORMER_USE_MLFLOW": "0",
            "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
            "SDFORMER_SNN_BACKEND": "cupy",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return env


def run(command: list[str], log: Path, label: str) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    record(f"START {label}: {' '.join(command)}")
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(command)}\n")
        handle.flush()
        proc = subprocess.run(
            command,
            cwd=REPO,
            env=environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    record(f"END {label}: exit_code={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed; log={log}")


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(float(out.strip().splitlines()[0].strip()))
    except Exception:
        return None


def list_matching_pids(patterns: list[str]) -> list[int]:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid=,cmd="], text=True)
    except Exception:
        return []
    pids: list[int] = []
    self_pid = os.getpid()
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == self_pid:
            continue
        cmd = parts[1]
        # avoid matching this takeover script itself
        if "run_dsec_fullres_window9_takeover.py" in cmd:
            continue
        if any(pat in cmd for pat in patterns):
            pids.append(pid)
    return sorted(set(pids))


def stop_old_fullres_jobs() -> None:
    patterns = [
        "run_dsec_fullres_window9_queue.py",
        "run_dsec_fullres_window9_formal_eval.py",
        "dsec_fullres_w9_h67_motion",
        "dsec_fullres_w9_h66d_local5",
        "dsec_fullres_w9_nb0_ep59",
    ]
    pids = list_matching_pids(patterns)
    if not pids:
        record("no old fullres train/queue/eval pids to stop")
        return
    record(f"SIGTERM old fullres pids={pids}")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 120
    while time.time() < deadline:
        alive = [pid for pid in pids if Path(f"/proc/{pid}").exists()]
        if not alive:
            break
        time.sleep(2)
    alive = [pid for pid in pids if Path(f"/proc/{pid}").exists()]
    if alive:
        record(f"SIGKILL remaining pids={alive}")
        for pid in alive:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(3)
    # wait GPU drain
    for _ in range(60):
        used = gpu_used_mib()
        busy = list_matching_pids(
            [
                "run_dsec_fullres_window9_queue.py",
                "dsec_fullres_w9_",
                "run_dsec_fullres_window9_formal_eval.py",
            ]
        )
        if not busy and used is not None and used < 4096:
            record(f"GPU drained used={used}MiB")
            return
        record(f"WAIT GPU drain used={used}MiB busy={busy}")
        time.sleep(5)
    record(f"WARN GPU not fully drained used={gpu_used_mib()}MiB; continuing")


def latest_force_save(run_dir: Path) -> tuple[int, Path] | None:
    best = None
    for epoch in FORMAL_EPOCHS:
        ckpt = run_dir / f"checkpoint_epoch{epoch}.pth"
        if ckpt.is_file():
            best = (epoch, ckpt)
    return best


def wait_h67_safe_checkpoint(prefer_epoch: int, timeout_hours: float) -> tuple[int, Path]:
    run_dir = H67["run_dir"]
    deadline = time.time() + timeout_hours * 3600.0
    prefer = run_dir / f"checkpoint_epoch{prefer_epoch}.pth"
    while time.time() < deadline:
        if prefer.is_file():
            # wait a moment for companion state_dict if written right after
            state = run_dir / f"checkpoint_epoch{prefer_epoch}_state_dict.pth"
            for _ in range(30):
                if state.is_file() or not prefer.is_file():
                    break
                time.sleep(2)
            record(f"H67 safe checkpoint ready: {prefer}")
            return prefer_epoch, prefer
        cur = latest_force_save(run_dir)
        record(
            f"WAIT H67 checkpoint_epoch{prefer_epoch}.pth "
            f"(latest_force_save={cur[0] if cur else None})"
        )
        time.sleep(30)
    cur = latest_force_save(run_dir)
    if cur is None:
        raise TimeoutError(f"no H67 force-save checkpoint under {run_dir}")
    record(f"TIMEOUT waiting ep{prefer_epoch}; falling back to epoch{cur[0]}")
    return cur


def formal_eval(model_id: str) -> None:
    command = [
        str(PY),
        "-u",
        str(EXP / "entrypoints/run_dsec_fullres_window9_formal_eval.py"),
        "--ids",
        model_id,
        "--batch-size",
        "2",
    ]
    for epoch in FORMAL_EPOCHS:
        command.extend(["--epoch", str(epoch)])
    run(
        command,
        RESULTS / f"dsec_fullres_window9_takeover_{model_id.lower()}_eval.log",
        f"{model_id} formal valid825",
    )


def train_or_resume(
    *,
    model_id: str,
    config: Path,
    run_dir: Path,
    prev_runid: Path,
    resume: bool,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    final = run_dir / "checkpoint_epoch29.pth"
    if final.is_file():
        record(f"REUSE completed train {model_id}: {final}")
        return
    command = [
        str(PY),
        "-u",
        str(EXP / "entrypoints/train.py"),
        "--config",
        str(config),
        "--prev_runid",
        str(prev_runid),
        "--save_path",
        str(run_dir / "checkpoint_epoch{}.pth"),
        "--finetune",
        "1",
    ]
    if resume:
        command.extend(["--resume", "True"])
    run(command, run_dir / "train.log", f"{model_id} train/resume fullres")
    if not final.is_file():
        raise RuntimeError(f"{model_id} finished without {final}")


def append_takeover_note(h67_resume_epoch: int) -> None:
    marker = "DSEC_FULLRES_W9_TAKEOVER_20260727"
    text = REDESIGN.read_text(encoding="utf-8") if REDESIGN.is_file() else ""
    if marker in text:
        return
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### DSEC fullres window9 接管编排（2026-07-27）\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(
            "- 用户要求：先完成 NB0 正式推理，再继续 H67/H66d 训练；"
            "本会话接管 Codex 原队列。\n"
        )
        handle.write(
            f"- H67 中断点：等待 force-save 后从 epoch{h67_resume_epoch} "
            f"resume（`--resume True --finetune 1`），避免丢掉 mid-run 权重。\n"
        )
        handle.write(
            "- 顺序：stop old queue → NB0 formal valid825 "
            f"(epochs {FORMAL_EPOCHS}) → H67 resume+eval → H66d train+eval。\n"
        )
        handle.write(f"- status：`{STATUS.relative_to(REPO)}`\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefer-h67-epoch",
        type=int,
        default=9,
        help="Wait for this force-save epoch before stopping H67 (default 9).",
    )
    parser.add_argument("--wait-timeout-hours", type=float, default=3.0)
    parser.add_argument(
        "--skip-wait-h67",
        action="store_true",
        help="Stop immediately using the latest existing H67 force-save.",
    )
    parser.add_argument(
        "--stop-only",
        action="store_true",
        help="Only stop old jobs (debug).",
    )
    args = parser.parse_args()

    record(
        f"TAKEOVER START prefer_h67_epoch={args.prefer_h67_epoch} "
        f"skip_wait={args.skip_wait_h67}"
    )

    if args.skip_wait_h67:
        cur = latest_force_save(H67["run_dir"])
        if cur is None:
            raise RuntimeError("no H67 checkpoint to fall back to")
        h67_epoch, h67_ckpt = cur
    else:
        h67_epoch, h67_ckpt = wait_h67_safe_checkpoint(
            args.prefer_h67_epoch, args.wait_timeout_hours
        )

    append_takeover_note(h67_epoch)
    stop_old_fullres_jobs()
    if args.stop_only:
        record("STOP ONLY done")
        return 0

    # 1) NB0 formal inference first
    if not (NB0["run_dir"] / "checkpoint_epoch29.pth").is_file():
        raise RuntimeError("NB0 fullres FT30 checkpoint_epoch29 missing; cannot formal-eval")
    formal_eval("NB0")

    # 2) Resume H67 if needed
    if not (H67["run_dir"] / "checkpoint_epoch29.pth").is_file():
        # Re-resolve latest after stop (may have advanced if epoch completed)
        cur = latest_force_save(H67["run_dir"])
        if cur is None:
            raise RuntimeError("H67 has no force-save after stop")
        h67_epoch, h67_ckpt = cur
        record(f"H67 resume from epoch{h67_epoch}: {h67_ckpt}")
        train_or_resume(
            model_id="H67",
            config=H67["config"],
            run_dir=H67["run_dir"],
            prev_runid=h67_ckpt,
            resume=True,
        )
    else:
        record("H67 already complete; skip train")
    formal_eval("H67")

    # 3) H66d train from crop winner + formal eval
    train_or_resume(
        model_id="H66d",
        config=H66D["config"],
        run_dir=H66D["run_dir"],
        prev_runid=H66D["crop_start"],
        resume=False,
    )
    formal_eval("H66d")

    record("ALL COMPLETE DSEC FULLRES WINDOW9 TAKEOVER")
    # marker for waiters
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            "ALL COMPLETE DSEC FULLRES WINDOW9 TAKEOVER\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
