#!/usr/bin/env python3
"""Keep the long-running DATE evidence followers alive until closure."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
RESULTS = EXP / "results"
HW = REPO / "hw_autoresearch_nts07"
PYTHON = "/opt/conda/envs/sdformerflow/bin/python"
STATUS = RESULTS / "date_closure_watchdog_20260805.log"
LOCK = RESULTS / "date_closure_watchdog_20260805.lock"
PID_FILE = RESULTS / "date_closure_watchdog_20260805.pid"
POLL_SECONDS = 120
MAX_RESTARTS = 8


@dataclass(frozen=True)
class Task:
    name: str
    command: tuple[str, ...]
    pid_file: Path
    launcher_log: Path
    completion_path: Path
    completion_marker: str | None = None
    completion_json_status: str | None = None
    required_paths: tuple[Path, ...] = ()


LOCAL = RESULTS / "dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805"
TASKS = (
    Task(
        name="local5_ep9_config_identity",
        command=(
            PYTHON,
            "-u",
            str(EXP / "entrypoints/enforce_local5_ep9_config_identity_20260805.py"),
        ),
        pid_file=LOCAL / "training_config_identity.pid",
        launcher_log=LOCAL / "training_config_identity_launcher.log",
        completion_path=LOCAL / "training_config_identity.json",
        completion_json_status="PASS",
    ),
    Task(
        name="local5_checkpoint_bound_rtl",
        command=(
            PYTHON,
            "-u",
            str(HW / "scripts/run_local5_bb1e4_checkpoint_bound_rtl.py"),
        ),
        pid_file=HW / "results/local5_bb1e4_checkpoint_bound_rtl_watcher_20260805.pid",
        launcher_log=HW / "results/local5_bb1e4_checkpoint_bound_rtl_launcher_20260805.log",
        completion_path=HW / "results/local5_bb1e4_checkpoint_bound_rtl_watcher_20260805.log",
        completion_marker="ALL COMPLETE checkpoint-bound Local-5",
        required_paths=(
            HW
            / "results/local5_bb1e4_qgasr2c_fivebank_postg0_rtl_20260805/checkpoint_bound_scope.json",
        ),
    ),
    Task(
        name="h67_nb0_equal_plus10",
        command=(
            PYTHON,
            "-u",
            str(EXP / "entrypoints/run_dsec_fullres_w15_equal_plus10_convergence.py"),
        ),
        pid_file=RESULTS / "dsec_fullres_w15_equal_plus10_convergence_20260805.pid",
        launcher_log=RESULTS / "dsec_fullres_w15_equal_plus10_convergence_launcher_20260805.log",
        completion_path=RESULTS / "dsec_fullres_w15_equal_plus10_convergence_20260805.log",
        # Producer emits Local5/H67/NB0; keep H67/NB0 as alternate substring.
        completion_marker="ALL COMPLETE Local5/H67/NB0 equal +10 convergence audit",
        required_paths=(
            RESULTS / "dsec_fullres_w15_equal_plus10_convergence_summary_20260805.json",
        ),
    ),
    Task(
        name="h67_ep30_component_rtl",
        command=(
            PYTHON,
            "-u",
            str(HW / "scripts/run_h67_ep30_fullres_t450_profile.py"),
        ),
        pid_file=HW / "results/h67_fullres_ep30_t450_profile_watcher_20260805.pid",
        launcher_log=HW / "results/h67_fullres_ep30_t450_profile_watcher_launcher_20260805.log",
        completion_path=HW / "results/h67_fullres_ep30_t450_profile_watcher_20260805.log",
        completion_marker="ALL COMPLETE H67 ep30 fullres T450",
        required_paths=(
            HW
            / "results/h67_fullres_ep30_t450_profile100_20260805/nts11_hardware_p0_profile.json",
            HW / "results/h67_fullres_ep30_t450_all12_bit_trace_20260805/manifest.json",
            HW
            / "results/h67_fullres_ep30_t450_all12_bit_trace_audit_20260805/audit.json",
            HW / "results/h67_fullres_ep30_t450_score_shiftmax_rtl_20260805/report.json",
            HW / "results/h67_ep30_checkpoint_atlif_dptme_rtl_20260805/report.json",
            HW / "results/h67_ep30_checkpoint_projection_rtl_20260805/report.json",
        ),
    ),
    Task(
        name="h67_postconvergence_component_rtl",
        command=(
            PYTHON,
            "-u",
            str(HW / "scripts/run_h67_postconvergence_rank1_profile.py"),
        ),
        pid_file=HW / "results/h67_postconvergence_rank1_profile_watcher_20260805.pid",
        launcher_log=HW / "results/h67_postconvergence_rank1_profile_watcher_launcher_20260805.log",
        completion_path=HW / "results/h67_postconvergence_rank1_profile_watcher_20260805.log",
        completion_marker="ALL COMPLETE H67 post-convergence",
        required_paths=(
            HW / "results/h67_postconvergence_rank1_hardware_evidence_20260805.json",
        ),
    ),
    Task(
        name="date_algorithm_closure",
        command=(
            PYTHON,
            "-u",
            str(EXP / "entrypoints/audit_date_algorithm_closure_20260805.py"),
            "--wait",
            "--poll-seconds",
            "300",
        ),
        pid_file=RESULTS / "date_algorithm_closure_audit_20260805.pid",
        launcher_log=RESULTS / "date_algorithm_closure_audit_launcher_20260805.log",
        completion_path=RESULTS / "date_algorithm_closure_audit_20260805.log",
        completion_marker="ALL COMPLETE DATE algorithm closure audit PASS",
        required_paths=(
            REPO / "neuron_autoresearch/DATE_ALGORITHM_CLOSURE_AUDIT_20260805.json",
            REPO / "neuron_autoresearch/DATE_ALGORITHM_CLOSURE_AUDIT_20260805.md",
        ),
    ),
)


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def task_complete(task: Task) -> bool:
    if not task.completion_path.is_file():
        return False
    if any(not path.is_file() for path in task.required_paths):
        return False
    if task.completion_json_status is not None:
        try:
            payload = json.loads(task.completion_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        return payload.get("status") == task.completion_json_status
    if task.completion_marker is None:
        raise RuntimeError(f"task has no completion contract: {task.name}")
    return task.completion_marker in task.completion_path.read_text(
        encoding="utf-8", errors="replace"
    )


def read_pid(path: Path) -> int | None:
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError, OSError):
        return None
    return value if value > 1 else None


def pid_cmdline(pid: int) -> tuple[str, ...] | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return tuple(item.decode(errors="replace") for item in raw.split(b"\0") if item)


def expected_script_name(task: Task) -> str | None:
    return next((Path(item).name for item in task.command if item.endswith(".py")), None)


def cmdline_matches(task: Task, actual: tuple[str, ...] | None) -> bool:
    if actual is None:
        return False
    expected_script = expected_script_name(task)
    return expected_script is not None and any(Path(item).name == expected_script for item in actual)


def matching_pids(task: Task) -> list[int]:
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if cmdline_matches(task, pid_cmdline(pid)):
            matches.append(pid)
    return sorted(matches)


def task_alive(task: Task) -> bool:
    pid = read_pid(task.pid_file)
    if pid is not None and cmdline_matches(task, pid_cmdline(pid)):
        return True
    discovered = matching_pids(task)
    if not discovered:
        return False
    # Adopt an already-running detached follower instead of launching a process
    # that would immediately lose the follower's own flock and burn retries.
    task.pid_file.write_text(f"{discovered[0]}\n", encoding="utf-8")
    return True


def reap_children() -> int:
    """Reap exited followers without blocking the supervision loop."""
    count = 0
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            break
        count += 1
    return count


def start_task(task: Task) -> int:
    task.launcher_log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "SDFORMER_USE_MLFLOW": "0",
            "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
            "SDFORMER_SNN_BACKEND": "cupy",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    with task.launcher_log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            list(task.command),
            cwd=REPO,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    task.pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
    return process.pid


def main() -> int:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            record("SKIP DATE closure watchdog lock already held")
            return 0
        PID_FILE.write_text(f"{os.getpid()}\n", encoding="utf-8")
        restarts = {task.name: 0 for task in TASKS}
        heartbeat = 0
        record(
            "START DATE closure watchdog tasks="
            + ",".join(task.name for task in TASKS)
        )
        while True:
            reaped = reap_children()
            if reaped:
                record(f"REAP exited_followers={reaped}")
            incomplete = [task for task in TASKS if not task_complete(task)]
            if not incomplete:
                record(f"ALL COMPLETE DATE closure watchdog restarts={restarts}")
                return 0
            for task in incomplete:
                if task_alive(task):
                    continue
                if restarts[task.name] >= MAX_RESTARTS:
                    raise RuntimeError(
                        f"{task.name} exceeded {MAX_RESTARTS} watchdog restarts"
                    )
                restarts[task.name] += 1
                pid = start_task(task)
                record(f"RESTART {task.name} attempt={restarts[task.name]} pid={pid}")
            if heartbeat % 30 == 0:
                alive = [task.name for task in incomplete if task_alive(task)]
                record(
                    f"HEARTBEAT incomplete={len(incomplete)} alive={len(alive)} "
                    f"tasks={','.join(alive)}"
                )
            heartbeat += 1
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
