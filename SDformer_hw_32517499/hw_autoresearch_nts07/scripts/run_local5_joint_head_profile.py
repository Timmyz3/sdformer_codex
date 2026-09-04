#!/usr/bin/env python3
"""等待 Motion T450 profile 释放后导出 Local5 最终同窗全 head trace。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
RUN = EXP / "results/dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805"
RANKING = RUN / "profile_ranking_valid825.md"
CHECKPOINT = RUN / "checkpoint_epoch29.pth"
CONFIG = EXP / (
    "configs/generated/"
    "dsec_fullres_w15_H66d_local5_bb1e4_ft30_hardware_order_q7q17_deploy.yml"
)
BASE_IDENTITY = ROOT / (
    "results/local5_fullres_bb1e4_postg0_profile100_20260805/"
    "post_g0_run_identity.json"
)
OUTPUT = ROOT / "results/local5_fullres_bb1e4_joint_heads_profile100_20260809"
IDENTITY = OUTPUT / "joint_head_run_identity.json"
SELECTION_PLAN = OUTPUT / "joint_window_selection_plan.json"
PLAN_FREEZE_RECEIPT = ROOT / "contracts/local5_joint_trace_plan_freeze_receipt_v1_20260810.json"
GPU_AUDIT = OUTPUT / "gpu_exclusivity_audit.json"
STATUS = ROOT / "results/local5_joint_head_profile_watcher_20260809.log"
LOCK = ROOT / "results/local5_joint_head_profile_watcher_20260809.lock"
MOTION_STATUS = ROOT / "results/h67_fullres_ep30_t450_profile_watcher_20260805.log"
MOTION_LOCK = ROOT / "results/h67_fullres_ep30_t450_profile_watcher_20260805.lock"
MOTION_COMPLETE = "ALL COMPLETE H67 ep30 fullres T450 profile100/all12 trace audit"
GPU_LEASE = ROOT / "results/gpu_profile_lease.lock"
PYTHON = "/opt/conda/envs/sdformerflow/bin/python"
MAX_GPU_USED_MIB = 1024
SAMPLING_ID = "uniform_plan_window_all_heads_v1"
SAMPLING_SEED = 20260809
COHORT_MANIFEST = ROOT / (
    "results/local5_fullres_bb1e4_postg0_profile100_20260805/"
    "ordered_term_manifest.json"
)
STAGE_HEADS = (3, 6, 12, 24)
STAGE_WINDOWS = (440, 120, 30, 10)
STAGE_DEPTHS = (2, 2, 6, 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def gpu_used_mib() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return max(int(line.strip()) for line in result.stdout.splitlines() if line.strip())


def gpu_compute_pids() -> set[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        int(line.strip())
        for line in result.stdout.splitlines()
        if line.strip().isdigit()
    }


def wait_gpu_compute_empty(timeout_seconds: int = 30) -> set[int]:
    deadline = time.monotonic() + timeout_seconds
    observed = gpu_compute_pids()
    while observed and time.monotonic() < deadline:
        time.sleep(2)
        observed = gpu_compute_pids()
    return observed


def descendant_pids(root_pid: int) -> set[int]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid="],
        check=True,
        capture_output=True,
        text=True,
    )
    children: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        pid, parent = (int(field) for field in fields)
        children.setdefault(parent, []).append(pid)
    descendants = {root_pid}
    frontier = [root_pid]
    while frontier:
        parent = frontier.pop()
        for child in children.get(parent, []):
            if child not in descendants:
                descendants.add(child)
                frontier.append(child)
    return descendants


def pid_namespace_host_pids(
    pids: set[int], *, proc_root: Path = Path("/proc")
) -> set[int]:
    """Map descendants to the outermost PID namespace used by nvidia-smi."""

    host_pids: set[int] = set()
    for pid in pids:
        status = proc_root / str(pid) / "status"
        try:
            lines = status.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            host_pids.add(pid)
            continue
        found = False
        for line in lines:
            if line.startswith("NSpid:"):
                fields = [int(field) for field in line.split()[1:] if field.isdigit()]
                if fields:
                    host_pids.add(fields[0])
                    found = True
                break
        if not found:
            host_pids.add(pid)
    return host_pids


def classify_gpu_processes(
    current: set[int],
    namespace_owned: set[int],
    claimed_unmapped: int | None,
    *,
    child_alive: bool,
) -> tuple[int | None, set[int]]:
    """Conservatively claim one NVML PID when host PID mapping is unavailable."""

    unknown = current - namespace_owned
    if current & namespace_owned:
        return claimed_unmapped, unknown
    if (
        claimed_unmapped is None
        and child_alive
        and not (current & namespace_owned)
        and len(unknown) == 1
    ):
        claimed_unmapped = next(iter(unknown))
    allowed_unmapped = {claimed_unmapped} if claimed_unmapped is not None else set()
    return claimed_unmapped, unknown - allowed_unmapped


def write_gpu_audit(status: str, **fields: object) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    value = {
        "schema": "local5_joint_gpu_exclusivity_audit_v1",
        "status": status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **fields,
    }
    GPU_AUDIT.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_process_group(
    process: subprocess.Popen[bytes], *, timeout_seconds: float = 30.0
) -> None:
    pgid = process.pid
    deadline = time.monotonic() + timeout_seconds
    process.poll()  # Reap an already-exited leader before probing its process group.
    if process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    while process_group_exists(pgid) and time.monotonic() < deadline:
        process.poll()
        if not process_group_exists(pgid):
            break
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    if process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=max(0.1, timeout_seconds))
    process.poll()
    if process_group_exists(pgid):
        raise RuntimeError(f"进程组{pgid}在有界终止后仍存在")


def write_gpu_pass_audit(
    *,
    command: list[str],
    child_pid: int,
    polls: int,
    claimed_unmapped_gpu_pid: int | None,
) -> None:
    manifest_path = OUTPUT / "ordered_term_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("profile退出为0但缺少ordered manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = OUTPUT / str(manifest.get("payload_file", ""))
    if not payload.is_file():
        raise RuntimeError("profile退出为0但缺少ordered payload")
    write_gpu_audit(
        "PASS",
        command=command,
        child_pid=child_pid,
        monitor_polls=polls,
        identity_sha256=sha256(IDENTITY),
        manifest_sha256=sha256(manifest_path),
        payload_sha256=sha256(payload),
        foreign_compute_pids=[],
        claimed_unmapped_gpu_pid=claimed_unmapped_gpu_pid,
    )


def run_with_gpu_exclusivity(
    command: list[str], *, env: dict[str, str]
) -> int:
    if gpu_compute_pids():
        raise RuntimeError("启动前目标GPU仍有compute进程")
    process = subprocess.Popen(
        command,
        cwd=REPO,
        env=env,
        start_new_session=True,
    )
    polls = 0
    claimed_unmapped_gpu_pid: int | None = None
    returncode: int | None = None
    failure: BaseException | None = None
    allowed_at_failure: list[int] = []
    remaining_compute: set[int] = set()
    try:
        write_gpu_audit(
            "RUNNING_UNVERIFIED",
            command=command,
            child_pid=process.pid,
            identity_sha256=sha256(IDENTITY),
        )
        while process.poll() is None:
            polls += 1
            allowed = pid_namespace_host_pids(descendant_pids(os.getpid()))
            claimed_unmapped_gpu_pid, foreign = classify_gpu_processes(
                gpu_compute_pids(),
                allowed,
                claimed_unmapped_gpu_pid,
                child_alive=process.poll() is None,
            )
            if foreign:
                record(
                    "ABORT foreign GPU compute pids="
                    + ",".join(str(pid) for pid in sorted(foreign))
                )
                raise RuntimeError(
                    "正式profile期间GPU独占性失效: "
                    + ",".join(str(pid) for pid in sorted(foreign))
                )
            time.sleep(2)
        returncode = int(process.returncode)
    except BaseException as error:
        failure = error
        allowed_at_failure = sorted(
            pid_namespace_host_pids(descendant_pids(os.getpid()))
        )
    finally:
        try:
            stop_process_group(process)
        except BaseException as cleanup_error:
            if failure is None:
                failure = cleanup_error
        remaining_compute = wait_gpu_compute_empty()
        if remaining_compute and failure is None:
            failure = RuntimeError(
                "profile停止后GPU仍有compute PID: "
                + ",".join(str(pid) for pid in sorted(remaining_compute))
            )

    if failure is not None:
        write_gpu_audit(
            "INVALID",
            command=command,
            child_pid=process.pid,
            monitor_polls=polls,
            reason=f"{type(failure).__name__}: {failure}",
            allowed_host_pids=allowed_at_failure,
            claimed_unmapped_gpu_pid=claimed_unmapped_gpu_pid,
            post_stop_remaining_compute_pids=sorted(remaining_compute),
        )
        raise failure
    if returncode is None:
        raise RuntimeError("profile退出但缺少returncode")
    if returncode:
        write_gpu_audit(
            "INVALID",
            command=command,
            child_pid=process.pid,
            monitor_polls=polls,
            reason=f"profile exit_code={returncode}",
            claimed_unmapped_gpu_pid=claimed_unmapped_gpu_pid,
            post_stop_remaining_compute_pids=[],
        )
        return returncode
    write_gpu_pass_audit(
        command=command,
        child_pid=process.pid,
        polls=polls,
        claimed_unmapped_gpu_pid=claimed_unmapped_gpu_pid,
    )
    return returncode


def motion_artifacts_complete() -> bool:
    if not (
        MOTION_STATUS.is_file()
        and MOTION_COMPLETE
        in MOTION_STATUS.read_text(encoding="utf-8", errors="replace")
    ):
        return False
    try:
        import run_h67_ep30_fullres_t450_profile as motion

        return motion.completed_evidence_matches_checkpoint()
    except (ImportError, OSError, RuntimeError, ValueError):
        return False


def motion_lock_is_free() -> bool:
    MOTION_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with MOTION_LOCK.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle, fcntl.LOCK_UN)
    return True


def wait_for_motion_release() -> None:
    while not (
        motion_artifacts_complete() and motion_lock_is_free()
    ):
        record("WAIT Motion fullres T450 artifacts and lock release")
        time.sleep(300)
    while True:
        used = gpu_used_mib()
        if used <= MAX_GPU_USED_MIB:
            record(f"RELEASE GPU memory_used={used}MiB")
            return
        record(f"WAIT GPU memory_used={used}MiB > {MAX_GPU_USED_MIB}MiB")
        time.sleep(300)


def source_paths() -> dict[str, Path]:
    overlay = EXP / "overlay/models/STSwinNet_SNN"
    baseline = REPO / "third_party/SDformerFlow"
    return {
        "runner": Path(__file__).resolve(),
        "joint_profiler": ROOT / "scripts/profile_local5_joint_head_trace.py",
        "base_profiler": ROOT / "scripts/profile_local5_hardware_features.py",
        "network_profiler": EXP / "entrypoints/profile_nts11_hardware_p0.py",
        "attention_impl": overlay / "bsa_attention.py",
        "checkpoint_loader": overlay / "h9_load_audit.py",
        "model_impl": baseline / "models/STSwinNet_SNN/Spiking_STSwinNet.py",
        "dataset_impl": baseline / "DSEC_dataloader/DSEC_dataset_lite.py",
        "trace_contract": ROOT / "scripts/et3_ordered_trace_replay.py",
        "projection_quantizer": EXP / "entrypoints/h67_bit_trace.py",
    }


def rank1_epoch() -> int:
    for line in RANKING.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*1\s*\|\s*(\d+)\s*\|", line)
        if match:
            return int(match.group(1))
    raise RuntimeError("无法解析Local5 rank-1 epoch")


def uniform_window(sample: int, stage: int, block: int) -> int:
    material = f"{SAMPLING_SEED}:{sample}:{stage}:{block}".encode("ascii")
    local_seed = int.from_bytes(hashlib.sha256(material).digest()[:16], "big")
    return random.Random(local_seed).randrange(STAGE_WINDOWS[stage])


def build_selection_plan() -> dict[str, object]:
    source_manifest = json.loads(COHORT_MANIFEST.read_text(encoding="utf-8"))
    cohort_sha = str(source_manifest["cohort_sha256"])
    records = []
    for sample in range(100):
        for stage, depth in enumerate(STAGE_DEPTHS):
            for block in range(depth):
                windows = STAGE_WINDOWS[stage]
                records.append(
                    {
                        "sample": sample,
                        "stage": stage,
                        "block": block,
                        "heads": STAGE_HEADS[stage],
                        "batch_windows": windows,
                        "window": uniform_window(sample, stage, block),
                        "inclusion_probability": 1.0 / windows,
                        "analysis_weight": float(windows),
                    }
                )
    return {
        "schema": "local5_uniform_joint_window_plan_v1",
        "sampling_id": SAMPLING_ID,
        "seed": SAMPLING_SEED,
        "cohort_sha256": cohort_sha,
        "source_cohort_manifest": str(COHORT_MANIFEST.resolve()),
        "source_cohort_manifest_sha256": sha256(COHORT_MANIFEST),
        "probability_contract": (
            "one data-independent uniform draw without replacement per "
            "sample/block; pi=1/batch_windows"
        ),
        "analysis_contract": (
            "cluster by sample; use batch_windows as Horvitz-Thompson weight"
        ),
        "records": records,
    }


def write_selection_plan() -> dict[str, object]:
    value = build_selection_plan()
    expected = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            SELECTION_PLAN,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        if SELECTION_PLAN.read_bytes() != expected:
            raise RuntimeError("selection plan已存在但字节不匹配；拒绝覆盖")
        return value
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        SELECTION_PLAN.unlink(missing_ok=True)
        raise
    return value


def validate_plan_freeze_receipt(plan: dict[str, object]) -> dict[str, object]:
    if not PLAN_FREEZE_RECEIPT.is_file():
        raise RuntimeError("缺少joint trace selection-plan freeze receipt")
    receipt = json.loads(PLAN_FREEZE_RECEIPT.read_text(encoding="utf-8"))
    blob_oid = str(receipt.get("selection_plan_git_blob", ""))
    blob = subprocess.run(
        ["git", "cat-file", "blob", blob_oid],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if (
        receipt.get("schema") != "local5_joint_trace_plan_freeze_receipt_v1"
        or receipt.get("status") != "LOCAL_BYTE_ANCHOR_NOT_EXTERNAL_TIMESTAMP"
        or Path(str(receipt.get("selection_plan", ""))).resolve()
        != SELECTION_PLAN.resolve()
        or receipt.get("selection_plan_sha256") != sha256(SELECTION_PLAN)
        or Path(str(receipt.get("generator", ""))).resolve()
        != Path(__file__).resolve()
        or receipt.get("generator_sha256") != sha256(Path(__file__).resolve())
        or receipt.get("sampling_id") != SAMPLING_ID
        or receipt.get("sampling_seed") != SAMPLING_SEED
        or receipt.get("cohort_sha256") != plan.get("cohort_sha256")
        or receipt.get("records") != len(plan.get("records", []))
        or not re.fullmatch(r"[0-9a-f]{40}", blob_oid)
        or blob.returncode != 0
        or blob.stdout != SELECTION_PLAN.read_bytes()
    ):
        raise RuntimeError("selection-plan freeze receipt绑定失效")
    return receipt


def write_identity() -> None:
    base_identity = json.loads(BASE_IDENTITY.read_text(encoding="utf-8"))
    plan = write_selection_plan()
    validate_plan_freeze_receipt(plan)
    if (
        base_identity.get("checkpoint_sha256") != sha256(CHECKPOINT)
        or base_identity.get("config_sha256") != sha256(CONFIG)
        or Path(str(base_identity.get("ranking", ""))).resolve() != RANKING.resolve()
    ):
        raise RuntimeError("最终Local5 base identity与epoch29不一致")
    if rank1_epoch() != 29:
        raise RuntimeError("最终Local5 rank-1不再是epoch29")
    receipt = Path(str(base_identity["release_receipt"])).resolve()
    value = {
        "schema": "local5_joint_head_run_identity_v1",
        "ranking": str(RANKING.resolve()),
        "ranking_sha256": sha256(RANKING),
        "best_epoch": 29,
        "config": str(CONFIG.resolve()),
        "config_sha256": sha256(CONFIG),
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "release_receipt": str(receipt),
        "release_receipt_sha256": sha256(receipt),
        "samples": 100,
        "groups_per_block_sample": 24,
        "joint_windows_per_block_sample": 1,
        "sampling_id": SAMPLING_ID,
        "sampling_seed": SAMPLING_SEED,
        "selection_plan": str(SELECTION_PLAN.resolve()),
        "selection_plan_sha256": sha256(SELECTION_PLAN),
        "selection_plan_freeze_receipt": str(PLAN_FREEZE_RECEIPT.resolve()),
        "selection_plan_freeze_receipt_sha256": sha256(PLAN_FREEZE_RECEIPT),
        "cohort_sha256": plan["cohort_sha256"],
        "dataset_sampling_id": "sequence_proportional_temporal_midpoint_v1",
        "source_bindings": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in source_paths().items()
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    IDENTITY.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def output_complete() -> bool:
    manifest_path = OUTPUT / "ordered_term_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        identity = json.loads(IDENTITY.read_text(encoding="utf-8"))
        gpu_audit = json.loads(GPU_AUDIT.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    groups = manifest.get("groups") or []
    qualification = manifest.get("qualification") or {}
    file_bindings = (
        ("payload_file", "payload_sha256"),
        ("cohort_file", "cohort_file_sha256"),
        ("projection_contract_file", "projection_contract_file_sha256"),
        ("projection_contract_payload", "projection_contract_payload_sha256"),
    )
    files_ok = True
    for path_key, sha_key in file_bindings:
        artifact = OUTPUT / str(manifest.get(path_key, ""))
        files_ok &= artifact.is_file() and manifest.get(sha_key) == sha256(artifact)
    source_bindings = identity.get("source_bindings") or {}
    sources_ok = set(source_bindings) == set(source_paths())
    if sources_ok:
        for name, path in source_paths().items():
            binding = source_bindings[name]
            sources_ok &= (
                Path(str(binding.get("path", ""))).resolve() == path.resolve()
                and path.is_file()
                and binding.get("sha256") == sha256(path)
            )
    plan_ok = (
        SELECTION_PLAN.is_file()
        and identity.get("selection_plan_sha256") == sha256(SELECTION_PLAN)
        and manifest.get("sampling", {}).get("selection_plan_sha256")
        == sha256(SELECTION_PLAN)
    )
    try:
        validate_plan_freeze_receipt(json.loads(SELECTION_PLAN.read_text(encoding="utf-8")))
        receipt_ok = (
            identity.get("selection_plan_freeze_receipt")
            == str(PLAN_FREEZE_RECEIPT.resolve())
            and identity.get("selection_plan_freeze_receipt_sha256")
            == sha256(PLAN_FREEZE_RECEIPT)
        )
    except (json.JSONDecodeError, OSError, RuntimeError):
        receipt_ok = False
    payload_path = OUTPUT / str(manifest.get("payload_file", ""))
    gpu_audit_ok = (
        gpu_audit.get("schema") == "local5_joint_gpu_exclusivity_audit_v1"
        and gpu_audit.get("status") == "PASS"
        and gpu_audit.get("foreign_compute_pids") == []
        and gpu_audit.get("identity_sha256") == sha256(IDENTITY)
        and gpu_audit.get("manifest_sha256") == sha256(manifest_path)
        and payload_path.is_file()
        and gpu_audit.get("payload_sha256") == sha256(payload_path)
    )
    return (
        files_ok
        and sources_ok
        and plan_ok
        and receipt_ok
        and gpu_audit_ok
        and
        manifest.get("checkpoint_sha256") == sha256(CHECKPOINT)
        and manifest.get("config_sha256") == sha256(CONFIG)
        and Path(str(manifest.get("run_identity_file", ""))).resolve()
        == IDENTITY.resolve()
        and manifest.get("run_identity_file_sha256") == sha256(IDENTITY)
        and manifest.get("cohort_sha256") == identity.get("cohort_sha256")
        and manifest.get("sampling", {}).get("method")
        == SAMPLING_ID
        and qualification.get("qualified") is True
        and qualification.get("captured_groups") == 13800
        and len(groups) == 13800
        and all(
            group.get("selection") == SAMPLING_ID
            for group in groups
        )
    )


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            record("EXIT another Local5 joint-head watcher owns the lock")
            return 0
        if output_complete():
            record("REUSE completed Local5 same-window all-head trace")
            return 0
        wait_for_motion_release()
        GPU_LEASE.parent.mkdir(parents=True, exist_ok=True)
        with GPU_LEASE.open("a", encoding="utf-8") as gpu_handle:
            fcntl.flock(gpu_handle, fcntl.LOCK_EX)
            try:
                used = gpu_used_mib()
                if used > MAX_GPU_USED_MIB:
                    raise RuntimeError(
                        f"GPU lease后显存回升到{used}MiB，拒绝启动"
                    )
                write_identity()
                command = [
                    PYTHON,
                    "hw_autoresearch_nts07/scripts/profile_local5_joint_head_trace.py",
                    "--config",
                    str(CONFIG),
                    "--checkpoint",
                    str(CHECKPOINT),
                    "--output-dir",
                    str(OUTPUT),
                    "--samples",
                    "100",
                    "--num-workers",
                    "0",
                    "--ordered-groups-per-block-sample",
                    "24",
                    "--ordered-evidence-level",
                    "post_g0",
                    "--run-identity",
                    str(IDENTITY),
                ]
                env = os.environ.copy()
                env.update(
                    {
                        "SDFORMER_USE_MLFLOW": "0",
                        "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
                        "SDFORMER_SNN_BACKEND": "cupy",
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                        "CUDA_VISIBLE_DEVICES": os.environ.get(
                            "SDFORMER_PROFILE_GPU", "0"
                        ),
                    }
                )
                record("START " + " ".join(command))
                returncode = run_with_gpu_exclusivity(command, env=env)
            finally:
                fcntl.flock(gpu_handle, fcntl.LOCK_UN)
        record(f"END joint-head profile exit_code={returncode}")
        if returncode or not output_complete():
            raise RuntimeError("Local5 same-window all-head正式profile失败")
        record("ALL COMPLETE Local5 epoch29 same-window all-head profile100")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
