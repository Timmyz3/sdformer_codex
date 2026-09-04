#!/usr/bin/env python3
"""等待fullres deploy完成后运行Local5 QFSA/FCSR post-G0 profile。"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path


HW_ROOT = Path(__file__).resolve().parents[1]
REPO = HW_ROOT.parent
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
RESULTS = EXP / "results"
GEN = EXP / "configs/generated"
DEPLOY_STATUS = RESULTS / "dsec_fullres_paper_w15_deploy_followup_status.log"
RUN_DIR = (
    RESULTS
    / "dsec_fullres_paper_w15_h66d_local5_ep29_ft30_bs2_20260728"
)
RANKING = RUN_DIR / "profile_ranking_valid825.md"
CONFIG = (
    GEN
    / "dsec_fullres_paper_w15_h66d_local5_ep29_ft30_"
    "hardware_order_q7q17_deploy.yml"
)
OUTPUT = HW_ROOT / "results/local5_fullres_postg0_qfsa_profile100_20260730"
REPLAY = HW_ROOT / "results/local5_fullres_postg0_qfsa_replay_20260730"
DESCRIPTOR_ANALYSIS = (
    HW_ROOT
    / "results/local5_fullres_ds_flm_descriptor_analysis_20260731"
)
ACCEPTANCE = (
    HW_ROOT / "results/local5_fullres_postg0_acceptance_20260731"
)
STATUS = HW_ROOT / "results/local5_fullres_postg0_qfsa_watcher_20260730.log"
LOCK = HW_ROOT / "results/local5_fullres_postg0_qfsa_watcher_20260730.lock"
RUN_IDENTITY = OUTPUT / "post_g0_run_identity.json"
RELEASE_RECEIPT = OUTPUT / "post_g0_release_receipt.json"
SAMPLING_ID = "coprime_rotating_flat_window_head_v1"
DATASET_SAMPLING_ID = "sequence_proportional_temporal_midpoint_v1"
RELEASE_MARKER = "ALL COMPLETE fullres deploy followup"


def source_binding_paths() -> dict[str, Path]:
    baseline = REPO / "third_party/SDformerFlow"
    overlay = EXP / "overlay/models/STSwinNet_SNN"
    return {
        "watcher": Path(__file__).resolve(),
        "profiler": HW_ROOT / "scripts/profile_local5_hardware_features.py",
        "base_profiler": EXP / "entrypoints/profile_nts11_hardware_p0.py",
        "attention_impl": overlay / "bsa_attention.py",
        "checkpoint_loader": overlay / "h9_load_audit.py",
        "model_impl": (
            baseline
            / "models/STSwinNet_SNN/Spiking_STSwinNet.py"
        ),
        "dataset_impl": (
            baseline / "DSEC_dataloader/DSEC_dataset_lite.py"
        ),
        "trace_loader": HW_ROOT / "scripts/et3_ordered_trace_replay.py",
        "replay": HW_ROOT / "scripts/replay_local5_frontier_trace.py",
        "descriptor_analyzer": (
            HW_ROOT / "scripts/analyze_ds_flm_descriptor_manifest.py"
        ),
        "acceptance": (
            HW_ROOT / "scripts/validate_local5_postg0_acceptance.py"
        ),
        "relation_transpose_rtl": (
            HW_ROOT / "rtl_qfit/qfit_relation_transpose_leaf.sv"
        ),
        "retirement_scheduler_rtl": (
            HW_ROOT / "rtl_qfit/qfit_retirement_scheduler.sv"
        ),
        "relation_sync_bank_rtl": (
            HW_ROOT / "rtl_qfit/qfit_sync_1r1w_bank.sv"
        ),
        "relation_assertions": (
            HW_ROOT / "verif_qfit/qfit_relation_transpose_assertions.sv"
        ),
        "relation_sync_bank_assertions": (
            HW_ROOT / "verif_qfit/qfit_sync_bank_assertions.sv"
        ),
        "relation_vector_generator": (
            HW_ROOT / "scripts/generate_local5_relation_transpose_vectors.py"
        ),
        "relation_miter_tb": (
            HW_ROOT / "tb_qfit/tb_qfit_relation_transpose_python_miter.sv"
        ),
        "relation_miter_script": (
            HW_ROOT / "sim_qfit/run_qfit_relation_transpose_python_miter.sh"
        ),
        "score_trace_generator": (
            HW_ROOT / "scripts/generate_local5_checkpoint_score_vectors.py"
        ),
        "score_trace_reporter": (
            HW_ROOT / "scripts/report_local5_checkpoint_score_rtl.py"
        ),
        "score_trace_tb": (
            HW_ROOT / "tb_local5/tb_local5_score_shiftmax_vectors.sv"
        ),
        "score_trace_script": (
            HW_ROOT / "sim_local5/run_local5_checkpoint_score_trace_checks.sh"
        ),
        "projection_quantizer": EXP / "entrypoints/h67_bit_trace.py",
        "projection_contract_verifier": (
            HW_ROOT / "scripts/verify_local5_theta_folded_projection_contract.py"
        ),
        "projection_trace_generator": (
            HW_ROOT / "scripts/generate_local5_active_projection_postg0_vectors.py"
        ),
        "projection_trace_reporter": (
            HW_ROOT / "scripts/summarize_local5_gasr2c_fivebank_rtl.py"
        ),
        "projection_trace_tb": (
            HW_ROOT / "tb_qfit/tb_qfit_local5_active_projection_postg0.sv"
        ),
        "projection_trace_script": (
            HW_ROOT / "sim_new_arch/run_local5_qgasr2c_fivebank_checks.sh"
        ),
    }


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def best_epoch(path: Path) -> int:
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*1\s*\|\s*(\d+)\s*\|", line)
        if match:
            return int(match.group(1))
    raise RuntimeError(f"无法解析rank-1 epoch: {path}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_artifact_binding() -> dict[str, object] | None:
    if not RANKING.is_file() or not CONFIG.is_file():
        return None
    try:
        epoch = best_epoch(RANKING)
    except (OSError, RuntimeError):
        return None
    checkpoint = RUN_DIR / f"checkpoint_epoch{epoch}.pth"
    if not checkpoint.is_file():
        return None
    return {
        "ranking_path": str(RANKING.resolve()),
        "ranking_sha256": file_sha256(RANKING),
        "best_epoch": epoch,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "config_path": str(CONFIG.resolve()),
        "config_sha256": file_sha256(CONFIG),
    }


def validate_release_receipt(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        status_path = Path(str(value["status_path"])).resolve()
        status = status_path.read_bytes()
        prefix_bytes = int(value["status_prefix_bytes"])
        marker_start = int(value["marker_start_offset"])
        marker_end = int(value["marker_end_offset"])
        marker_line = str(value["marker_line"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    current_binding = release_artifact_binding()
    if (
        value.get("schema") != "local5_release_receipt_v2"
        or value.get("release_marker") != RELEASE_MARKER
        or current_binding is None
        or any(value.get(key) != expected for key, expected in current_binding.items())
        or not 0 <= prefix_bytes <= marker_start < marker_end <= len(status)
        or hashlib.sha256(status[:prefix_bytes]).hexdigest()
        != value.get("status_prefix_sha256")
        or status[marker_start:marker_end].decode(
            "utf-8", errors="strict"
        ).rstrip("\n") != marker_line
        or RELEASE_MARKER not in marker_line
        or "H67" not in marker_line
        or "H66d" not in marker_line
    ):
        return None
    return value


def wait_for_marker(
    *,
    marker: str,
    timeout_hours: float,
    poll_seconds: int,
) -> dict[str, object]:
    existing = validate_release_receipt(RELEASE_RECEIPT)
    if existing is not None:
        record(
            "RELEASE receipt复用 uuid="
            + str(existing["watcher_session_uuid"])
        )
        return existing
    watcher_session_uuid = str(uuid.uuid4())
    baseline = DEPLOY_STATUS.read_bytes() if DEPLOY_STATUS.is_file() else b""
    baseline_size = len(baseline)
    baseline_hash = hashlib.sha256(baseline).hexdigest()
    deadline = time.time() + timeout_hours * 3600
    while time.time() < deadline:
        status = DEPLOY_STATUS.read_bytes() if DEPLOY_STATUS.is_file() else b""
        if status[:baseline_size] != baseline:
            raise RuntimeError("deploy status历史前缀被改写，拒绝继续")
        appended = status[baseline_size:]
        marker_bytes = marker.encode("utf-8")
        marker_relative = appended.find(marker_bytes)
        if marker_relative >= 0:
            line_start_relative = appended.rfind(
                b"\n", 0, marker_relative
            ) + 1
            line_end_relative = appended.find(b"\n", marker_relative)
            if line_end_relative < 0:
                line_end_relative = len(appended)
            else:
                line_end_relative += 1
            marker_start = baseline_size + line_start_relative
            marker_end = baseline_size + line_end_relative
            marker_line = status[marker_start:marker_end].decode(
                "utf-8", errors="strict"
            ).rstrip("\n")
            if "H67" not in marker_line or "H66d" not in marker_line:
                raise RuntimeError("deploy release marker未绑定H67和H66d")
            artifact_binding = release_artifact_binding()
            if artifact_binding is None:
                raise RuntimeError("deploy marker已到但ranking/checkpoint/config不完整")
            receipt: dict[str, object] = {
                "schema": "local5_release_receipt_v2",
                "watcher_session_uuid": watcher_session_uuid,
                "release_marker": marker,
                "marker_line": marker_line,
                "status_path": str(DEPLOY_STATUS.resolve()),
                "status_prefix_bytes": baseline_size,
                "status_prefix_sha256": baseline_hash,
                "marker_start_offset": marker_start,
                "marker_end_offset": marker_end,
                **artifact_binding,
            }
            RELEASE_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
            temporary = RELEASE_RECEIPT.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(RELEASE_RECEIPT)
            record(
                f"RELEASE marker={marker} uuid={watcher_session_uuid}"
            )
            return receipt
        record("WAIT fullres deploy follower")
        time.sleep(max(30, poll_seconds))
    raise TimeoutError(f"等待marker超时: {marker}")


def run(command: list[str]) -> None:
    record("START " + " ".join(command))
    subprocess.run(command, cwd=HW_ROOT, check=True)
    record("END " + command[1])


def gpu_compute_used_mib() -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return sum(
        int(line.strip())
        for line in completed.stdout.splitlines()
        if line.strip().isdigit()
    )


def wait_for_gpu_capacity(
    *,
    max_used_mib: int,
    timeout_hours: float,
    poll_seconds: int,
) -> None:
    deadline = time.time() + timeout_hours * 3600
    while time.time() < deadline:
        used_mib = gpu_compute_used_mib()
        if used_mib <= max_used_mib:
            record(
                f"GPU RELEASE used_mib={used_mib} threshold={max_used_mib}"
            )
            return
        record(
            f"WAIT GPU used_mib={used_mib} threshold={max_used_mib}"
        )
        time.sleep(max(30, poll_seconds))
    raise TimeoutError("等待GPU显存释放超时")


def has_v3_descriptor_contract(manifest_path: Path) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("evidence_level") == "post_g0"
        and value.get("source_descriptor_contract", {}).get("id")
        == "qfit_relation_transpose_source_descriptor_v3"
        and value.get("qualification", {}).get("qualified") is True
        and value.get("run_identity_file_sha256")
        == file_sha256(RUN_IDENTITY)
    )


def write_run_identity(
    *,
    checkpoint: Path,
    samples: int,
    groups_per_block_sample: int,
    release_receipt: dict[str, object],
) -> None:
    relation_rtl = HW_ROOT / "rtl_qfit/qfit_relation_transpose_leaf.sv"
    value = {
        "schema": "local5_post_g0_run_identity_v3",
        "release_marker": RELEASE_MARKER,
        "deploy_status": str(DEPLOY_STATUS.resolve()),
        "release_receipt": str(RELEASE_RECEIPT.resolve()),
        "release_receipt_sha256": file_sha256(RELEASE_RECEIPT),
        "watcher_session_uuid": release_receipt["watcher_session_uuid"],
        "ranking": str(RANKING.resolve()),
        "ranking_sha256": file_sha256(RANKING),
        "config": str(CONFIG.resolve()),
        "config_sha256": file_sha256(CONFIG),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "best_epoch": best_epoch(RANKING),
        "relation_rtl": str(relation_rtl.resolve()),
        "relation_rtl_sha256": file_sha256(relation_rtl),
        "samples": samples,
        "groups_per_block_sample": groups_per_block_sample,
        "sampling_id": SAMPLING_ID,
        "dataset_sampling_id": DATASET_SAMPLING_ID,
        "source_bindings": {
            name: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for name, path in source_binding_paths().items()
        },
    }
    RUN_IDENTITY.parent.mkdir(parents=True, exist_ok=True)
    temporary = RUN_IDENTITY.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(RUN_IDENTITY)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--ordered-groups", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--timeout-hours", type=float, default=240.0)
    parser.add_argument(
        "--max-gpu-used-mib",
        type=int,
        default=4096,
        help="启动profile前允许的全卡compute显存占用上限",
    )
    args = parser.parse_args()

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            record("SKIP watcher lock已被占用")
            return 0

        release_receipt = wait_for_marker(
            marker=RELEASE_MARKER,
            timeout_hours=args.timeout_hours,
            poll_seconds=args.poll_seconds,
        )
        if not RANKING.is_file():
            raise FileNotFoundError(RANKING)
        epoch = best_epoch(RANKING)
        checkpoint = RUN_DIR / f"checkpoint_epoch{epoch}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if not CONFIG.is_file():
            raise FileNotFoundError(CONFIG)
        wait_for_gpu_capacity(
            max_used_mib=args.max_gpu_used_mib,
            timeout_hours=args.timeout_hours,
            poll_seconds=args.poll_seconds,
        )
        write_run_identity(
            checkpoint=checkpoint,
            samples=args.samples,
            groups_per_block_sample=args.ordered_groups,
            release_receipt=release_receipt,
        )

        manifest = OUTPUT / "ordered_term_manifest.json"
        if not has_v3_descriptor_contract(manifest):
            run(
                [
                    "/opt/conda/envs/sdformerflow/bin/python",
                    "scripts/profile_local5_hardware_features.py",
                    "--config",
                    str(CONFIG),
                    "--checkpoint",
                    str(checkpoint),
                    "--output-dir",
                    str(OUTPUT),
                    "--samples",
                    str(args.samples),
                    "--num-workers",
                    "0",
                    "--ordered-groups-per-block-sample",
                    str(args.ordered_groups),
                    "--ordered-evidence-level",
                    "post_g0",
                    "--run-identity",
                    str(RUN_IDENTITY),
                ]
            )
        run(
            [
                "/opt/conda/envs/sdformerflow/bin/python",
                "scripts/replay_local5_frontier_trace.py",
                "--manifest",
                str(manifest),
                "--output-dir",
                str(REPLAY),
            ]
        )
        run(
            [
                "/opt/conda/envs/sdformerflow/bin/python",
                "scripts/analyze_ds_flm_descriptor_manifest.py",
                "--manifest",
                str(manifest),
                "--output-dir",
                str(DESCRIPTOR_ANALYSIS),
            ]
        )
        run(
            [
                "/opt/conda/envs/sdformerflow/bin/python",
                "scripts/validate_local5_postg0_acceptance.py",
                "--manifest",
                str(manifest),
                "--replay-report",
                str(REPLAY / "report.json"),
                "--descriptor-report",
                str(DESCRIPTOR_ANALYSIS / "report.json"),
                "--run-identity",
                str(RUN_IDENTITY),
                "--output-dir",
                str(ACCEPTANCE),
            ]
        )
        record(
            "ALL COMPLETE Local5 fullres post-G0 "
            "QFSA/FCSR profile+replay+DS-FLM descriptor analysis"
            "+fail-closed acceptance"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
