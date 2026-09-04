"""Queue equal +10 full-resolution convergence audits for Local5, H67, and NB0."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
H67_EP30_RTL = (
    REPO
    / "hw_autoresearch_nts07/results/h67_fullres_ep30_t450_score_shiftmax_rtl_20260805/report.json"
)
H67_EP30_ATLIF_RTL = (
    REPO
    / "hw_autoresearch_nts07/results/h67_ep30_checkpoint_atlif_dptme_rtl_20260805/report.json"
)
H67_EP30_PROJECTION_RTL = (
    REPO
    / "hw_autoresearch_nts07/results/h67_ep30_checkpoint_projection_rtl_20260805/report.json"
)
H67_EP30_STATUS = (
    REPO
    / "hw_autoresearch_nts07/results/h67_fullres_ep30_t450_profile_watcher_20260805.log"
)
H67_EP30_CONFIG = GEN / (
    "dsec_fullres_w15_H67_crop_bb1e4_resume_ep30_"
    "hardware_order_q7q17_deploy.yml"
)
H67_EP30_PROFILE = REPO / (
    "hw_autoresearch_nts07/results/h67_fullres_ep30_t450_profile100_20260805/"
    "nts11_hardware_p0_profile.json"
)
H67_EP30_TRACE = REPO / (
    "hw_autoresearch_nts07/results/h67_fullres_ep30_t450_all12_bit_trace_20260805/"
    "manifest.json"
)
H67_EP30_AUDIT = REPO / (
    "hw_autoresearch_nts07/results/"
    "h67_fullres_ep30_t450_all12_bit_trace_audit_20260805/audit.json"
)
H67_EP30_COMPLETE = (
    "ALL COMPLETE H67 ep30 fullres T450 profile100/all12 trace audit/"
    "score-Shiftmax, ATLIF DP-TME, and real-weight projection component RTL"
)
LOCAL5_RUN = RESULTS / "dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805"
LOCAL5_RANKING = LOCAL5_RUN / "profile_ranking_valid825.md"
LOCAL5_HARDWARE_CONFIG = GEN / (
    "dsec_fullres_w15_H66d_local5_bb1e4_ft30_hardware_order_q7q17_deploy.yml"
)
LOCAL5_RTL = REPO / (
    "hw_autoresearch_nts07/results/local5_bb1e4_qgasr2c_fivebank_postg0_rtl_20260805/"
    "checkpoint_bound_scope.json"
)
LOCAL5_RTL_STATUS = REPO / (
    "hw_autoresearch_nts07/results/local5_bb1e4_checkpoint_bound_rtl_watcher_20260805.log"
)
LOCAL5_RTL_COMPLETE = (
    "ALL COMPLETE checkpoint-bound Local-5 score/Shiftmax, projection partial "
    "accumulator, and ATLIF temporal-matrix component RTL exact"
)
LOCAL5_PROJECTION_WEIGHT_MODE = "checkpoint_theta_folded_dyadic_int8_head_slice"
CONVERGENCE_CRITERION = (
    "not_plateaued iff the largest observed budget is AEE rank1; "
    "last5 slope is descriptive only"
)
STATUS = RESULTS / "dsec_fullres_w15_equal_plus10_convergence_20260805.log"
LOCK = RESULTS / "dsec_fullres_w15_equal_plus10_convergence_20260805.lock"
SUMMARY_JSON = RESULTS / "dsec_fullres_w15_equal_plus10_convergence_summary_20260805.json"
SUMMARY_MD = RESULTS / "dsec_fullres_w15_equal_plus10_convergence_summary_20260805.md"
PYTHON = Path(sys.executable)


@dataclass(frozen=True)
class Candidate:
    name: str
    config: Path
    source_model: Path
    source_state: Path
    root: Path
    source_label: int
    final_label: int
    eval_labels: tuple[int, ...]
    expected_overlay_keys: int
    expected_atlif: int
    expected_shiftmax: int


H67 = Candidate(
    name="H67",
    config=GEN / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml",
    source_model=RESULTS / "dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804/checkpoint_epoch30.pth",
    source_state=RESULTS / "dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804/checkpoint_epoch30_state_dict.pth",
    root=RESULTS / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805",
    source_label=30,
    final_label=40,
    eval_labels=(30, 35, 40),
    expected_overlay_keys=210,
    expected_atlif=105,
    expected_shiftmax=12,
)
LOCAL5 = Candidate(
    name="Local5",
    config=GEN / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus10_ep40.yml",
    source_model=LOCAL5_RUN / "checkpoint_epoch29.pth",
    source_state=LOCAL5_RUN / "checkpoint_epoch29_state_dict.pth",
    root=RESULTS / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus10_ep40_20260809",
    source_label=29,
    final_label=39,
    eval_labels=(29, 34, 39),
    expected_overlay_keys=210,
    expected_atlif=105,
    expected_shiftmax=12,
)
NB0 = Candidate(
    name="NB0",
    config=GEN / "dsec_fullres_w15_NB0_equal_plus10_ep40.yml",
    source_model=RESULTS / "dsec_fullres_paper_w15_nb0_ep59_ft30_bs2_20260728/checkpoint_epoch29.pth",
    source_state=RESULTS / "dsec_fullres_paper_w15_nb0_ep59_ft30_bs2_20260728/checkpoint_epoch29_state_dict.pth",
    root=RESULTS / "dsec_fullres_w15_NB0_equal_plus10_ep40_20260805",
    source_label=29,
    final_label=39,
    eval_labels=(29, 34, 39),
    expected_overlay_keys=0,
    expected_atlif=0,
    expected_shiftmax=0,
)
CANDIDATES = (LOCAL5, H67, NB0)


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
    record(f"START {label}: {' '.join(command)}")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=REPO,
            env=environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    record(f"END {label}: exit_code={result.returncode}")
    if result.returncode:
        raise RuntimeError(f"{label} failed; see {log}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def h67_ep30_evidence_complete() -> bool:
    required = (
        H67.source_model,
        H67_EP30_CONFIG,
        H67_EP30_PROFILE,
        H67_EP30_TRACE,
        H67_EP30_AUDIT,
        H67_EP30_RTL,
        H67_EP30_ATLIF_RTL,
        H67_EP30_PROJECTION_RTL,
        H67_EP30_STATUS,
    )
    if not all(path.is_file() for path in required):
        return False
    if H67_EP30_COMPLETE not in H67_EP30_STATUS.read_text(
        encoding="utf-8", errors="replace"
    ):
        return False
    try:
        profile = json.loads(H67_EP30_PROFILE.read_text(encoding="utf-8"))
        trace = json.loads(H67_EP30_TRACE.read_text(encoding="utf-8"))
        audit = json.loads(H67_EP30_AUDIT.read_text(encoding="utf-8"))
        score = json.loads(H67_EP30_RTL.read_text(encoding="utf-8"))
        atlif = json.loads(H67_EP30_ATLIF_RTL.read_text(encoding="utf-8"))
        projection = json.loads(H67_EP30_PROJECTION_RTL.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    checkpoint_sha = sha256(H67.source_model)
    config_sha = sha256(H67_EP30_CONFIG)
    profile_identity = profile.get("artifact_identity", {})
    profile_protocol = profile.get("eval_protocol", {})
    profile_counts = profile.get("module_counts", {})
    trace_identity = trace.get("run_context", {}).get("artifact_identity", {})
    trace_records = trace.get("records") or []
    score_identity = score.get("run_context", {}).get("artifact_identity", {})
    score_trace = Path(str(score.get("source_trace_manifest", "")))
    return (
        profile_identity.get("checkpoint_sha256") == checkpoint_sha
        and profile_identity.get("config_sha256") == config_sha
        and int(profile.get("samples", 0)) == 100
        and int(profile.get("bit_trace_records", 0)) == 12
        and profile_protocol.get("resolution") == [480, 640]
        and profile_protocol.get("crop") is None
        and profile_protocol.get("window_size") == [2, 15, 15]
        and int(profile_protocol.get("tokens_per_window", 0)) == 450
        and profile_counts.get("ATLIFTernaryPSN") == 105
        and profile_counts.get("ShiftmaxAttention") == 12
        and trace_identity.get("checkpoint_sha256") == checkpoint_sha
        and trace_identity.get("config_sha256") == config_sha
        and len(trace_records) == 12
        and {int(record.get("temporal_tokens", 0)) for record in trace_records}
        == {450}
        and all(
            Path(str(record.get("file", ""))).is_file()
            and record.get("sha256") == sha256(Path(str(record["file"])))
            for record in trace_records
        )
        and audit.get("status") == "PASS"
        and Path(str(audit.get("source_manifest", ""))).resolve()
        == H67_EP30_TRACE.resolve()
        and audit.get("coverage", {}).get("four_stage_complete") is True
        and audit.get("coverage", {}).get("stages") == [0, 1, 2, 3]
        and len(audit.get("records") or []) == 12
        and all(row.get("sha256_ok") is True for row in audit.get("records") or [])
        and score.get("status") == "PASS"
        and "component_rtl_exact" in str(score.get("scope", ""))
        and score_identity.get("checkpoint_sha256") == checkpoint_sha
        and score_identity.get("config_sha256") == config_sha
        and score_trace.resolve() == H67_EP30_TRACE.resolve()
        and score_trace.is_file()
        and score.get("source_trace_manifest_sha256") == sha256(score_trace)
        and atlif.get("status") == "PASS"
        and atlif.get("checkpoint_identity", {}).get("checkpoint_sha256")
        == checkpoint_sha
        and atlif.get("checkpoint_identity", {}).get("config_sha256")
        == config_sha
        and projection.get("status") == "PASS"
        and "projection_component_rtl_exact" in str(projection.get("scope", ""))
        and projection.get("checkpoint_identity", {}).get("checkpoint_sha256")
        == checkpoint_sha
        and projection.get("checkpoint_identity", {}).get("config_sha256")
        == config_sha
        and int(projection.get("record_count", 0)) == 12
        and projection.get("required_stage_coverage") == [0, 1, 2, 3]
        and int(projection.get("temporal_tokens", 0)) == 450
        and int(projection.get("token_id_width", 0)) == 9
    )


def local5_rtl_evidence_complete() -> bool:
    required = (
        LOCAL5_RANKING,
        LOCAL5_HARDWARE_CONFIG,
        LOCAL5_RTL,
        LOCAL5_RTL_STATUS,
    )
    if not all(path.is_file() for path in required):
        return False
    if LOCAL5_RTL_COMPLETE not in LOCAL5_RTL_STATUS.read_text(
        encoding="utf-8", errors="replace"
    ):
        return False
    rank1_epoch = None
    for line in LOCAL5_RANKING.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*1\s*\|\s*(\d+)\s*\|", line)
        if match:
            rank1_epoch = int(match.group(1))
            break
    if rank1_epoch is None:
        return False
    checkpoint = LOCAL5_RUN / f"checkpoint_epoch{rank1_epoch}.pth"
    if not checkpoint.is_file():
        return False
    try:
        result = json.loads(LOCAL5_RTL.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    identity = result.get("checkpoint_identity") or {}
    score = result.get("score_shiftmax") or {}
    projection = result.get("projection") or {}
    projection_verification = projection.get("verification") or {}
    atlif = result.get("atlif_temporal_matrix") or {}
    return (
        result.get("status") == "PASS"
        and "component_rtl_exact" in str(result.get("evidence_scope", ""))
        and "not_full_network" in str(result.get("evidence_scope", ""))
        and int(identity.get("best_epoch", -1)) == rank1_epoch
        and Path(str(identity.get("checkpoint", ""))).resolve()
        == checkpoint.resolve()
        and identity.get("checkpoint_sha256") == sha256(checkpoint)
        and Path(str(identity.get("config", ""))).resolve()
        == LOCAL5_HARDWARE_CONFIG.resolve()
        and identity.get("config_sha256") == sha256(LOCAL5_HARDWARE_CONFIG)
        and score.get("status") == "PASS"
        and all(
            projection_verification.get(name) == "PASS"
            for name in (
                "checkpoint_weight_binding",
                "random_sva",
                "verilator_lint",
                "yosys_check",
            )
        )
        and projection.get("weight_mode") == LOCAL5_PROJECTION_WEIGHT_MODE
        and atlif.get("status") == "PASS"
    )


def wait_for_gpu_queue() -> None:
    while not (
        h67_ep30_evidence_complete() and local5_rtl_evidence_complete()
    ):
        record("WAIT Local-5 RTL and H67 ep30 T450 evidence release")
        time.sleep(300)
    record(
        "RELEASE Local-5 and H67 checkpoint-bound component RTL complete; "
        "start equal +10 audit"
    )


def stage_resume(candidate: Candidate) -> None:
    for required in (candidate.config, candidate.source_model, candidate.source_state):
        if not required.is_file():
            raise FileNotFoundError(required)
    candidate.root.mkdir(parents=True, exist_ok=True)
    model = candidate.root / f"checkpoint_epoch{candidate.source_label}.pth"
    state_path = candidate.root / f"checkpoint_epoch{candidate.source_label}_state_dict.pth"
    audit_path = candidate.root / "resume_stage_audit.json"
    if audit_path.is_file() and model.is_file() and state_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        expected_config_sha = sha256(candidate.config)
        expected_metadata = {
            "config_sha256": expected_config_sha,
            "resume_protocol": (
                "audited_model_optimizer_scheduler_scaler_equal_plus10_from_fullres30"
            ),
            "resume_source_budget": 30,
            "resume_source_checkpoint_label": candidate.source_label,
        }
        checks = {
            "source model hash": audit.get("source_model_sha256") == sha256(candidate.source_model),
            "source state hash": audit.get("source_state_sha256") == sha256(candidate.source_state),
            "staged state hash": audit.get("staged_state_sha256") == sha256(state_path),
            "model hardlink": model.stat().st_ino == candidate.source_model.stat().st_ino,
            "RNG disclosure": audit.get("rng_state_present") is False,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(
                f"{candidate.name} staged resume drift: {', '.join(failed)}"
            )
        metadata_drift = [
            key
            for key, expected in expected_metadata.items()
            if key in audit and audit.get(key) != expected
        ]
        if metadata_drift:
            raise RuntimeError(
                f"{candidate.name} staged resume metadata drift: {metadata_drift}"
            )
        if any(key not in audit for key in expected_metadata):
            audit.update(expected_metadata)
            audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            record(f"UPGRADE {candidate.name} staged resume config provenance")
        return

    if model.exists():
        model.unlink()
    os.link(candidate.source_model, model)
    source_state = torch.load(candidate.source_state, map_location="cpu", weights_only=False)
    if int(source_state.get("epoch", -1)) != 29:
        raise RuntimeError(f"{candidate.name} source state epoch is not 29")
    scheduler = source_state.get("scheduler")
    if not isinstance(scheduler, dict) or int(scheduler.get("last_epoch", -1)) != 29:
        raise RuntimeError(f"{candidate.name} source scheduler is not at epoch29")
    optimizer_lrs = [float(group["lr"]) for group in source_state["optimizer"]["param_groups"]]
    if not optimizer_lrs or optimizer_lrs[0] != 2.5e-5:
        raise RuntimeError(f"{candidate.name} unexpected source LR: {optimizer_lrs}")
    milestones_before = dict(scheduler.get("milestones", {}))
    scheduler["milestones"] = Counter()
    torch.save(source_state, state_path)
    audit = {
        "candidate": candidate.name,
        "scope": "model_optimizer_scheduler_scaler_resume_not_rng_bit_exact",
        "source_model": str(candidate.source_model),
        "source_state": str(candidate.source_state),
        "config_sha256": sha256(candidate.config),
        "resume_protocol": (
            "audited_model_optimizer_scheduler_scaler_equal_plus10_from_fullres30"
        ),
        "resume_source_budget": 30,
        "resume_source_checkpoint_label": candidate.source_label,
        "source_model_sha256": sha256(candidate.source_model),
        "source_state_sha256": sha256(candidate.source_state),
        "staged_state_sha256": sha256(state_path),
        "model_hardlink": model.stat().st_ino == candidate.source_model.stat().st_ino,
        "state_epoch": int(source_state["epoch"]),
        "scheduler_last_epoch": int(scheduler["last_epoch"]),
        "scheduler_milestones_before": milestones_before,
        "scheduler_milestones_after": {},
        "optimizer_lrs_unchanged": optimizer_lrs,
        "scaler_present": bool(source_state.get("scaler")),
        "rng_state_present": any("rng" in key.lower() for key in source_state),
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if audit["rng_state_present"]:
        raise RuntimeError(f"{candidate.name} historical RNG disclosure changed")
    record(f"PASS {candidate.name} staged resume audit")


def latest_paired_checkpoint(candidate: Candidate) -> Path:
    paired: list[tuple[int, Path]] = []
    for state in candidate.root.glob("checkpoint_epoch*_state_dict.pth"):
        match = re.fullmatch(r"checkpoint_epoch(\d+)_state_dict\.pth", state.name)
        if not match:
            continue
        label = int(match.group(1))
        model = candidate.root / f"checkpoint_epoch{label}.pth"
        if model.is_file():
            paired.append((label, model))
    if not paired:
        raise RuntimeError(f"{candidate.name} has no paired resume checkpoint")
    return max(paired, key=lambda item: item[0])[1]


def verify_load_chain(candidate: Candidate) -> None:
    text = (candidate.root / "train.log").read_text(encoding="utf-8", errors="replace")
    audits = re.findall(
        r"\[H9\] load audit: checkpoint_overlay_keys=(\d+), missing=(\d+), unexpected=(\d+)",
        text,
    )
    if not audits:
        raise RuntimeError(f"{candidate.name} has no checkpoint load audit")
    overlay, missing, unexpected = (int(value) for value in audits[-1])
    if (overlay, missing, unexpected) != (candidate.expected_overlay_keys, 0, 0):
        raise RuntimeError(
            f"{candidate.name} load audit mismatch: {(overlay, missing, unexpected)}"
        )
    if candidate.expected_atlif:
        if not re.search(r"ATLIFTernaryPSN summary: \{'num_modules': 105,", text):
            raise RuntimeError(f"{candidate.name} ATLIFTernaryPSN count105 missing")
        if not re.search(r"Shiftmax attention summary: \{'num_modules': 12,", text):
            raise RuntimeError(f"{candidate.name} Shiftmax count12 missing")
    record(
        f"PASS {candidate.name} load chain: overlay={overlay} "
        f"missing={missing} unexpected={unexpected} ATLIF={candidate.expected_atlif} "
        f"Shiftmax={candidate.expected_shiftmax}"
    )


def validate_checkpoint_contract(candidate: Candidate) -> None:
    offset = 1 if candidate.name == "H67" else 0
    missing: list[str] = []
    for label in candidate.eval_labels:
        model = candidate.root / f"checkpoint_epoch{label}.pth"
        state = candidate.root / f"checkpoint_epoch{label}_state_dict.pth"
        if not model.is_file():
            missing.append(str(model))
        if not state.is_file():
            missing.append(str(state))
    if missing:
        raise RuntimeError(f"{candidate.name} checkpoint contract missing: {missing}")
    for label in candidate.eval_labels:
        state_path = candidate.root / f"checkpoint_epoch{label}_state_dict.pth"
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        expected_internal_epoch = label - offset
        if int(state.get("epoch", -1)) != expected_internal_epoch:
            raise RuntimeError(
                f"{candidate.name} label/internal epoch mismatch: "
                f"label={label} state={state.get('epoch')}"
            )
        scheduler = state.get("scheduler") or {}
        if dict(scheduler.get("milestones", {})):
            raise RuntimeError(f"{candidate.name} staged scheduler milestones reappeared")
        del state
    record(
        f"PASS {candidate.name} checkpoint contract: labels={candidate.eval_labels} "
        f"paired_states={len(candidate.eval_labels)}"
    )


def validate_eval_profiles(candidate: Candidate) -> None:
    config_sha = sha256(candidate.config)
    for label in candidate.eval_labels:
        checkpoint = candidate.root / f"checkpoint_epoch{label}.pth"
        profile = candidate.root / f"standard_valid825/epoch{label}/spike_profile.json"
        raw = json.loads(profile.read_text(encoding="utf-8"))
        audit = raw.get("checkpoint_load_audit") or {}
        counts = raw.get("module_counts") or {}
        protocol = raw.get("eval_protocol") or {}
        metric_contract = raw.get("metric_contract") or {}
        aggregation_audit = raw.get("metric_aggregation_audit") or {}
        frame_equal = aggregation_audit.get("frame_equal_mean") or {}
        pixel_global = aggregation_audit.get("pixel_global_mean") or {}
        sequence_balanced = aggregation_audit.get("sequence_balanced_mean") or {}
        metrics = raw.get("metrics") or {}
        validation_file_list = raw.get("validation_file_list") or {}
        validation_file = Path(str(validation_file_list.get("path", "")))
        identity = raw.get("artifact_identity") or {}
        checks = {
            "resolution480x640": protocol.get("resolution") == [480, 640],
            "crop null": protocol.get("crop") is None,
            "window2x15x15": protocol.get("window_size") == [2, 15, 15],
            "BN no_running": protocol.get("bn_policy") == "no_running",
            "eval batch1": protocol.get("eval_batch_size") == 1,
            "AAE-2D contract": metric_contract.get("AAE")
            == "legacy_2d_direction_angle_degrees_between_uv",
            "AE-3D contract": metric_contract.get("AAE_Benchmark")
            == "middlebury_barron_3d_angle_degrees_between_normalized_uv1",
            "DSEC Fl contract": metric_contract.get("DSEC_Fl")
            == "percentage_epe_gt_3px_and_gt_5pct_of_ground_truth_flow_magnitude",
            "frame-mean aggregation": metric_contract.get("aggregation")
            == "masked_mean_per_frame_then_equal_mean_over_validation_frames",
            "local validation population": metric_contract.get("population")
            == "local_DSEC_valid_file_list_not_official_hidden_test",
            "samples825": int(raw.get("samples", 0)) == 825,
            "aggregation schema": aggregation_audit.get("schema")
            == "flow_metric_aggregation_audit_v1",
            "aggregation frames825": int(aggregation_audit.get("frame_count", 0)) == 825,
            "aggregation sequences18": int(aggregation_audit.get("sequence_count", 0)) == 18,
            "aggregation valid pixels": float(aggregation_audit.get("valid_pixels", 0)) > 0,
            "aggregation per-sequence18": len(aggregation_audit.get("per_sequence") or {}) == 18,
            "aggregation matches production": all(
                key in metrics
                and key in frame_equal
                and abs(float(metrics[key]) - float(frame_equal[key])) <= 1.0e-5
                for key in ("AEE", "AAE", "AAE_Benchmark", "DSEC_Fl")
            ),
            "aggregation modes complete": all(
                key in pixel_global and key in sequence_balanced
                for key in ("AEE", "AAE", "AAE_Benchmark", "DSEC_Fl")
            ),
            "validation list exists": validation_file.is_file(),
            "validation list SHA": validation_file.is_file()
            and validation_file_list.get("sha256") == sha256(validation_file),
            "config path": identity.get("config_path")
            == str(candidate.config.resolve()),
            "config SHA": identity.get("config_sha256") == config_sha,
            "checkpoint path": identity.get("checkpoint_path")
            == str(checkpoint.resolve()),
            "checkpoint SHA": identity.get("checkpoint_sha256") == sha256(checkpoint),
            "overlay count": audit.get("checkpoint_overlay_keys")
            == candidate.expected_overlay_keys
            and audit.get("model_overlay_keys") == candidate.expected_overlay_keys,
            "missing0": audit.get("missing_count") == 0,
            "unexpected0": audit.get("unexpected_count") == 0,
            "ATLIF count": counts.get("ATLIFTernaryPSN")
            == candidate.expected_atlif,
            "Shiftmax count": counts.get("ShiftmaxAttention")
            == candidate.expected_shiftmax,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(
                f"{candidate.name} ep{label} eval profile audit failed: {failed}"
            )
    record(
        f"PASS {candidate.name} standard valid825 profiles: protocol/load/modules/SHA"
    )


def eval_profiles_reusable(candidate: Candidate) -> bool:
    if not (candidate.root / "profile_ranking_valid825.md").is_file():
        return False
    try:
        validate_eval_profiles(candidate)
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False
    return True


def train_and_evaluate(candidate: Candidate) -> None:
    stage_resume(candidate)
    final_model = candidate.root / f"checkpoint_epoch{candidate.final_label}.pth"
    final_state = candidate.root / f"checkpoint_epoch{candidate.final_label}_state_dict.pth"
    if not final_model.is_file() or not final_state.is_file():
        resume = latest_paired_checkpoint(candidate)
        run(
            [
                str(PYTHON),
                "-u",
                str(EXP / "entrypoints/train.py"),
                "--config",
                str(candidate.config),
                "--prev_runid",
                str(resume),
                "--save_path",
                str(candidate.root / "checkpoint_epoch{}.pth"),
                "--finetune",
                "1",
                "--resume",
                "1",
            ],
            candidate.root / "train.log",
            f"{candidate.name} equal +10 fullres convergence train",
        )
    verify_load_chain(candidate)
    validate_checkpoint_contract(candidate)
    ranking = candidate.root / "profile_ranking_valid825.md"
    if not eval_profiles_reusable(candidate):
        record(
            f"REBUILD {candidate.name} standard valid825: missing or stale profile contract"
        )
        epoch_args = [
            item for epoch in candidate.eval_labels for item in ("--epoch", str(epoch))
        ]
        run(
            [
                str(PYTHON),
                "-u",
                str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
                "--config",
                str(candidate.config),
                "--run-dir",
                str(candidate.root),
                "--ranking-mode",
                "aee",
                *epoch_args,
            ],
            candidate.root / "valid825.log",
            f"{candidate.name} equal +10 standard valid825",
        )
    validate_eval_profiles(candidate)


def load_metrics(candidate: Candidate, label: int) -> dict[str, float]:
    profile = candidate.root / f"standard_valid825/epoch{label}/spike_profile.json"
    raw = json.loads(profile.read_text(encoding="utf-8"))
    metrics = raw.get("metrics", {})
    required = ("AEE", "AAE", "AAE_Benchmark", "DSEC_Fl")
    if any(key not in metrics for key in required):
        raise RuntimeError(f"{candidate.name} ep{label} profile metrics incomplete")
    return {
        "AEE": float(metrics["AEE"]),
        "AAE": float(metrics["AAE"]),
        "AAE_Benchmark": float(metrics["AAE_Benchmark"]),
        "DSEC_Fl": float(metrics["DSEC_Fl"]),
        "total_spikes_g": float(raw["total_spikes"]) / 1e9,
    }


def relative_improvement(previous: float, current: float) -> float:
    return 100.0 * (previous - current) / previous


def relative_change(previous: float, current: float) -> float:
    return 100.0 * (current - previous) / previous


def write_convergence_summary() -> dict:
    output: dict[str, object] = {
        "schema": "dsec_fullres_equal_plus10_convergence_v1",
        "criterion": CONVERGENCE_CRITERION,
        "budgets": [30, 35, 40],
        "candidates": {},
    }
    lines = [
        "# DSEC full-resolution equal +10 convergence audit",
        "",
        "Criterion: the largest observed budget being AEE rank-1 is right-censored and "
        "therefore not plateaued; the last-five slope is descriptive only.",
        "",
        "| candidate | budget | checkpoint label | AEE | AAE-2D | AE-3D | DSEC Fl(%) | spikes(G) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    decisions: list[str] = []
    for candidate in CANDIDATES:
        points = []
        for budget, label in zip((30, 35, 40), candidate.eval_labels, strict=True):
            metrics = load_metrics(candidate, label)
            point = {"budget": budget, "checkpoint_label": label, **metrics}
            points.append(point)
            lines.append(
                f"| {candidate.name} | {budget} | {label} | {metrics['AEE']:.6f} | "
                f"{metrics['AAE']:.6f} | {metrics['AAE_Benchmark']:.6f} | "
                f"{metrics['DSEC_Fl']:.4f} | "
                f"{metrics['total_spikes_g']:.4f} |"
            )
        rank1 = min(points, key=lambda point: point["AEE"])
        aee_last5 = relative_improvement(points[-2]["AEE"], points[-1]["AEE"])
        aee_last10 = relative_improvement(points[0]["AEE"], points[-1]["AEE"])
        ae3d_last5 = relative_improvement(
            points[-2]["AAE_Benchmark"], points[-1]["AAE_Benchmark"]
        )
        ae3d_last10 = relative_improvement(
            points[0]["AAE_Benchmark"], points[-1]["AAE_Benchmark"]
        )
        aae2d_last5 = relative_improvement(points[-2]["AAE"], points[-1]["AAE"])
        aae2d_last10 = relative_improvement(points[0]["AAE"], points[-1]["AAE"])
        spikes_last5 = relative_change(
            points[-2]["total_spikes_g"], points[-1]["total_spikes_g"]
        )
        spikes_last10 = relative_change(
            points[0]["total_spikes_g"], points[-1]["total_spikes_g"]
        )
        # A rank-1 at the largest observed budget is right-censored. A small
        # last-five slope is descriptive but cannot prove an unseen next point.
        not_plateaued = rank1["budget"] == 40
        decision = "not_plateaued" if not_plateaued else "operationally_plateaued_or_overfit"
        angle_plateaued = abs(aae2d_last5) <= 1.0 and abs(ae3d_last5) <= 1.0
        angle_decision = "angle_plateaued" if angle_plateaued else "angle_not_plateaued_or_noisy"
        output["candidates"][candidate.name] = {
            "points": points,
            "rank1_budget": rank1["budget"],
            "rank1_checkpoint_label": rank1["checkpoint_label"],
            "aee_last5_improvement_pct": aee_last5,
            "aee_last10_improvement_pct": aee_last10,
            "aae2d_last5_improvement_pct": aae2d_last5,
            "aae2d_last10_improvement_pct": aae2d_last10,
            "ae3d_last5_improvement_pct": ae3d_last5,
            "ae3d_last10_improvement_pct": ae3d_last10,
            "spikes_last5_change_pct": spikes_last5,
            "spikes_last10_change_pct": spikes_last10,
            "decision": decision,
            "angle_decision": angle_decision,
        }
        decisions.append(
            f"- {candidate.name}: `{decision}`; AEE last5 `{aee_last5:.3f}%`, "
            f"last10 `{aee_last10:.3f}%`; AAE-2D last5/last10 "
            f"`{aae2d_last5:.3f}%/{aae2d_last10:.3f}%`, AE-3D last5/last10 "
            f"`{ae3d_last5:.3f}%/{ae3d_last10:.3f}%`; spikes change last5/last10 "
            f"`{spikes_last5:+.3f}%/{spikes_last10:+.3f}%`; "
            f"angle `{angle_decision}`, rank-1 budget `{rank1['budget']}`."
        )
    lines.extend(["", "## Decision", "", *decisions, ""])
    SUMMARY_JSON.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    record(f"PASS convergence summary: {SUMMARY_JSON}")
    return output


def append_results() -> None:
    marker = "DSEC_FULLRES_W15_H67_NB0_EQUAL_PLUS10_RESULT_20260805"
    if marker in REDESIGN.read_text(encoding="utf-8"):
        return
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### Local5/H67/NB0 fullres 等预算 +10 收敛审计结果\n\n")
        handle.write(f"<!-- {marker} -->\n\n")
        for candidate in CANDIDATES:
            handle.write(f"#### {candidate.name}\n\n")
            handle.write((candidate.root / "profile_ranking_valid825.md").read_text(encoding="utf-8"))
            handle.write("\n")
        handle.write(SUMMARY_MD.read_text(encoding="utf-8"))
        handle.write("\n")


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            record("EXIT another equal +10 convergence runner owns the lock")
            return 0
        run(
            [str(PYTHON), str(EXP / "entrypoints/make_dsec_fullres_w15_equal_plus10_configs.py")],
            STATUS,
            "generate equal +10 configs",
        )
        wait_for_gpu_queue()
        for candidate in CANDIDATES:
            train_and_evaluate(candidate)
        write_convergence_summary()
        append_results()
        record("ALL COMPLETE Local5/H67/NB0 equal +10 convergence audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
