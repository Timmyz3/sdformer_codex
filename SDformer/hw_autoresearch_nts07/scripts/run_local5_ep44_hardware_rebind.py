#!/usr/bin/env python3
"""Rebind the frozen Local5 hardware flow to the ranked ep44 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


HW_ROOT = Path(__file__).resolve().parents[1]
REPO = HW_ROOT.parent
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
ENTRYPOINTS = EXP / "entrypoints"
sys.path.insert(0, str(ENTRYPOINTS))

from run_dsec_fullres_paper_w15_deploy_followup import (  # noqa: E402
    make_deploy_configs,
    reusable_profile,
)
from run_h60_family_deploy_eval import run_eval  # noqa: E402

import run_local5_qfsa_profile_after_fullres as profile_flow  # noqa: E402
from local5_release_receipt import (  # noqa: E402
    file_sha256,
    validate_release_receipt,
)


PYTHON = "/opt/conda/envs/sdformerflow/bin/python"
RUN = (
    EXP
    / "results/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812"
)
TRAINING_CONFIG = (
    EXP
    / "configs/generated/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50.yml"
)
RANKING = RUN / "profile_ranking_valid825.md"
CONVERGENCE = RUN / "convergence_summary.json"
EPOCH = 44
CHECKPOINT = RUN / f"checkpoint_epoch{EPOCH}.pth"
FLOAT_PROFILE = RUN / f"standard_valid825/epoch{EPOCH}/spike_profile.json"
DYADIC_OUTPUT = RUN / f"deploy_valid825/dyadic_q7q17/epoch{EPOCH}"
HARDWARE_OUTPUT = RUN / f"deploy_valid825/hardware_order_q7q17/epoch{EPOCH}"
ORIGIN_TRAINING_IDENTITY = (
    EXP
    / "results/dsec_fullres_w15_H66d_local5_bb1e4_ft30_20260805/"
    "training_config_identity.json"
)
RESUME_30_TO_40 = (
    EXP
    / "results/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus10_ep40_20260809/"
    "resume_stage_audit.json"
)
RESUME_40_TO_50 = RUN / "resume_stage_audit.json"

TAG = "local5_ep44_hardware_rebind_20260815"
OUTPUT = HW_ROOT / f"results/{TAG}_profile100"
REPLAY = HW_ROOT / f"results/{TAG}_replay"
DESCRIPTOR = HW_ROOT / f"results/{TAG}_descriptor_analysis"
ACCEPTANCE = HW_ROOT / f"results/{TAG}_acceptance"
RECEIPT = OUTPUT / "ranked_checkpoint_release_receipt.json"
RUN_IDENTITY = OUTPUT / "post_g0_run_identity.json"
STATUS = HW_ROOT / f"results/{TAG}.log"

POSTSCORE_VECTORS = HW_ROOT / f"tb_qfit/vectors/{TAG}_postscore100"
POSTSCORE_RESULTS = HW_ROOT / f"results/{TAG}_postscore_rtl"
SCORE_PROJECTION_VECTORS = HW_ROOT / f"tb_qfit/vectors/{TAG}_score_projection100"
SCORE_PROJECTION_RESULTS = HW_ROOT / f"results/{TAG}_score_projection_rtl"
SCORE_PROJECTION_BUILD = HW_ROOT / f"build_new_arch/{TAG}_score_projection"


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(command: list[str], label: str, env: dict[str, str] | None = None) -> None:
    record(f"START {label}: {' '.join(command)}")
    subprocess.run(command, cwd=HW_ROOT, env=env, check=True)
    record(f"END {label}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def verify_selection() -> dict[str, Any]:
    convergence = _load_json(CONVERGENCE)
    points = convergence.get("points") or []
    selected = next(
        (
            point
            for point in points
            if int(point.get("checkpoint_label", -1)) == EPOCH
        ),
        None,
    )
    if (
        convergence.get("status") != "PASS"
        or convergence.get("decision") != "operationally_plateaued_or_overfit"
        or int(convergence.get("rank1_checkpoint_label", -1)) != EPOCH
        or not isinstance(selected, dict)
        or selected.get("checkpoint_sha256") != file_sha256(CHECKPOINT)
        or Path(str(selected.get("profile", ""))).resolve()
        != FLOAT_PROFILE.resolve()
        or selected.get("profile_sha256") != file_sha256(FLOAT_PROFILE)
    ):
        raise RuntimeError("Local5 ep44 selection/convergence binding failed")
    rank1_line = next(
        (
            line
            for line in RANKING.read_text(encoding="utf-8").splitlines()
            if line.startswith("| 1 |")
        ),
        "",
    )
    if not rank1_line.startswith(f"| 1 | {EPOCH} |"):
        raise RuntimeError("Local5 ep44 is not ranking row 1")
    return selected


def ensure_deploy_profiles() -> tuple[Path, Path, Path, Path]:
    verify_selection()
    dyadic_config, hardware_config = make_deploy_configs("H66d", TRAINING_CONFIG)
    config = _load_yaml(hardware_config)
    attention = config.get("bsa_attention") or {}
    contract = (config.get("runtime") or {}).get("deployment_contract") or {}
    if (
        attention.get("hardware_quant_enabled") is not True
        or attention.get("hardware_rtl_shiftmax_enabled") is not True
        or attention.get("hardware_mask_invalid_candidates") is not True
        or float(attention.get("hardware_score_step", 0.0)) != 1.0 / 128.0
        or float(attention.get("hardware_gate_step", 0.0)) != 1.0 / 128.0
        or contract.get("scope") != "attention_core_hardware_order_numeric"
    ):
        raise RuntimeError("generated Local5 ep44 hardware-order config failed contract")
    reusable = reusable_profile(hardware_config, CHECKPOINT, HARDWARE_OUTPUT)
    if reusable is None:
        record("START ep44 hardware-order full valid825")
        run_eval(hardware_config, CHECKPOINT, HARDWARE_OUTPUT)
        record("END ep44 hardware-order full valid825")
        reusable = reusable_profile(hardware_config, CHECKPOINT, HARDWARE_OUTPUT)
    if reusable is None:
        raise RuntimeError("ep44 hardware-order valid825 failed provenance/load audit")
    hardware_profile = reusable[0]
    reusable = reusable_profile(dyadic_config, CHECKPOINT, DYADIC_OUTPUT)
    if reusable is None:
        record("START ep44 dyadic full valid825")
        run_eval(dyadic_config, CHECKPOINT, DYADIC_OUTPUT)
        record("END ep44 dyadic full valid825")
        reusable = reusable_profile(dyadic_config, CHECKPOINT, DYADIC_OUTPUT)
    if reusable is None:
        raise RuntimeError("ep44 dyadic valid825 failed provenance/load audit")
    return dyadic_config, reusable[0], hardware_config, hardware_profile


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise RuntimeError(f"expected YAML mapping: {path}")
    return value


def write_ranked_receipt(
    dyadic_config: Path,
    dyadic_profile: Path,
    hardware_config: Path,
    hardware_profile: Path,
) -> dict[str, Any]:
    selected = verify_selection()
    dyadic_metrics = _load_json(dyadic_profile).get("metrics") or {}
    hardware_metrics = _load_json(hardware_profile).get("metrics") or {}
    existing_uuid = None
    if RECEIPT.is_file():
        try:
            existing_uuid = validate_release_receipt(RECEIPT).get(
                "watcher_session_uuid"
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            existing_uuid = None
    receipt = {
        "schema": "local5_ranked_checkpoint_release_receipt_v1",
        "status": "PASS",
        "watcher_session_uuid": existing_uuid or str(uuid.uuid4()),
        "release_marker": "RANKED CHECKPOINT HARDWARE-ORDER VALID825 PASS",
        "selection_metric": "AEE",
        "selection_decision": "operationally_plateaued_or_overfit",
        "best_epoch": EPOCH,
        "selected_float_aee": float(selected["AEE"]),
        "selected_dyadic_aee": float(dyadic_metrics["AEE"]),
        "selected_hardware_order_aee": float(hardware_metrics["AEE"]),
        "ranking_path": str(RANKING.resolve()),
        "ranking_sha256": file_sha256(RANKING),
        "convergence_summary_path": str(CONVERGENCE.resolve()),
        "convergence_summary_sha256": file_sha256(CONVERGENCE),
        "checkpoint_path": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": file_sha256(CHECKPOINT),
        "training_config_path": str(TRAINING_CONFIG.resolve()),
        "training_config_sha256": file_sha256(TRAINING_CONFIG),
        "origin_training_identity_path": str(ORIGIN_TRAINING_IDENTITY.resolve()),
        "origin_training_identity_sha256": file_sha256(ORIGIN_TRAINING_IDENTITY),
        "resume_30_to_40_path": str(RESUME_30_TO_40.resolve()),
        "resume_30_to_40_sha256": file_sha256(RESUME_30_TO_40),
        "resume_40_to_50_path": str(RESUME_40_TO_50.resolve()),
        "resume_40_to_50_sha256": file_sha256(RESUME_40_TO_50),
        "dyadic_config_path": str(dyadic_config.resolve()),
        "dyadic_config_sha256": file_sha256(dyadic_config),
        "dyadic_profile_path": str(dyadic_profile.resolve()),
        "dyadic_profile_sha256": file_sha256(dyadic_profile),
        "config_path": str(hardware_config.resolve()),
        "config_sha256": file_sha256(hardware_config),
        "float_profile_path": str(FLOAT_PROFILE.resolve()),
        "float_profile_sha256": file_sha256(FLOAT_PROFILE),
        "hardware_profile_path": str(hardware_profile.resolve()),
        "hardware_profile_sha256": file_sha256(hardware_profile),
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    temporary = RECEIPT.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(RECEIPT)
    return validate_release_receipt(RECEIPT, file_sha256(RECEIPT))


def write_run_identity(
    hardware_config: Path,
    receipt: dict[str, Any],
    *,
    samples: int,
    ordered_groups: int,
) -> None:
    relation_rtl = HW_ROOT / "rtl_qfit/qfit_relation_transpose_leaf.sv"
    source_paths = profile_flow.source_binding_paths()
    source_paths.update(
        {
            "ranked_rebind_runner": Path(__file__).resolve(),
            "ranked_release_validator": (
                HW_ROOT / "scripts/local5_release_receipt.py"
            ),
            "ranked_profiler_adapter": (
                HW_ROOT / "scripts/profile_local5_hardware_features_ranked.py"
            ),
            "ranked_descriptor_adapter": (
                HW_ROOT
                / "scripts/analyze_ds_flm_descriptor_manifest_ranked.py"
            ),
            "training_config": TRAINING_CONFIG,
            "origin_training_identity": ORIGIN_TRAINING_IDENTITY,
            "resume_30_to_40": RESUME_30_TO_40,
            "resume_40_to_50": RESUME_40_TO_50,
            "dyadic_valid825_profile": Path(receipt["dyadic_profile_path"]),
            "convergence_summary": CONVERGENCE,
            "float_valid825_profile": FLOAT_PROFILE,
            "hardware_valid825_profile": Path(receipt["hardware_profile_path"]),
        }
    )
    value = {
        "schema": "local5_post_g0_run_identity_v3",
        "release_marker": receipt["release_marker"],
        "deploy_status": str(CONVERGENCE.resolve()),
        "release_receipt": str(RECEIPT.resolve()),
        "release_receipt_sha256": file_sha256(RECEIPT),
        "watcher_session_uuid": receipt["watcher_session_uuid"],
        "ranking": str(RANKING.resolve()),
        "ranking_sha256": file_sha256(RANKING),
        "config": str(hardware_config.resolve()),
        "config_sha256": file_sha256(hardware_config),
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": file_sha256(CHECKPOINT),
        "best_epoch": EPOCH,
        "relation_rtl": str(relation_rtl.resolve()),
        "relation_rtl_sha256": file_sha256(relation_rtl),
        "samples": samples,
        "groups_per_block_sample": ordered_groups,
        "sampling_id": profile_flow.SAMPLING_ID,
        "dataset_sampling_id": profile_flow.DATASET_SAMPLING_ID,
        "source_bindings": {
            name: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for name, path in source_paths.items()
        },
    }
    RUN_IDENTITY.parent.mkdir(parents=True, exist_ok=True)
    temporary = RUN_IDENTITY.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(RUN_IDENTITY)


def run_profile_flow(samples: int, ordered_groups: int) -> None:
    (
        dyadic_config,
        dyadic_profile,
        hardware_config,
        hardware_profile,
    ) = ensure_deploy_profiles()
    receipt = write_ranked_receipt(
        dyadic_config,
        dyadic_profile,
        hardware_config,
        hardware_profile,
    )
    write_run_identity(
        hardware_config,
        receipt,
        samples=samples,
        ordered_groups=ordered_groups,
    )
    manifest = OUTPUT / "ordered_term_manifest.json"
    reusable = False
    if manifest.is_file():
        try:
            value = _load_json(manifest)
            reusable = (
                value.get("evidence_level") == "post_g0"
                and value.get("qualification", {}).get("qualified") is True
                and value.get("run_identity_file_sha256")
                == file_sha256(RUN_IDENTITY)
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            reusable = False
    if not reusable:
        run(
            [
                PYTHON,
                "scripts/profile_local5_hardware_features_ranked.py",
                "--config",
                str(hardware_config),
                "--checkpoint",
                str(CHECKPOINT),
                "--output-dir",
                str(OUTPUT),
                "--samples",
                str(samples),
                "--num-workers",
                "0",
                "--ordered-groups-per-block-sample",
                str(ordered_groups),
                "--ordered-evidence-level",
                "post_g0",
                "--run-identity",
                str(RUN_IDENTITY),
            ],
            "ep44 post-G0 hardware feature profile",
        )
    run(
        [
            PYTHON,
            "scripts/replay_local5_frontier_trace.py",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(REPLAY),
        ],
        "ep44 ordered frontier replay",
    )
    run(
        [
            PYTHON,
            "scripts/analyze_ds_flm_descriptor_manifest_ranked.py",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(DESCRIPTOR),
        ],
        "ep44 descriptor analysis",
    )
    run(
        [
            PYTHON,
            "scripts/validate_local5_postg0_acceptance.py",
            "--manifest",
            str(manifest),
            "--replay-report",
            str(REPLAY / "report.json"),
            "--descriptor-report",
            str(DESCRIPTOR / "report.json"),
            "--run-identity",
            str(RUN_IDENTITY),
            "--output-dir",
            str(ACCEPTANCE),
        ],
        "ep44 post-G0 acceptance",
    )


def validate_acceptance() -> None:
    path = ACCEPTANCE / "acceptance.json"
    value = _load_json(path)
    if (
        value.get("accepted") is not True
        or int(value.get("samples", 0)) != 100
        or int(value.get("blocks", 0)) != 12
        or Path(str(value.get("run_identity", ""))).resolve()
        != RUN_IDENTITY.resolve()
        or value.get("run_identity_sha256") != file_sha256(RUN_IDENTITY)
        or not all((value.get("checks") or {}).values())
    ):
        raise RuntimeError("ep44 post-G0 acceptance is not fail-closed PASS")


def run_component_rtl() -> None:
    validate_acceptance()
    run(
        [
            PYTHON,
            "scripts/generate_local5_active_projection_postg0_vectors.py",
            "--input-dir",
            str(OUTPUT),
            "--output-dir",
            str(POSTSCORE_VECTORS),
            "--per-stage",
            "25",
            "--out-dim",
            "2",
            "--weight-mode",
            "checkpoint_theta_folded_dyadic_int8_head_slice",
        ],
        "ep44 real-weight post-score vectors",
    )
    env = os.environ.copy()
    env.update(
        {
            "RESULT_DIR": str(POSTSCORE_RESULTS),
            "VECTOR_DIR": str(POSTSCORE_VECTORS),
            "CHECKPOINT_WEIGHTS": "1",
        }
    )
    run(
        ["bash", "sim_new_arch/run_local5_qgasr2c_fivebank_checks.sh"],
        "ep44 post-score projection RTL/SVA",
        env=env,
    )
    run(
        [
            PYTHON,
            "scripts/generate_local5_score_projection_vectors.py",
            "--postscore-vector-dir",
            str(POSTSCORE_VECTORS),
            "--output-dir",
            str(SCORE_PROJECTION_VECTORS),
        ],
        "ep44 raw-QK score-to-projection vectors",
    )
    env = os.environ.copy()
    env.update(
        {
            "BUILD_DIR": str(SCORE_PROJECTION_BUILD),
            "RESULT_DIR": str(SCORE_PROJECTION_RESULTS),
            "VECTOR_DIR": str(SCORE_PROJECTION_VECTORS),
            "POSTSCORE_REPORT": str(POSTSCORE_RESULTS / "report.json"),
        }
    )
    run(
        ["bash", "sim_new_arch/run_local5_score_projection_checks.sh"],
        "ep44 score/Shiftmax5-to-source-owned-TCFM5-Acc32 RTL/SVA",
        env=env,
    )
    report = _load_json(SCORE_PROJECTION_RESULTS / "report.json")
    if report.get("status") != "PASS" or not all(
        (report.get("checks") or {}).values()
    ):
        raise RuntimeError("ep44 integrated score/projection RTL report failed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("deploy", "profile", "rtl", "all"),
        default="all",
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--ordered-groups", type=int, default=4)
    args = parser.parse_args()
    verify_selection()
    if args.stage in {"deploy", "profile", "all"}:
        (
            dyadic_config,
            dyadic_profile,
            hardware_config,
            hardware_profile,
        ) = ensure_deploy_profiles()
        receipt = write_ranked_receipt(
            dyadic_config,
            dyadic_profile,
            hardware_config,
            hardware_profile,
        )
        record(
            "RELEASE ep44 float/dyadic/hardware-order valid825 receipt sha="
            + file_sha256(RECEIPT)
            + " hw_profile="
            + str(hardware_profile)
        )
    if args.stage in {"profile", "all"}:
        run_profile_flow(args.samples, args.ordered_groups)
    if args.stage in {"rtl", "all"}:
        run_component_rtl()
    record(f"ALL COMPLETE Local5 ep44 hardware rebind stage={args.stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
