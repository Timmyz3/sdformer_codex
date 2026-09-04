#!/usr/bin/env python3
"""Prepare and run the no-RTL H81 RQTB G0 ordered profile."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


HW_ROOT = Path(__file__).resolve().parents[1]
REPO = HW_ROOT.parent
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
PYTHON = Path("/opt/conda/envs/sdformerflow/bin/python")
SOURCE_CONFIG = EXP / "configs/generated/dsec_fullres_w15_H81_nomotion_bb1e4_ft40.yml"
CHECKPOINT = EXP / (
    "results/dsec_fullres_w15_H81_nomotion_bb1e4_ft40_20260811/"
    "checkpoint_epoch29.pth"
)
ROOT = HW_ROOT / "results/h81_rqtb_g0_20260816"
CONFIG = ROOT / "h81_ep29_hardware_order_pow2alpha_q7q17.yml"
PROFILE = ROOT / "profile10"
PROFILE_JSON = PROFILE / "nts11_hardware_p0_profile.json"
ARCH_JSON = ROOT / "arch_dse.json"
REPORT = ROOT / "g0_report.json"
VALID825 = ROOT / "hardware_order_valid825"
VALID825_PROFILE = VALID825 / "spike_profile.json"
VALID825_RECEIPT = ROOT / "hardware_order_valid825_receipt.json"
MVSEC_RECEIPT = ROOT / "h81_mvsec_gate_receipt.json"
FLOAT_PROFILE = (
    CHECKPOINT.parent / "standard_valid825/epoch29/spike_profile.json"
)
STATUS = ROOT / "status.log"
SAMPLES = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_empty_active(row: dict[str, object]) -> dict[str, object]:
    pairs = int(row["pairs"])
    empty = int(row["pair_empty"])
    equal = int(row["score_equal_ttx"])
    if not 0 <= empty <= equal <= pairs:
        raise ValueError(
            "empty/equal counts violate all-four-vector-empty subset contract"
        )
    nonempty = pairs - empty
    nonempty_equal = equal - empty
    return {
        "pairs": pairs,
        "empty_pairs": empty,
        "empty_ratio": empty / pairs if pairs else 0.0,
        "all_pair_equal_pairs": equal,
        "all_pair_equal_ratio": equal / pairs if pairs else 0.0,
        "nonempty_pairs": nonempty,
        "nonempty_equal_pairs": nonempty_equal,
        "nonempty_equal_ratio": (
            nonempty_equal / nonempty if nonempty else 0.0
        ),
        "slot_reduction_all_pairs": 0.5 * equal / pairs if pairs else 0.0,
        "slot_reduction_nonempty_pairs": (
            0.5 * nonempty_equal / nonempty if nonempty else 0.0
        ),
        "conservation": {
            "pairs_equal_empty_plus_nonempty": pairs == empty + nonempty,
            "equal_equal_empty_plus_nonempty_equal": (
                equal == empty + nonempty_equal
            ),
        },
    }


def apply_mvsec_gate(
    report: dict[str, object], receipt: dict[str, object]
) -> None:
    report["h81_mvsec_gate"] = receipt
    report["blocking_gates"] = [
        item
        for item in report["blocking_gates"]
        if item != "H81 MVSEC is missing"
        and not str(item).startswith("H81 MVSEC all-sequence gate failed:")
    ]
    if receipt.get("status") == "FAIL_H81_MVSEC_ALL_SEQUENCE_GATE":
        failure = ", ".join(str(item) for item in receipt["failing_sequences"])
        report["blocking_gates"].append(
            f"H81 MVSEC all-sequence gate failed: {failure}"
        )
        report["status"] = "G0_PASS_G1_BLOCKED_BY_SELECTOR_AND_MVSEC_FAIL"


def record(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def prepare() -> None:
    for path in (SOURCE_CONFIG, CHECKPOINT, PYTHON):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    attention = config["bsa_attention"]
    if attention.get("mode") != "h60" or float(
        attention.get("binary_motion_xor_alpha", -1.0)
    ) != 0.0:
        raise RuntimeError("H81 source config is not the frozen no-motion H60 model")
    source_alpha0 = float(attention["alpha0"])
    attention.update(
        {
            "alpha0": 1.0 / 64.0,
            "binary_motion_xor_alpha": 0.0,
            "hardware_quant_enabled": True,
            "hardware_mu_pow2_shift": 0,
            "hardware_score_step": 1.0 / 128.0,
            "hardware_score_min": -2.0,
            "hardware_score_max": 2.0,
            "hardware_gate_step": 1.0 / 128.0,
            "hardware_gate_min": 0.0,
            "hardware_gate_max": 2.0,
            "hardware_mask_invalid_candidates": True,
            "hardware_rtl_shiftmax_enabled": True,
        }
    )
    config["experiment"] = "h81_ep29_hardware_order_pow2alpha_q7q17_g0"
    config["test"]["bn_policy"] = "no_running"
    runtime = config.setdefault("runtime", {})
    runtime["deployment_contract"] = {
        "scope": "H81_RQTB_G0_pow2_alpha_attention_core_hardware_order",
        "source_alpha0": source_alpha0,
        "deployed_alpha0": 1.0 / 64.0,
        "alpha0_changed_for_pow2_score_front": True,
        "full_network_fixed_point": False,
        "not_algorithm_checkpoint_identity_until_valid825": True,
        "no_rtl_or_h67_performance_inheritance": True,
    }
    config["note"] = (
        "H81 ep29 no-RTL G0 deployment probe. Q7/Q1.7 hardware-order attention "
        "uses alpha0=1/64 instead of trained alpha0=0.02; workload statistics do "
        "not establish H81 algorithm identity or inherit H67 performance."
    )
    rendered = yaml.safe_dump(config, sort_keys=False, width=100)
    ROOT.mkdir(parents=True, exist_ok=True)
    if CONFIG.is_file() and CONFIG.read_text(encoding="utf-8") != rendered:
        raise RuntimeError(f"H81 G0 config drift: {CONFIG}")
    if not CONFIG.is_file():
        CONFIG.write_text(rendered, encoding="utf-8")
    identity = {
        "schema": "h81_rqtb_g0_identity_v1",
        "status": "PREPARED",
        "source_config": str(SOURCE_CONFIG.resolve()),
        "source_config_sha256": sha256(SOURCE_CONFIG),
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "deploy_config": str(CONFIG.resolve()),
        "deploy_config_sha256": sha256(CONFIG),
        "samples": SAMPLES,
        "claim_boundary": (
            "No H81 RTL, valid825, MVSEC, cycle, energy, encoder, PPA, or H67 "
            "performance inheritance."
        ),
    }
    (ROOT / "identity.json").write_text(
        json.dumps(identity, indent=2) + "\n", encoding="utf-8"
    )
    record("PREPARED H81 ep29 pow2-alpha Q7/Q1.7 G0 config")


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "SDFORMER_USE_MLFLOW": "0",
            "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
            "SDFORMER_SNN_BACKEND": "cupy",
        }
    )
    return env


def run(command: list[str], label: str) -> None:
    record(f"START {label}: {' '.join(command)}")
    result = subprocess.run(command, cwd=REPO, env=environment())
    record(f"END {label}: exit_code={result.returncode}")
    if result.returncode:
        raise RuntimeError(f"{label} failed")


def profile() -> None:
    if PROFILE_JSON.is_file():
        record("REUSE existing H81 profile10")
        return
    run(
        [
            str(PYTHON),
            "-u",
            str(EXP / "entrypoints/profile_nts11_hardware_p0.py"),
            "--config",
            str(CONFIG),
            "--checkpoint",
            str(CHECKPOINT),
            "--output-dir",
            str(PROFILE),
            "--samples",
            str(SAMPLES),
            "--num-workers",
            "0",
            "--ordered-trace",
        ],
        "H81 ordered profile10",
    )


def analyze() -> None:
    run(
        [
            str(PYTHON),
            str(EXP / "entrypoints/analyze_binary_temporal_pair_arch.py"),
            "--profile-json",
            str(PROFILE_JSON),
            "--output",
            str(ARCH_JSON),
        ],
        "H81 temporal-pair architecture analysis",
    )
    profile_data = json.loads(PROFILE_JSON.read_text(encoding="utf-8"))
    arch = json.loads(ARCH_JSON.read_text(encoding="utf-8"))
    model = arch["model_summary"]
    stage_summary = arch["stage_summary"]
    equal_ratio = float(model["score_equal_ttx_ratio"])
    slot_reduction = 0.5 * equal_ratio
    stage_equal = {
        stage: float(row["score_equal_ttx_ratio"])
        for stage, row in stage_summary.items()
    }
    empty_active_split = {
        "definition": (
            "pair_empty means Q0/Q1/K0/K1 are all zero and is therefore a "
            "strict subset of equal Q7 score pairs; nonempty_equal is "
            "score_equal_ttx - pair_empty"
        ),
        "scope": "[prof] workload statistics; not RTL cycles or innovation",
        "overall": split_empty_active(model),
        "per_stage": {
            stage: split_empty_active(row)
            for stage, row in stage_summary.items()
        },
    }
    artifact_identity = profile_data.get("artifact_identity") or {}
    checks = {
        "samples10": int(profile_data.get("samples", 0)) == SAMPLES,
        "ordered_trace": profile_data.get("ordered_trace") is True,
        "checkpoint_sha": artifact_identity.get("checkpoint_sha256")
        == sha256(CHECKPOINT),
        "config_sha": artifact_identity.get("config_sha256") == sha256(CONFIG),
        "load_missing0": int(
            (profile_data.get("checkpoint_load_audit") or {}).get("missing_count", -1)
        )
        == 0,
        "load_unexpected0": int(
            (profile_data.get("checkpoint_load_audit") or {}).get("unexpected_count", -1)
        )
        == 0,
        "all_stage_positive_equality": all(value > 0.0 for value in stage_equal.values()),
    }
    if not all(checks.values()):
        status = "FAIL_G0_IDENTITY_OR_COVERAGE"
    elif slot_reduction >= 0.10:
        status = "GO_TO_H81_G1_RTL_ONLY_IF_SELECTOR_AND_ACCURACY_PASS"
    else:
        status = "NO_GO_RQTB_SLOT_REDUCTION_BELOW_10_PERCENT"
    output = {
        "schema": "h81_rqtb_g0_report_v1",
        "status": status,
        "evidence": "[prof]+[model] 10-sample no-RTL ordered H81 deployment probe",
        "checkpoint_sha256": sha256(CHECKPOINT),
        "config_sha256": sha256(CONFIG),
        "profile_sha256": sha256(PROFILE_JSON),
        "arch_dse_sha256": sha256(ARCH_JSON),
        "pair_score_equal_ratio": equal_ratio,
        "rqtb_slot_reduction": slot_reduction,
        "stage_pair_score_equal_ratio": stage_equal,
        "empty_active_split": empty_active_split,
        "checks": checks,
        "blocking_gates": [
            "algorithm selector has not selected H81",
            "pow2-alpha hardware-order valid825 is missing",
            "H81 MVSEC is missing",
            "H81 Fixed2S/RQTB fair RTL and true INT8 Acc32 are missing",
        ],
        "innovation_boundary": (
            "Exact quotient/multiplicity/K recovery may be reused. MSSB5, Motion-XOR, "
            "temporal-polarity narrative, H67 1.1865x, VCD, and PPA may not."
        ),
    }
    REPORT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    record(f"SEALED H81 G0 status={status}")


def audited_valid825_profile(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    identity = data.get("artifact_identity") or {}
    protocol = data.get("eval_protocol") or {}
    load = data.get("checkpoint_load_audit") or {}
    expected_protocol = {
        "resolution": [480, 640],
        "crop": None,
        "window_size": [2, 15, 15],
        "remap": "v1",
        "bn_policy": "no_running",
        "eval_batch_size": 1,
    }
    checks = {
        "config_sha": identity.get("config_sha256") == sha256(CONFIG),
        "checkpoint_sha": identity.get("checkpoint_sha256") == sha256(CHECKPOINT),
        "load_missing0": int(load.get("missing_count", -1)) == 0,
        "load_unexpected0": int(load.get("unexpected_count", -1)) == 0,
        "protocol": all(protocol.get(key) == value for key, value in expected_protocol.items()),
        "deployment_contract": data.get("deployment_contract")
        == (yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {})
        .get("runtime", {})
        .get("deployment_contract"),
    }
    if not all(checks.values()):
        raise RuntimeError(f"H81 valid825 identity audit failed: {checks}")
    metrics = data.get("metrics") or {}
    return {
        "profile_sha256": sha256(path),
        "samples": int(data.get("samples", data.get("num_samples", 825))),
        "AEE": float(metrics["AEE"]),
        "AAE": float(metrics["AAE"]),
        "DSEC_Fl": float(metrics.get("DSEC_Fl", "nan")),
        "checks": checks,
    }


def valid825() -> None:
    if not VALID825_PROFILE.is_file():
        VALID825.mkdir(parents=True, exist_ok=True)
        command = [
            str(PYTHON),
            "-u",
            "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
            "--config",
            str(CONFIG),
            "--checkpoint",
            str(CHECKPOINT),
            "--path_results",
            str(VALID825),
            "--mode",
            "valid",
        ]
        record(f"START H81 hardware-order valid825: {' '.join(command)}")
        with (VALID825 / "eval.log").open("w", encoding="utf-8") as handle:
            handle.write("$ " + " ".join(command) + "\n")
            handle.flush()
            result = subprocess.run(
                command,
                cwd=REPO,
                env=environment(),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        record(f"END H81 hardware-order valid825: exit_code={result.returncode}")
        if result.returncode:
            raise RuntimeError(f"H81 valid825 failed: {VALID825 / 'eval.log'}")
    deployed = audited_valid825_profile(VALID825_PROFILE)
    float_data = json.loads(FLOAT_PROFILE.read_text(encoding="utf-8"))
    float_aee = float(float_data["metrics"]["AEE"])
    relative_aee_delta = float(deployed["AEE"]) / float_aee - 1.0
    receipt = {
        "schema": "h81_hardware_order_valid825_receipt_v1",
        "status": (
            "PASS_H81_HARDWARE_ORDER_ACCURACY_GATE"
            if relative_aee_delta <= 0.005
            else "FAIL_H81_HARDWARE_ORDER_ACCURACY_GATE"
        ),
        "evidence": "[model] full-resolution valid825 attention-core hardware-order numeric",
        "claim_boundary": (
            "Not full-network fixed point, RTL, MVSEC, cycle, energy, encoder, or PPA evidence."
        ),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "config_sha256": sha256(CONFIG),
        "float_profile_sha256": sha256(FLOAT_PROFILE),
        "float_AEE": float_aee,
        "hardware_order": deployed,
        "relative_AEE_delta": relative_aee_delta,
        "accuracy_gate": "hardware-order AEE <= float AEE * 1.005",
    }
    VALID825_RECEIPT.write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    if REPORT.is_file():
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        report["hardware_order_valid825"] = receipt
        report["blocking_gates"] = [
            item
            for item in report["blocking_gates"]
            if item != "pow2-alpha hardware-order valid825 is missing"
        ]
        if receipt["status"] == "PASS_H81_HARDWARE_ORDER_ACCURACY_GATE":
            report["status"] = (
                "G0_PASS_G1_BLOCKED_BY_SELECTOR_MVSEC_AND_FAIR_RTL"
            )
        else:
            report["status"] = "NO_GO_H81_HARDWARE_ORDER_ACCURACY"
        if MVSEC_RECEIPT.is_file():
            apply_mvsec_gate(
                report,
                json.loads(MVSEC_RECEIPT.read_text(encoding="utf-8")),
            )
        REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    record(f"SEALED H81 valid825 status={receipt['status']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "profile", "analyze", "valid825", "all"),
        default="all",
    )
    args = parser.parse_args()
    prepare()
    if args.stage in {"profile", "all"}:
        profile()
    if args.stage in {"analyze", "all"}:
        if not PROFILE_JSON.is_file():
            raise FileNotFoundError(PROFILE_JSON)
        analyze()
    if args.stage in {"valid825", "all"}:
        if not FLOAT_PROFILE.is_file():
            raise FileNotFoundError(FLOAT_PROFILE)
        valid825()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
