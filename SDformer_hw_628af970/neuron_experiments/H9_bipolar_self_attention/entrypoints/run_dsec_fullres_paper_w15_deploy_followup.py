"""Run full-resolution dyadic and hardware-order deploy evaluation.

The training queue remains float/AMP. This follower selects each full-resolution
float winner, evaluates Q7 score + Q1.7 gate deployment, then evaluates the
integer/LUT Shiftmax hardware-order model. Local-5 additionally enables the
true masked-candidate contract used by the corrected Local-5 RTL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from run_h60_family_deploy_eval import parse_profile, run_eval


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
OUT_ROOT = REPO / "neuron_autoresearch/experiments/dsec_fullres_paper_w15"
STATUS = RESULTS / "dsec_fullres_paper_w15_deploy_followup_status.log"
MAIN_STATUS = RESULTS / "dsec_fullres_paper_w15_queue_status.log"
RUN_TAG = "20260728"
NB0_SOURCE_CONFIG = GEN / "dsec_fullres_paper_w15_nb0_ep59_ft30.yml"
NB0_RUN_NAME = "dsec_fullres_paper_w15_nb0_ep59_ft30"

CASES = {
    "H67": {
        "label": "H67 Motion-XOR",
        "source_config": GEN / "dsec_fullres_paper_w15_h67_motion_ep19_ft30.yml",
        "run_name": "dsec_fullres_paper_w15_h67_motion_ep19_ft30",
        "true_mask": False,
        "rtl_scope": (
            "hardware-order numeric exact; existing H67 SV row RTL is verified at "
            "window9/T162, while fullres window15/T450 controller parameterization "
            "still requires RTL regression"
        ),
    },
    "H66d": {
        "label": "H66d Local-5",
        "source_config": GEN / "dsec_fullres_paper_w15_h66d_local5_ep29_ft30.yml",
        "run_name": "dsec_fullres_paper_w15_h66d_local5_ep29_ft30",
        "true_mask": True,
        "rtl_scope": (
            "score/gate hardware-order numeric exact with true masked candidates; "
            "fullres window15 line-buffer/address-control SV replay remains a separate "
            "hardware sign-off item"
        ),
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def best_epoch(ranking: Path) -> int:
    for line in ranking.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*1\s*\|\s*(\d+)\s*\|", line)
        if match:
            return int(match.group(1))
    raise RuntimeError(f"cannot parse rank-1 epoch from {ranking}")


def make_deploy_configs(model_id: str, source: Path) -> tuple[Path, Path]:
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    common = {
        "alpha0": 1.0 / 64.0,
        "castling_matrix_aux_weight": 0.0,
        "castling_matrix_aux_end_step": 0,
        "hardware_quant_enabled": True,
        "hardware_mu_pow2_shift": 0,
        "hardware_score_step": 1.0 / 128.0,
        "hardware_score_min": -2.0,
        "hardware_score_max": 2.0,
        "hardware_gate_step": 1.0 / 128.0,
        "hardware_gate_min": 0.0,
        "hardware_gate_max": 2.0,
    }
    if model_id == "H66d":
        common["binary_motion_xor_alpha"] = 0.0
        common["hardware_mask_invalid_candidates"] = True

    dyadic = deepcopy(config)
    dyadic["experiment"] = source.stem + "_dyadic_q7q17_deploy"
    dyadic["bsa_attention"].update(common)
    dyadic["bsa_attention"]["hardware_rtl_shiftmax_enabled"] = False
    dyadic.setdefault("runtime", {})["deployment_contract"] = {
        "scope": "attention_core_numeric",
        "score_quantization": "Q7_step_2^-7",
        "shiftmax": "float_exp2",
        "gate_quantization": "Q1.7_RNE",
        "full_network_fixed_point": False,
        "window15_t450_sv_regression": False,
    }
    dyadic["note"] = (
        f"{model_id} paper-window15 dyadic deploy: Q7 score, float 2^x Shiftmax, "
        "Q1.7 gate, no-running-stat BN evaluation."
    )
    dyadic_path = GEN / f"{dyadic['experiment']}.yml"
    dyadic_path.write_text(
        yaml.safe_dump(dyadic, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    hardware = deepcopy(dyadic)
    hardware["experiment"] = source.stem + "_hardware_order_q7q17_deploy"
    hardware["bsa_attention"]["hardware_rtl_shiftmax_enabled"] = True
    hardware.setdefault("runtime", {})["deployment_contract"] = {
        "scope": "attention_core_hardware_order_numeric",
        "score_quantization": "Q7_step_2^-7",
        "shiftmax": "Q8_LUT_integer_rowsum_ceil_pow2",
        "gate_quantization": "Q1.7_RNE",
        "invalid_candidate_mask": bool(model_id == "H66d"),
        "full_network_fixed_point": False,
        "window15_t450_sv_regression": False,
    }
    hardware["note"] = (
        f"{model_id} paper-window15 hardware-order numeric deploy: Q7 score, "
        "16-entry Q8 exp2 LUT, integer row sum, ceil-pow2 normalization, Q1.7 "
        "RNE gate. See summary rtl_scope before calling this full RTL exact."
    )
    hardware_path = GEN / f"{hardware['experiment']}.yml"
    hardware_path.write_text(
        yaml.safe_dump(hardware, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return dyadic_path, hardware_path


def protocol_from_profile(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    protocol = data.get("eval_protocol", {})
    expected = {
        "resolution": [480, 640],
        "crop": None,
        "window_size": [2, 15, 15],
        "remap": "v1",
        "bn_policy": "no_running",
        "eval_batch_size": 1,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise RuntimeError(
                f"{path}: protocol mismatch {key}={protocol.get(key)!r}, expected {value!r}"
            )
    return protocol


def reusable_profile(
    config: Path,
    checkpoint: Path,
    output: Path,
) -> tuple[Path, dict[str, Any]] | None:
    """Reuse only a fully audited profile produced from these exact artifacts."""
    profile = output / "spike_profile.json"
    if not profile.is_file():
        return None
    try:
        raw = json.loads(profile.read_text(encoding="utf-8"))
        protocol_from_profile(profile)
        identity = raw["artifact_identity"]
        stat = checkpoint.stat()
        expected_contract = (
            yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        ).get("runtime", {}).get("deployment_contract")
        config_data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        h9_enabled = bool(
            config_data.get("atlif_ternary_psn", {}).get("enabled")
            or config_data.get("bsa_attention", {}).get("enabled")
        )
        expected = {
            "config_path": str(config.resolve()),
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_size": stat.st_size,
            "checkpoint_mtime_ns": stat.st_mtime_ns,
            "checkpoint_sha256": file_sha256(checkpoint),
        }
        if any(identity.get(key) != value for key, value in expected.items()):
            return None
        if raw.get("deployment_contract") != expected_contract:
            return None
        load_audit = raw.get("checkpoint_load_audit")
        if not isinstance(load_audit, dict):
            return None
        if load_audit.get("missing_count") != 0:
            return None
        if load_audit.get("unexpected_count") != 0:
            return None
        if h9_enabled:
            counts = raw.get("module_counts")
            if not isinstance(counts, dict):
                return None
            if (
                load_audit.get("checkpoint_overlay_keys") != 210
                or load_audit.get("model_overlay_keys") != 210
                or counts.get("ATLIFTernaryPSN") != 105
                or counts.get("ShiftmaxAttention") != 12
            ):
                return None
        return profile, parse_profile(profile)
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError):
        return None


def run_or_reuse_eval(
    label: str,
    config: Path,
    checkpoint: Path,
    output: Path,
) -> tuple[Path, dict[str, Any]]:
    reusable = reusable_profile(config, checkpoint, output)
    if reusable is not None:
        profile, metrics = reusable
        record(f"REUSE {label} audited profile AEE={metrics['AEE']:.6f}")
        return profile, metrics
    target = output
    if (output / "spike_profile.json").is_file():
        stat = checkpoint.stat()
        fingerprint = hashlib.sha256(
            config.read_bytes()
            + str(checkpoint.resolve()).encode("utf-8")
            + str(stat.st_size).encode("ascii")
            + str(stat.st_mtime_ns).encode("ascii")
        ).hexdigest()[:12]
        target = output.parent / f"{output.name}_audited_{fingerprint}"
        reusable = reusable_profile(config, checkpoint, target)
        if reusable is not None:
            profile, metrics = reusable
            record(
                f"REUSE {label} audited replacement AEE={metrics['AEE']:.6f}"
            )
            return profile, metrics
        record(
            f"STALE {label} profile retained at {output}; "
            f"writing audited result to {target}"
        )
    record(f"START {label}")
    run_eval(config, checkpoint, target)
    profile = target / "spike_profile.json"
    if reusable_profile(config, checkpoint, target) is None:
        raise RuntimeError(f"new deploy profile failed provenance/load audit: {profile}")
    protocol_from_profile(profile)
    metrics = parse_profile(profile)
    record(f"END {label} AEE={metrics['AEE']:.6f}")
    return profile, metrics


def evaluate_nb0_reference(batch_size: int) -> dict[str, Any]:
    """Replay the retained NB0 winner with explicit protocol/load provenance."""
    run_dir = RESULTS / f"{NB0_RUN_NAME}_bs{batch_size}_{RUN_TAG}"
    checkpoint = run_dir / "checkpoint_epoch29.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = run_dir / "paper_valid825_b1" / "epoch29"
    profile, metrics = run_or_reuse_eval(
        "NB0 audited paper-protocol replay epoch29",
        NB0_SOURCE_CONFIG,
        checkpoint,
        output,
    )
    return {
        "model_id": "NB0",
        "label": "NB0 baseline",
        "epoch": 29,
        "checkpoint": str(checkpoint),
        "config": str(NB0_SOURCE_CONFIG),
        "profile": str(profile),
        "metrics": metrics,
        "note": (
            "Audited replay with explicit eval_batch_size, checkpoint-load audit, "
            "and artifact identity. The historical standard_valid825 evaluator also "
            "forced batch size 1, but did not serialize those provenance fields."
        ),
    }


def evaluate_case(model_id: str, batch_size: int) -> dict[str, Any]:
    case = CASES[model_id]
    source = Path(case["source_config"])
    run_dir = RESULTS / f"{case['run_name']}_bs{batch_size}_{RUN_TAG}"
    ranking = run_dir / "profile_ranking_valid825.md"
    if not ranking.is_file():
        raise FileNotFoundError(ranking)
    epoch = best_epoch(ranking)
    checkpoint = run_dir / f"checkpoint_epoch{epoch}.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    float_profile = (
        run_dir / "standard_valid825" / f"epoch{epoch}" / "spike_profile.json"
    )
    protocol_from_profile(float_profile)
    float_metrics = parse_profile(float_profile)

    dyadic_config, hardware_config = make_deploy_configs(model_id, source)
    dyadic_out = run_dir / "deploy_valid825" / "dyadic_q7q17" / f"epoch{epoch}"
    hardware_out = run_dir / "deploy_valid825" / "hardware_order_q7q17" / f"epoch{epoch}"

    dyadic_profile, dyadic = run_or_reuse_eval(
        f"{model_id} dyadic fullres valid825 epoch{epoch}",
        dyadic_config,
        checkpoint,
        dyadic_out,
    )
    hardware_profile, hardware = run_or_reuse_eval(
        f"{model_id} hardware-order fullres valid825 epoch{epoch}",
        hardware_config,
        checkpoint,
        hardware_out,
    )

    return {
        "model_id": model_id,
        "label": case["label"],
        "epoch": epoch,
        "checkpoint": str(checkpoint),
        "float_profile": str(float_profile),
        "dyadic_config": str(dyadic_config),
        "hardware_order_config": str(hardware_config),
        "dyadic_profile": str(dyadic_profile),
        "hardware_order_profile": str(hardware_profile),
        "true_mask_invalid_candidates": bool(case["true_mask"]),
        "rtl_scope": case["rtl_scope"],
        "float": float_metrics,
        "dyadic": dyadic,
        "hardware_order": hardware,
        "delta_dyadic_minus_float": {
            "AEE": dyadic["AEE"] - float_metrics["AEE"],
            "AAE": dyadic["AAE"] - float_metrics["AAE"],
            "total_spikes_g": dyadic["total_spikes_g"]
            - float_metrics["total_spikes_g"],
        },
        "delta_hardware_minus_dyadic": {
            "AEE": hardware["AEE"] - dyadic["AEE"],
            "AAE": hardware["AAE"] - dyadic["AAE"],
            "total_spikes_g": hardware["total_spikes_g"] - dyadic["total_spikes_g"],
        },
        "delta_hardware_minus_float": {
            "AEE": hardware["AEE"] - float_metrics["AEE"],
            "AAE": hardware["AAE"] - float_metrics["AAE"],
            "total_spikes_g": hardware["total_spikes_g"]
            - float_metrics["total_spikes_g"],
        },
    }


def write_summary(rows: list[dict[str, Any]], nb0_reference: dict[str, Any]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUT_ROOT / "fullres_w15_deploy_summary.json"
    md_path = OUT_ROOT / "fullres_w15_deploy_summary.md"
    json_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "protocol": "480x640_window2x15x15_bn_no_running_eval_batch1_valid825",
                "nb0_reference": nb0_reference,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# DSEC Fullres Window15 Deploy Evaluation",
        "",
        "| baseline | epoch | AEE | AAE legacy | AAE benchmark | spikes(G) |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| {nb0_reference['label']} | {nb0_reference['epoch']} | "
            f"{nb0_reference['metrics']['AEE']:.4f} | "
            f"{nb0_reference['metrics']['AAE']:.4f} | "
            f"{nb0_reference['metrics']['AAE_Benchmark']:.4f} | "
            f"{nb0_reference['metrics']['total_spikes_g']:.4f} |"
        ),
        "",
        "| candidate | epoch | float AEE | dyadic AEE | hardware-order AEE | "
        "hardware-float delta | "
        "hardware-order AAE legacy | AAE benchmark | spikes(G) | true mask |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['epoch']} | {row['float']['AEE']:.4f} | "
            f"{row['dyadic']['AEE']:.4f} | "
            f"{row['hardware_order']['AEE']:.4f} | "
            f"{row['delta_hardware_minus_float']['AEE']:+.4f} | "
            f"{row['hardware_order']['AAE']:.4f} | "
            f"{row['hardware_order']['AAE_Benchmark']:.4f} | "
            f"{row['hardware_order']['total_spikes_g']:.4f} | "
            f"{row['true_mask_invalid_candidates']} |"
        )
    lines.extend(
        [
            "",
            "The hardware-order column is the frozen integer/LUT numerical path. "
            "Fullres SV sign-off additionally requires window15/T450 controller, "
            "address, line-buffer, and ordered-trace regression.",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- {row['label']}: {row['rtl_scope']}.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    marker = "DSEC_FULLRES_W15_DEPLOY_FOLLOWUP_RESULTS"
    text = REDESIGN.read_text(encoding="utf-8")
    if marker not in text:
        with REDESIGN.open("a", encoding="utf-8") as handle:
            handle.write("\n\n### DSEC fullres window15 定点/硬件顺序评估\n\n")
            handle.write(f"<!-- {marker} -->\n")
            handle.write(f"- summary：`{md_path.relative_to(REPO)}`\n")
            handle.write(
                "- 口径：Q7 score、Q1.7 gate、16-entry Q8 exp2 LUT、integer "
                "row sum、ceil-pow2 normalize、RNE；Local-5 使用真正 masked "
                "candidate 合同。\n"
            )
            handle.write(
                "- 命名边界：该表先关闭 fullres valid825 数值精度；window15/T450 "
                "SV 控制、地址、line-buffer 与 ordered trace 仍须硬件侧独立签核。\n\n"
            )
            for line in lines[2:]:
                handle.write(line + "\n")


def wait_for_main_queue(timeout_hours: float, poll_seconds: int) -> None:
    deadline = time.time() + timeout_hours * 3600.0
    marker = "ALL COMPLETE DSEC PAPER-W15 QUEUE"
    while time.time() < deadline:
        text = MAIN_STATUS.read_text(encoding="utf-8", errors="ignore") if MAIN_STATUS.is_file() else ""
        if marker in text:
            record("Main paper-window15 queue complete; deploy follower released")
            return
        record("WAIT main paper-window15 queue")
        time.sleep(max(30, poll_seconds))
    raise TimeoutError(f"timed out waiting for {marker}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=2)
    parser.add_argument("--wait-main-queue", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=240.0)
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()

    if args.wait_main_queue:
        wait_for_main_queue(args.timeout_hours, args.poll_seconds)
    nb0_reference = evaluate_nb0_reference(args.batch_size)
    rows = [evaluate_case(model_id, args.batch_size) for model_id in args.ids]
    write_summary(rows, nb0_reference)
    record(f"ALL COMPLETE fullres deploy followup ids={args.ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
