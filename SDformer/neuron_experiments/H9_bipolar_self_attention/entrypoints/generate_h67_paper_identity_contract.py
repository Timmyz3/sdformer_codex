#!/usr/bin/env python3
"""Freeze the algorithm-facing identity and claim boundary for H67 ep35."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
RESULTS = EXP / "results"
GEN = EXP / "configs/generated"
CHECKPOINT = RESULTS / (
    "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth"
)
TRAIN_CONFIG = GEN / "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40.yml"
DEPLOY_CONFIG = GEN / (
    "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml"
)
PROFILE = RESULTS / (
    "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/"
    "standard_valid825/epoch35/spike_profile.json"
)
HW_EVIDENCE = REPO / (
    "hw_autoresearch_nts07/results/h67_postconvergence_rank1_hardware_evidence_20260805.json"
)
PROJECTION = REPO / (
    "hw_autoresearch_nts07/results/"
    "h67_fullres_ep35_postconvergence_t450_20260805_checkpoint_projection_rtl/report.json"
)
OUTPUT = REPO / "neuron_autoresearch/H67_PAPER_IDENTITY_CONTRACT_20260813.json"
OUTPUT_MD = OUTPUT.with_suffix(".md")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def main() -> int:
    train = yaml.safe_load(TRAIN_CONFIG.read_text(encoding="utf-8"))
    deploy = yaml.safe_load(DEPLOY_CONFIG.read_text(encoding="utf-8"))
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    hw = json.loads(HW_EVIDENCE.read_text(encoding="utf-8"))
    projection = json.loads(PROJECTION.read_text(encoding="utf-8"))
    identity = profile["artifact_identity"]
    load = profile["checkpoint_load_audit"]
    counts = profile["module_counts"]
    protocol = profile["eval_protocol"]
    metrics = profile["metrics"]
    attention = deploy["bsa_attention"]
    window = deploy["swin_transformer"]["window_size"]
    checks = {
        "checkpoint_sha": identity["checkpoint_sha256"] == sha256(CHECKPOINT),
        "checkpoint_path": Path(identity["checkpoint_path"]).resolve() == CHECKPOINT.resolve(),
        "rank1_epoch35_hardware_evidence": hw.get("status") == "PASS" and hw.get("rank1_epoch") == 35,
        "real_weight_projection_ep35": (
            projection.get("status") == "PASS"
            and projection.get("weight_mode") == "checkpoint_dyadic_int8_projection_weight"
            and projection.get("checkpoint_identity", {}).get("checkpoint_sha256") == sha256(CHECKPOINT)
        ),
        "load_overlay210_missing0_unexpected0": (
            load.get("checkpoint_overlay_keys") == 210
            and load.get("missing_count") == 0
            and load.get("unexpected_count") == 0
        ),
        "module_counts": counts.get("ATLIFTernaryPSN") == 105 and counts.get("ShiftmaxAttention") == 12,
        "full_resolution": protocol.get("resolution") == [480, 640] and protocol.get("crop") is None,
        "window_T2_15_15": window == [2, 15, 15] and math.prod(window) == 450,
        "all12_attention": len(attention.get("target_blocks") or []) == 12,
        "motion_alpha": float(attention["binary_motion_xor_alpha"]) == 0.25,
        "Q7_score": float(attention["hardware_score_step"]) == 1.0 / 128.0,
        "Q1_7_gate": float(attention["hardware_gate_step"]) == 1.0 / 128.0,
        "hardware_order_shiftmax": attention.get("hardware_rtl_shiftmax_enabled") is True,
        "gated_K_not_separate_V": attention.get("value_mode") == "threshold",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"H67 paper identity contract failed: {failed}")
    payload = {
        "schema": "h67_paper_identity_contract_v1",
        "status": "PASS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "paper_checkpoint_epoch": 35,
        "model_lineage": {
            "public_operator": "SDformerFlow SDSA without Shiftmax",
            "internal_baseline": "NB0",
            "paper_model": "H67 retrained all12 H60 Motion-XOR Shiftmax gated-K attention",
            "identity_rule": "H67 is a new trained attention operator, not an algebraically equivalent implementation of public SDSA",
        },
        "symbols": {
            "T_snn": int(deploy["spiking_neuron"]["num_steps"]),
            "T_window": int(window[0]),
            "H_window": int(window[1]),
            "W_window": int(window[2]),
            "N_tokens": math.prod(window),
            "N_temporal_pairs": int(window[1]) * int(window[2]),
        },
        "operator": {
            "attention_mode": attention["mode"],
            "motion_xor_alpha": float(attention["binary_motion_xor_alpha"]),
            "score": "binary alpha-XNOR consensus plus temporal K-XOR motion term, normalized by head_dim",
            "normalization": "row-max subtraction, Q8 exp2 LUT, integer row sum, ceil-power-of-two denominator",
            "gate": "unsigned Q1.7 RNE saturated to [0,2]",
            "value": "K reused as value; threshold-mode ATLIF output, no independent V stream",
            "exact_scope": "bit-exact relative to frozen hardware-order fixed-point reference",
        },
        "metrics_valid825": {key: float(metrics[key]) for key in ("AEE", "AAE", "AAE_Benchmark", "DSEC_Fl")},
        "total_spikes_g": float(profile["total_spikes"]) / 1e9,
        "checks": checks,
        "artifacts": {
            "checkpoint": binding(CHECKPOINT),
            "training_config": binding(TRAIN_CONFIG),
            "deployment_config": binding(DEPLOY_CONFIG),
            "valid825_profile": binding(PROFILE),
            "hardware_evidence_read_only": binding(HW_EVIDENCE),
            "real_weight_projection_read_only": binding(PROJECTION),
        },
        "claim_boundary": {
            "allowed": "checkpoint-bound component RTL exact for score/SCS/Shiftmax, ATLIF temporal matrix, and real-weight projection",
            "disallowed": "full-network RTL exact, full-encoder measured speedup, or equivalence to public float SDSA",
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(
        "\n".join(
            (
                "# H67 paper identity contract",
                "",
                "Status: `PASS`; the unique paper checkpoint is full-resolution H67 `ep35`.",
                "",
                "- H67 is a retrained all12 H60 Motion-XOR Shiftmax gated-K operator, not the public SDformerFlow SDSA.",
                f"- Symbols: `T_snn={payload['symbols']['T_snn']}`, `T_w=2`, `H_w=W_w=15`, `N_tok=450`, `N_pair=225`.",
                "- Deployment: Q7 score, Q8 LUT Shiftmax, Q1.7 gate, K reused as value.",
                f"- Valid825: AEE `{payload['metrics_valid825']['AEE']:.6f}`, AAE-2D `{payload['metrics_valid825']['AAE']:.6f}`, AE-3D `{payload['metrics_valid825']['AAE_Benchmark']:.6f}`, spikes `{payload['total_spikes_g']:.4f}G`.",
                "- Hardware evidence is read-only and component-level; it includes checkpoint-bound real-weight projection but not full-network RTL exactness.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
