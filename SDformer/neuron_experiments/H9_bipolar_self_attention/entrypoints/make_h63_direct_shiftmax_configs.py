"""Generate DSEC all12 H63 symmetric-binary direct-Shiftmax configs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GEN = EXP_ROOT / "configs/generated"
BASE = GEN / "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8.yml"
MANIFEST = GEN / "h63_direct_shiftmax_manifest.json"

TTX_CHECKPOINT = (
    EXP_ROOT
    / "results/date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid"
    / "checkpoint_epoch2.pth"
)
TERNARY_CHECKPOINT = (
    EXP_ROOT
    / "results/date11full_all_ternary_atlif_tx_w720_fastlr_full30_bs8_20260616_022014_setsid"
    / "checkpoint_epoch29.pth"
)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, config: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def make_candidate(
    base: dict[str, Any], name: str, *, mode: str, groups: int | None = None,
    output_mode: str = "binary", checkpoint: Path = TTX_CHECKPOINT,
    status: str = "config_generated_not_run",
    center_output: bool = True,
) -> tuple[Path, dict[str, Any]]:
    cfg = deepcopy(base)
    cfg["experiment"] = name

    neuron = cfg["atlif_ternary_psn"]
    threshold_mode = "symmetric_binary_abs" if output_mode == "binary" else "symmetric_bsa_tsn"
    neuron.update(
        {
            "output_mode": output_mode,
            "threshold_mode": threshold_mode,
            "center_mode": "zero",
            "negative_threshold_scale": 1.0,
            "target_rate": None,
            "target_rate_eta": 0.0,
            "activity_eta": 0.0,
            "threshold_eta": 0.0,
        }
    )
    for group in neuron.get("target_groups", []):
        group.update(
            {
                "output_mode": output_mode,
                "threshold_mode": threshold_mode,
                "center_mode": "zero",
                "negative_threshold_scale": 1.0,
                "target_rate": None,
                "target_rate_eta": 0.0,
                "activity_eta": 0.0,
                "threshold_eta": 0.0,
            }
        )

    attention = cfg["bsa_attention"]
    attention.update(
        {
            "mode": mode,
            "direct_shiftmax_groups": int(groups or 1),
            "direct_shiftmax_center_output": bool(center_output),
            "direct_shiftmax_signed_events": output_mode == "ternary",
            "alpha0": 1.0 / 64.0,
            "mismatch_penalty": 0.0,
            "single_active_penalty": 0.0,
            "bipolar_mu": 0.0,
            "k_magnitude_alpha": 0.0,
            "sc_mu_schedule_enabled": False,
            "target_rate": None,
            "hardware_quant_enabled": False,
        }
    )

    cfg["loader"]["n_epochs"] = 1
    cfg["loader"]["batch_size"] = 8
    cfg["runtime"]["max_train_steps"] = 120
    cfg["runtime"]["skip_state_save"] = False
    cfg["runtime"]["force_save_epochs"] = [0]
    cfg["runtime"]["use_mlflow_model_logging"] = False
    if output_mode == "ternary":
        cfg["optimizer"]["scheduler"] = "none"
        cfg["optimizer"]["lr_warmup"] = {"enabled": False, "steps": 0, "start_factor": 1.0}
        cfg["optimizer"]["param_groups"].update(
            {
                "backbone_lr": 5.0e-6,
                "norm_lr": 1.0e-6,
                "neuron_lr": 1.0e-5,
                "threshold_lr": 3.0e-6,
            }
        )
    cfg["note"] = (
        f"H63 symmetric-{output_mode} ATLIF + all12 {mode} G={groups or 'token-channel'}. "
        "Pure binary TX 64:1; no gate*K, K/value carrier, SC, Kmag, target-rate, or partial replacement."
    )

    path = GEN / f"{name}.yml"
    write_yaml(path, cfg)
    return path, {
        "name": name,
        "config": str(path),
        "checkpoint": str(checkpoint),
        "scope": "full-network all12",
        "atlif_expected": 105,
        "atlif_threshold_mode": threshold_mode,
        "attention_expected": 12,
        "attention_mode": mode,
        "groups": groups,
        "centered_direct_output": bool(center_output),
        "signed_event_score": output_mode == "ternary",
        "uses_gate_k": False,
        "status": status,
    }


def main() -> int:
    if not BASE.is_file():
        raise FileNotFoundError(BASE)
    if not TTX_CHECKPOINT.is_file():
        raise FileNotFoundError(TTX_CHECKPOINT)
    if not TERNARY_CHECKPOINT.is_file():
        raise FileNotFoundError(TERNARY_CHECKPOINT)
    base = load_yaml(BASE)
    rows = []
    specs = (
        ("g1_symbin", "tx_direct_group_shiftmax", 1, "binary", TTX_CHECKPOINT, "stopped_activity_gate", True),
        ("stc_symbin", "tx_direct_token_channel_shiftmax", None, "binary", TTX_CHECKPOINT, "stopped_activity_gate", True),
        ("g4_symbin", "tx_direct_group_shiftmax", 4, "binary", TTX_CHECKPOINT, "stopped_activity_gate", True),
        ("stc_symtern", "tx_direct_token_channel_shiftmax", None, "ternary", TERNARY_CHECKPOINT, "stopped_activity_gate", True),
        ("g4_symtern", "tx_direct_group_shiftmax", 4, "ternary", TERNARY_CHECKPOINT, "held_after_stc_failure", True),
        ("stc_raw_symtern", "tx_direct_token_channel_shiftmax", None, "ternary", TERNARY_CHECKPOINT, "stopped_activity_gate_zero_shot", False),
    )
    for suffix, mode, groups, output_mode, checkpoint, status, center_output in specs:
        path, row = make_candidate(
            base,
            f"h63_direct_shiftmax_{suffix}_s120",
            mode=mode,
            groups=groups,
            output_mode=output_mode,
            checkpoint=checkpoint,
            status=status,
            center_output=center_output,
        )
        rows.append(row)
        print(path)
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
