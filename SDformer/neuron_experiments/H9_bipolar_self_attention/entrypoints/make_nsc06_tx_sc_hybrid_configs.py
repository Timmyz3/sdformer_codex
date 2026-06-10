"""Generate NSC-06 TX+SC hybrid attention screening configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = EXP_ROOT / "configs" / "generated"
BASE = GENERATED_DIR / "ntx04h_cptc_ntx01_warm.yml"

S0_BLOCKS = ["0:0", "0:1"]
S1_BLOCKS = ["1:0", "1:1"]
S2_BLOCKS = ["2:0", "2:1", "2:2", "2:3", "2:4", "2:5"]
S3_BLOCKS = ["3:0", "3:1"]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def blocks_for(scope: str) -> list[str]:
    if scope == "all":
        return []
    if scope == "s2":
        return S2_BLOCKS
    if scope == "s02":
        return S0_BLOCKS + S2_BLOCKS
    if scope == "s012":
        return S0_BLOCKS + S1_BLOCKS + S2_BLOCKS
    if scope == "s23":
        return S2_BLOCKS + S3_BLOCKS
    raise ValueError(f"unknown scope: {scope}")


def set_common(cfg: dict[str, Any], variant: dict[str, Any]) -> None:
    cfg["experiment"] = variant["name"]
    cfg.setdefault("test", {})["scale_factor"] = 1

    loss = cfg.setdefault("loss", {})
    loss["lambda_ang"] = float(variant.get("lambda_ang", 0.0))
    loss["use_angular_loss"] = bool(loss["lambda_ang"])

    attn = cfg.setdefault("bsa_attention", {})
    attn["enabled"] = True
    attn["mode"] = str(variant.get("mode", "tx_sc_residual_selector_shiftmax"))
    attn["alpha0"] = float(variant.get("alpha0", 0.02))
    attn["mismatch_penalty"] = float(variant.get("mismatch_penalty", 0.25))
    attn["single_active_penalty"] = float(variant.get("single_active_penalty", 0.0))
    attn["single_active_penalty_grad"] = str(variant.get("single_active_penalty_grad", "ste"))
    attn["score_scale"] = float(variant.get("score_scale", 1.0))
    attn["consensus_score_norm"] = str(variant.get("consensus_score_norm", "head_dim"))
    attn["center_scores"] = bool(variant.get("center_scores", True))
    attn["preserve_mean"] = bool(variant.get("preserve_mean", True))
    attn["bipolar_mu"] = float(variant.get("bipolar_mu", 0.15))
    attn["bipolar_lambda"] = float(variant.get("bipolar_lambda", 0.35))
    attn["deadzone_epsilon"] = float(variant.get("deadzone_epsilon", 0.0))
    attn["confidence_enabled"] = bool(variant.get("confidence_enabled", False))
    attn["k_consistency_mod"] = bool(variant.get("k_consistency_mod", False))
    attn["bipolar_gate_min"] = variant.get("bipolar_gate_min", None)
    attn["bipolar_gate_max"] = variant.get("bipolar_gate_max", None)
    attn["residual_alpha"] = float(variant.get("residual_alpha", attn.get("residual_alpha", 1.0)))
    scope = str(variant.get("scope", "all"))
    target_blocks = blocks_for(scope)
    if target_blocks:
        attn["target_blocks"] = target_blocks
    else:
        attn.pop("target_blocks", None)
        attn["stage_selection"] = "all"

    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif["target_rate"] = variant.get("target_rate", atlif.get("target_rate"))
    atlif["target_rate_eta"] = float(variant.get("target_rate_eta", atlif.get("target_rate_eta", 0.0) or 0.0))
    atlif["activity_eta"] = float(variant.get("activity_eta", atlif.get("activity_eta", 0.0) or 0.0))

    opt = cfg.setdefault("optimizer", {})
    groups = opt.setdefault("param_groups", {})
    groups["enabled"] = True
    groups["backbone_lr"] = float(variant.get("backbone_lr", groups.get("backbone_lr", 1.0e-6)))
    groups["norm_lr"] = float(variant.get("norm_lr", groups.get("norm_lr", 1.0e-6)))
    groups["neuron_lr"] = float(variant.get("neuron_lr", groups.get("neuron_lr", 3.0e-5)))
    groups["threshold_lr"] = float(variant.get("threshold_lr", groups.get("threshold_lr", 5.0e-6)))
    opt["lr_warmup"] = {
        "enabled": True,
        "steps": int(variant.get("warmup_steps", 450)),
        "start_factor": float(variant.get("warmup_start_factor", 0.05)),
    }
    opt["milestones"] = list(variant.get("milestones", [22, 27]))

    cfg["note"] = (
        "NSC-06 TX+SC hybrid. "
        f"name={variant['name']}; mode={attn['mode']}; scope={scope}; "
        f"mu={attn['bipolar_mu']}; lambda={attn['bipolar_lambda']}; "
        f"mismatch={attn['mismatch_penalty']}; single={attn['single_active_penalty']}; "
        f"deadzone={attn['deadzone_epsilon']}; confidence={attn['confidence_enabled']}; "
        f"kmod={attn['k_consistency_mod']}; target_rate={atlif['target_rate']}; "
        f"lr(backbone/neuron/threshold)={groups['backbone_lr']}/{groups['neuron_lr']}/{groups['threshold_lr']}; "
        f"warmup={opt['lr_warmup']['steps']}; lambda_ang={loss['lambda_ang']}."
    )


def main() -> int:
    base = read_yaml(BASE)
    variants: list[dict[str, Any]] = [
        {
            "name": "nsc06a_h57_tx_control_all_mu0",
            "bipolar_mu": 0.0,
            "bipolar_lambda": 0.35,
            "scope": "all",
        },
        {
            "name": "nsc06b_h57_all_mu010_l03",
            "bipolar_mu": 0.10,
            "bipolar_lambda": 0.30,
            "scope": "all",
        },
        {
            "name": "nsc06c_h57_all_mu020_l03",
            "bipolar_mu": 0.20,
            "bipolar_lambda": 0.30,
            "scope": "all",
            "warmup_steps": 600,
        },
        {
            "name": "nsc06d_h57_s2_mu020_l03",
            "bipolar_mu": 0.20,
            "bipolar_lambda": 0.30,
            "scope": "s2",
            "warmup_steps": 600,
        },
        {
            "name": "nsc06e_h57_s02_mu015_l03",
            "bipolar_mu": 0.15,
            "bipolar_lambda": 0.30,
            "scope": "s02",
        },
        {
            "name": "nsc06f_h57_s012_mu015_l04",
            "bipolar_mu": 0.15,
            "bipolar_lambda": 0.40,
            "scope": "s012",
        },
        {
            "name": "nsc06g_h57_s2_conf_mu020_l03",
            "bipolar_mu": 0.20,
            "bipolar_lambda": 0.30,
            "scope": "s2",
            "deadzone_epsilon": 1.0 / 64.0,
            "confidence_enabled": True,
            "warmup_steps": 600,
        },
        {
            "name": "nsc06h_h56r_s2_alpha025_l03",
            "mode": "sc_agree_disagree_residual_shiftmax",
            "residual_alpha": 0.25,
            "bipolar_lambda": 0.30,
            "scope": "s2",
            "lambda_ang": 0.01,
        },
        {
            "name": "nsc06i_h57_s23_mu010_l03",
            "bipolar_mu": 0.10,
            "bipolar_lambda": 0.30,
            "scope": "s23",
        },
    ]

    generated: list[str] = []
    for variant in variants:
        cfg = deepcopy(base)
        set_common(cfg, variant)

        short_cfg = deepcopy(cfg)
        short_cfg["loader"]["n_epochs"] = 1
        short_cfg["runtime"]["max_train_steps"] = 360
        short_cfg["runtime"]["force_save_epochs"] = [0]
        short_cfg["runtime"]["skip_state_save"] = True
        short = GENERATED_DIR / f"{variant['name']}_steps360.yml"
        write_yaml(short, short_cfg)
        generated.append(f"generated/{short.name}")

        full_cfg = deepcopy(cfg)
        full_cfg["loader"]["n_epochs"] = 30
        full_cfg["runtime"]["max_train_steps"] = 0
        full_cfg["runtime"]["force_save_epochs"] = list(range(30))
        full_cfg["runtime"]["skip_state_save"] = True
        full = GENERATED_DIR / f"{variant['name']}_full30.yml"
        write_yaml(full, full_cfg)
        generated.append(f"generated/{full.name}")

    print("\n".join(generated))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
