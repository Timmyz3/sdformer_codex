"""Generate NSC-08 H58 late/annealed SC residual screening configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
BASE = GENERATED / "ntx04h_cptc_ntx01_warm.yml"

S1_BLOCKS = ["1:0", "1:1"]
S2_BLOCKS = ["2:0", "2:1", "2:2", "2:3", "2:4", "2:5"]


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
    if scope == "s12":
        return S1_BLOCKS + S2_BLOCKS
    raise ValueError(f"unknown scope: {scope}")


def set_common(cfg: dict[str, Any], variant: dict[str, Any]) -> None:
    cfg["experiment"] = variant["name"]
    cfg.setdefault("test", {})["scale_factor"] = 1

    attn = cfg.setdefault("bsa_attention", {})
    attn["enabled"] = True
    attn["mode"] = "tx_sc_late_residual_selector_shiftmax"
    attn["alpha0"] = 0.02
    attn["mismatch_penalty"] = 0.25
    attn["single_active_penalty"] = 0.0
    attn["single_active_penalty_grad"] = "ste"
    attn["score_scale"] = 1.0
    attn["consensus_score_norm"] = "head_dim"
    attn["center_scores"] = True
    attn["preserve_mean"] = True
    attn["bipolar_mu"] = float(variant["mu"])
    attn["bipolar_lambda"] = float(variant["lam"])
    attn["confidence_enabled"] = bool(variant.get("confidence", False))
    attn["k_consistency_mod"] = bool(variant.get("kmod", False))
    attn["deadzone_epsilon"] = float(variant.get("deadzone", 0.0))
    attn["bipolar_gate_min"] = None
    attn["bipolar_gate_max"] = None
    attn["sc_mu_schedule_enabled"] = True
    attn["sc_mu_start"] = float(variant.get("start_mu", 0.0))
    attn["sc_mu_start_step"] = int(variant.get("start_step", 720))
    attn["sc_mu_warmup_steps"] = int(variant.get("mu_warmup_steps", 720))

    scope = str(variant.get("scope", "all"))
    target_blocks = blocks_for(scope)
    if target_blocks:
        attn["target_blocks"] = target_blocks
        attn.pop("stage_selection", None)
    else:
        attn.pop("target_blocks", None)
        attn["stage_selection"] = "all"

    loss = cfg.setdefault("loss", {})
    loss["lambda_ang"] = float(variant.get("lambda_ang", 0.0))
    loss["use_angular_loss"] = bool(loss["lambda_ang"])

    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif["target_rate"] = None
    atlif["target_rate_eta"] = 0.0
    atlif["activity_eta"] = 0.0
    atlif["threshold_base_lr"] = 5.0e-6

    opt = cfg.setdefault("optimizer", {})
    opt["lr"] = float(variant.get("lr", 2.0e-5))
    opt["milestones"] = list(variant.get("milestones", [22, 27]))
    opt["lr_warmup"] = {"enabled": True, "steps": int(variant.get("lr_warmup", 450)), "start_factor": 0.05}
    groups = opt.setdefault("param_groups", {})
    groups["enabled"] = True
    groups["backbone_lr"] = float(variant.get("backbone_lr", 1.0e-6))
    groups["norm_lr"] = float(variant.get("norm_lr", 1.0e-6))
    groups["neuron_lr"] = float(variant.get("neuron_lr", 3.0e-5))
    groups["threshold_lr"] = float(variant.get("threshold_lr", 5.0e-6))

    cfg["note"] = (
        "NSC-08/H58 late SC residual. "
        f"name={variant['name']}; scope={scope}; final_mu={attn['bipolar_mu']}; "
        f"lambda={attn['bipolar_lambda']}; start_step={attn['sc_mu_start_step']}; "
        f"mu_warmup={attn['sc_mu_warmup_steps']}; lambda_ang={loss['lambda_ang']}; "
        f"lr={opt['lr']}; backbone_lr={groups['backbone_lr']}; target_rate=None."
    )


def main() -> int:
    base = read_yaml(BASE)
    variants = [
        {"name": "nsc08a_h58_all_mu010_l03_late720", "scope": "all", "mu": 0.10, "lam": 0.30},
        {"name": "nsc08b_h58_all_mu008_l03_late720", "scope": "all", "mu": 0.08, "lam": 0.30},
        {"name": "nsc08c_h58_all_mu010_l03_late360", "scope": "all", "mu": 0.10, "lam": 0.30, "start_step": 360, "mu_warmup_steps": 720},
        {"name": "nsc08d_h58_s2_mu010_l03_late360", "scope": "s2", "mu": 0.10, "lam": 0.30, "start_step": 360, "mu_warmup_steps": 720},
        {"name": "nsc08e_h58_s12_mu008_l03_late360", "scope": "s12", "mu": 0.08, "lam": 0.30, "start_step": 360, "mu_warmup_steps": 720},
        {"name": "nsc08f_h58_all_mu010_l03_ang005", "scope": "all", "mu": 0.10, "lam": 0.30, "lambda_ang": 0.005},
        {"name": "nsc08g_h58_all_mu010_l03_lr1e5", "scope": "all", "mu": 0.10, "lam": 0.30, "lr": 1.0e-5},
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
        short_attn = short_cfg.setdefault("bsa_attention", {})
        short_attn["sc_mu_start_step"] = int(variant.get("short_start_step", 120))
        short_attn["sc_mu_warmup_steps"] = int(variant.get("short_mu_warmup_steps", 120))
        short = GENERATED / f"{variant['name']}_steps360.yml"
        write_yaml(short, short_cfg)
        generated.append(f"generated/{short.name}")

        full_cfg = deepcopy(cfg)
        full_cfg["loader"]["n_epochs"] = 30
        full_cfg["runtime"]["max_train_steps"] = 0
        full_cfg["runtime"]["force_save_epochs"] = list(range(30))
        full_cfg["runtime"]["skip_state_save"] = True
        full = GENERATED / f"{variant['name']}_full30.yml"
        write_yaml(full, full_cfg)
        generated.append(f"generated/{full.name}")

    print("\n".join(generated))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
