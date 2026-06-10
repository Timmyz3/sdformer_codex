"""Generate NSC-09 H59 score-residual SC screening configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
BASE = GENERATED / "ntx04h_cptc_ntx01_warm.yml"
S2_BLOCKS = ["2:0", "2:1", "2:2", "2:3", "2:4", "2:5"]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def configure(base: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(base)
    cfg["experiment"] = variant["name"]
    attn = cfg.setdefault("bsa_attention", {})
    attn["enabled"] = True
    attn["mode"] = "tx_sc_score_residual_shiftmax"
    attn["bipolar_mu"] = float(variant["mu"])
    attn["bipolar_lambda"] = float(variant.get("lam", 0.30))
    attn["alpha0"] = 0.02
    attn["mismatch_penalty"] = 0.25
    attn["single_active_penalty"] = 0.0
    attn["single_active_penalty_grad"] = "ste"
    attn["center_scores"] = True
    attn["preserve_mean"] = True
    attn["confidence_enabled"] = False
    attn["k_consistency_mod"] = False
    attn["sc_mu_schedule_enabled"] = bool(variant.get("schedule", False))
    attn["sc_mu_start"] = 0.0
    attn["sc_mu_start_step"] = int(variant.get("start_step", 120))
    attn["sc_mu_warmup_steps"] = int(variant.get("warmup_steps", 120))
    if variant.get("scope", "all") == "s2":
        attn["target_blocks"] = S2_BLOCKS
        attn.pop("stage_selection", None)
    else:
        attn.pop("target_blocks", None)
        attn["stage_selection"] = "all"

    cfg.setdefault("loss", {})["lambda_ang"] = float(variant.get("lambda_ang", 0.0))
    cfg["loss"]["use_angular_loss"] = bool(cfg["loss"]["lambda_ang"])
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif["target_rate"] = None
    atlif["target_rate_eta"] = 0.0
    atlif["activity_eta"] = 0.0
    opt = cfg.setdefault("optimizer", {})
    opt["lr_warmup"] = {"enabled": True, "steps": 450, "start_factor": 0.05}
    groups = opt.setdefault("param_groups", {})
    groups["enabled"] = True
    groups["backbone_lr"] = 1.0e-6
    groups["norm_lr"] = 1.0e-6
    groups["neuron_lr"] = 3.0e-5
    groups["threshold_lr"] = 5.0e-6
    cfg["note"] = (
        "NSC-09/H59 score residual: carrier and TX shiftmax stay dominant; "
        f"SC enters scores before one final shiftmax. name={variant['name']}; "
        f"scope={variant.get('scope', 'all')}; mu={attn['bipolar_mu']}; "
        f"schedule={attn['sc_mu_schedule_enabled']}; lambda_ang={cfg['loss']['lambda_ang']}."
    )
    return cfg


def main() -> int:
    base = read_yaml(BASE)
    variants = [
        {"name": "nsc09a_h59_all_mu002", "scope": "all", "mu": 0.02},
        {"name": "nsc09b_h59_all_mu005", "scope": "all", "mu": 0.05},
        {"name": "nsc09c_h59_s2_mu005", "scope": "s2", "mu": 0.05},
        {"name": "nsc09d_h59_all_mu005_sched", "scope": "all", "mu": 0.05, "schedule": True},
    ]
    generated: list[str] = []
    for variant in variants:
        cfg = configure(base, variant)
        short_cfg = deepcopy(cfg)
        short_cfg["loader"]["n_epochs"] = 1
        short_cfg["runtime"]["max_train_steps"] = 360
        short_cfg["runtime"]["force_save_epochs"] = [0]
        short_cfg["runtime"]["skip_state_save"] = True
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
