"""Generate NSC-04 configs for carrier-blended SC repair experiments."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = EXP_ROOT / "configs" / "generated"
BASE = GENERATED_DIR / "nsc03b_l05_tr03_s01_ang02_full30.yml"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def set_common(cfg: dict[str, Any], variant: dict[str, Any]) -> None:
    cfg["experiment"] = variant["name"]
    cfg.setdefault("test", {})["scale_factor"] = 1

    loss = cfg.setdefault("loss", {})
    loss["lambda_ang"] = float(variant.get("lambda_ang", 0.02))
    loss["use_angular_loss"] = loss["lambda_ang"] != 0.0

    atlif = cfg.setdefault("atlif_ternary_psn", {})
    tr = variant.get("target_rate", 0.03)
    atlif["target_rate"] = None if tr is None else float(tr)
    atlif["target_rate_eta"] = float(variant.get("target_rate_eta", 0.0 if tr is None else 0.02))
    atlif["threshold_lr_scale"] = float(variant.get("threshold_lr_scale", 20000.0))
    atlif["threshold_eta"] = float(variant.get("threshold_eta", 5.0e-4))
    atlif["threshold_base_lr"] = float(variant.get("threshold_base_lr", 3.0e-6))

    attn = cfg.setdefault("bsa_attention", {})
    attn["mode"] = str(variant.get("mode", "sc_ad_carrier_blend_shiftmax"))
    attn["bipolar_mu"] = float(variant.get("bipolar_mu", 0.5))
    attn["bipolar_lambda"] = float(variant.get("bipolar_lambda", 0.5))
    attn["consensus_score_norm"] = str(variant.get("consensus_score_norm", "head_dim"))
    attn["single_active_penalty"] = float(variant.get("single_active_penalty", 0.0))
    attn["single_active_penalty_grad"] = str(variant.get("single_active_penalty_grad", "hard"))
    attn["bipolar_gate_min"] = variant.get("bipolar_gate_min")
    attn["bipolar_gate_max"] = variant.get("bipolar_gate_max")

    opt = cfg.setdefault("optimizer", {})
    groups = opt.setdefault("param_groups", {})
    groups["enabled"] = True
    groups["backbone_lr"] = float(variant.get("backbone_lr", 3.0e-7))
    groups["norm_lr"] = float(variant.get("norm_lr", 3.0e-7))
    groups["neuron_lr"] = float(variant.get("neuron_lr", 1.5e-5))
    groups["threshold_lr"] = float(variant.get("threshold_lr", 3.0e-6))
    opt["lr_warmup"] = {
        "enabled": True,
        "steps": int(variant.get("warmup_steps", 200)),
        "start_factor": float(variant.get("warmup_start_factor", 0.1)),
    }
    opt["milestones"] = list(variant.get("milestones", [20, 25]))

    cfg["note"] = (
        "NSC-04 carrier-blended SC repair sweep. "
        f"mode={attn['mode']}; mu={attn['bipolar_mu']}; lambda={attn['bipolar_lambda']}; "
        f"target_rate={atlif['target_rate']}; target_rate_eta={atlif['target_rate_eta']}; "
        f"lr(backbone/neuron/threshold)={groups['backbone_lr']}/{groups['neuron_lr']}/{groups['threshold_lr']}; "
        f"warmup={opt['lr_warmup']['steps']}; lambda_ang={loss['lambda_ang']}."
    )


def main() -> int:
    base = read_yaml(BASE)
    variants: list[dict[str, Any]] = [
        {
            "name": "nsc04a_blend_mu025_l05_tr03_ang02",
            "bipolar_mu": 0.25,
            "bipolar_lambda": 0.5,
            "target_rate": 0.03,
        },
        {
            "name": "nsc04b_blend_mu05_l05_tr03_ang02",
            "bipolar_mu": 0.5,
            "bipolar_lambda": 0.5,
            "target_rate": 0.03,
        },
        {
            "name": "nsc04c_blend_mu075_l05_tr03_ang02",
            "bipolar_mu": 0.75,
            "bipolar_lambda": 0.5,
            "target_rate": 0.03,
        },
        {
            "name": "nsc04d_blend_mu05_l06_tr03_ang02",
            "bipolar_mu": 0.5,
            "bipolar_lambda": 0.6,
            "target_rate": 0.03,
        },
        {
            "name": "nsc04e_blend_mu05_l05_notr_ang02",
            "bipolar_mu": 0.5,
            "bipolar_lambda": 0.5,
            "target_rate": None,
        },
        {
            "name": "nsc04f_blend_mu05_l05_tr02_ang02",
            "bipolar_mu": 0.5,
            "bipolar_lambda": 0.5,
            "target_rate": 0.02,
            "target_rate_eta": 0.015,
        },
        {
            "name": "nsc04g_blend_mu05_l05_tr03_slowlr_ang02",
            "bipolar_mu": 0.5,
            "bipolar_lambda": 0.5,
            "target_rate": 0.03,
            "backbone_lr": 1.0e-7,
            "norm_lr": 1.0e-7,
            "neuron_lr": 1.0e-5,
            "threshold_lr": 2.0e-6,
            "warmup_steps": 400,
        },
        {
            "name": "nsc04h_blend_mu05_l05_tr03_clamp_ang02",
            "bipolar_mu": 0.5,
            "bipolar_lambda": 0.5,
            "target_rate": 0.03,
            "bipolar_gate_min": -1.0,
            "bipolar_gate_max": 1.5,
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

        full_cfg = deepcopy(cfg)
        full_cfg["loader"]["n_epochs"] = 30
        full_cfg["runtime"]["max_train_steps"] = 0
        full_cfg["runtime"]["force_save_epochs"] = list(range(30))
        full_cfg["runtime"]["skip_state_save"] = True
        full = GENERATED_DIR / f"{variant['name']}_full30.yml"
        write_yaml(full, full_cfg)
        generated.append(f"generated/{short.name}\tgenerated/{full.name}")

    print("\n".join(generated))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
