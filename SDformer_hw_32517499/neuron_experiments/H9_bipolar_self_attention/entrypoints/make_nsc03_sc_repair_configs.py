"""Generate NSC-03 short-screen and full30 configs for repairing the SC route."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = EXP_ROOT / "configs" / "generated"
BASE = GENERATED_DIR / "stride_h56a_sc_agree_disagree_l08_fast_warm_tr07_full30.yml"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def set_ffn_scope(cfg: dict[str, Any], scope: str) -> None:
    groups = cfg.setdefault("atlif_ternary_psn", {}).setdefault("target_groups", [])
    if scope == "s0":
        keep = {"s0_ffn"}
    elif scope == "s01":
        keep = {"s0_ffn", "s1_ffn"}
    elif scope == "s012half":
        keep = {"s0_ffn", "s1_ffn", "s2_half"}
    else:
        raise ValueError(f"unknown FFN scope: {scope}")
    cfg["atlif_ternary_psn"]["target_groups"] = [g for g in groups if g.get("name") in keep]


def apply_variant(cfg: dict[str, Any], variant: dict[str, Any]) -> None:
    cfg["experiment"] = variant["name"]
    cfg.setdefault("test", {})["scale_factor"] = 1

    loss = cfg.setdefault("loss", {})
    loss["lambda_ang"] = float(variant.get("lambda_ang", 0.02))
    loss["use_angular_loss"] = loss["lambda_ang"] != 0.0

    atlif = cfg.setdefault("atlif_ternary_psn", {})
    tr = variant.get("target_rate")
    atlif["target_rate"] = None if tr is None else float(tr)
    atlif["target_rate_eta"] = float(variant.get("target_rate_eta", 0.0 if tr is None else 0.02))
    atlif["threshold_lr_scale"] = float(variant.get("threshold_lr_scale", 20000.0))
    atlif["threshold_eta"] = float(variant.get("threshold_eta", 5.0e-4))
    atlif["threshold_base_lr"] = float(variant.get("threshold_base_lr", 3.0e-6))
    set_ffn_scope(cfg, str(variant.get("ffn_scope", "s01")))

    for group in atlif.get("target_groups", []):
        group["threshold_lr_scale"] = int(variant.get("ffn_threshold_lr_scale", 3000))
        group["threshold_eta"] = float(variant.get("ffn_threshold_eta", 4.0e-5))
        group["activity_eta"] = float(variant.get("ffn_activity_eta", 1.0))

    attn = cfg.setdefault("bsa_attention", {})
    attn["mode"] = str(variant.get("mode", "sc_agree_disagree_shiftmax"))
    attn["bipolar_lambda"] = float(variant.get("bipolar_lambda", 0.5))
    attn["residual_alpha"] = float(variant.get("residual_alpha", 0.3))
    attn["deadzone_epsilon"] = float(variant.get("deadzone_epsilon", 0.0))
    attn["confidence_enabled"] = bool(variant.get("confidence_enabled", False))
    attn["k_consistency_mod"] = bool(variant.get("k_consistency_mod", False))
    attn["consensus_score_norm"] = str(variant.get("consensus_score_norm", "head_dim"))

    opt = cfg.setdefault("optimizer", {})
    groups = opt.setdefault("param_groups", {})
    groups["enabled"] = True
    groups["backbone_lr"] = float(variant.get("backbone_lr", 3.0e-7))
    groups["norm_lr"] = float(variant.get("norm_lr", 3.0e-7))
    groups["neuron_lr"] = float(variant.get("neuron_lr", 1.5e-5))
    groups["threshold_lr"] = float(variant.get("threshold_lr", 3.0e-6))
    opt["lr_warmup"] = {"enabled": True, "steps": 200, "start_factor": 0.1}
    opt["milestones"] = [20, 25]

    cfg["note"] = (
        "NSC-03 SC repair sweep. "
        f"mode={attn['mode']}; lambda={attn['bipolar_lambda']}; residual_alpha={attn['residual_alpha']}; "
        f"target_rate={atlif['target_rate']}; target_rate_eta={atlif['target_rate_eta']}; "
        f"ffn_scope={variant.get('ffn_scope', 's01')}; lambda_ang={loss['lambda_ang']}."
    )


def main() -> int:
    base = read_yaml(BASE)
    variants: list[dict[str, Any]] = [
        {
            "name": "nsc03a_l05_notr_s01_ang02",
            "bipolar_lambda": 0.5,
            "target_rate": None,
            "ffn_scope": "s01",
        },
        {
            "name": "nsc03b_l05_tr03_s01_ang02",
            "bipolar_lambda": 0.5,
            "target_rate": 0.03,
            "target_rate_eta": 0.02,
            "ffn_scope": "s01",
        },
        {
            "name": "nsc03c_l06_notr_s01_ang02",
            "bipolar_lambda": 0.6,
            "target_rate": None,
            "ffn_scope": "s01",
        },
        {
            "name": "nsc03d_l05_notr_s0_ang02",
            "bipolar_lambda": 0.5,
            "target_rate": None,
            "ffn_scope": "s0",
        },
        {
            "name": "nsc03e_l05_notr_s01_conf_ang02",
            "mode": "sc_ad_confidence_shiftmax",
            "bipolar_lambda": 0.5,
            "target_rate": None,
            "ffn_scope": "s01",
            "deadzone_epsilon": 1.0 / 32.0,
            "confidence_enabled": True,
        },
        {
            "name": "nsc03f_l05_notr_s01_actnorm_ang02",
            "mode": "sc_ad_activenorm_shiftmax",
            "bipolar_lambda": 0.5,
            "target_rate": None,
            "ffn_scope": "s01",
            "consensus_score_norm": "active",
        },
        {
            "name": "nsc03g_l05_notr_s01_res03_ang02",
            "mode": "sc_agree_disagree_residual_shiftmax",
            "bipolar_lambda": 0.5,
            "target_rate": None,
            "ffn_scope": "s01",
            "residual_alpha": 0.3,
        },
        {
            "name": "nsc03h_l06_tr03_s01_res03_ang02",
            "mode": "sc_agree_disagree_residual_shiftmax",
            "bipolar_lambda": 0.6,
            "target_rate": 0.03,
            "target_rate_eta": 0.02,
            "ffn_scope": "s01",
            "residual_alpha": 0.3,
        },
    ]

    generated: list[str] = []
    for variant in variants:
        cfg = deepcopy(base)
        apply_variant(cfg, variant)

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
