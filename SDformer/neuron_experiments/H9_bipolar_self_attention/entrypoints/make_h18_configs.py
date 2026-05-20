"""Generate H18 paper-backed attention screening configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXP_ROOT / "configs"


def load_config(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_config(name: str, config: dict) -> None:
    with (CONFIG_DIR / name).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def common(base: dict, experiment: str) -> dict:
    cfg = deepcopy(base)
    cfg["experiment"] = experiment
    cfg.setdefault("runtime", {})["max_train_steps"] = 120
    cfg["runtime"]["skip_state_save"] = True
    cfg.setdefault("loader", {})["n_epochs"] = 1
    cfg["loader"]["batch_size"] = 8
    cfg["loader"]["n_workers"] = 8
    cfg["loader"]["pin_memory"] = False
    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    cfg.setdefault("test", {})["sample"] = 10
    return cfg


def main() -> None:
    base = load_config("h13n_biascenter_shiftmax_target05_halfffn_down02_guard120.yml")

    cfg = common(base, "h18a_alpha_xnor_shiftmax_guard120")
    cfg["bsa_attention"]["mode"] = "ternary_alpha_xnor_shiftmax"
    cfg["bsa_attention"]["alpha0"] = 0.02
    cfg["bsa_attention"]["mismatch_penalty"] = 0.5
    cfg["bsa_attention"]["score_scale"] = 2.0
    cfg["bsa_attention"]["center_scores"] = True
    cfg["bsa_attention"]["preserve_mean"] = True
    cfg["bsa_attention"]["consensus_score_norm"] = "head_dim"
    cfg["note"] = "H18a alpha-XNOR ternary auxiliary gate, carrier-preserving."
    write_config("h18a_alpha_xnor_shiftmax_guard120.yml", cfg)

    cfg = common(base, "h18a_alpha_xnor_l1_guard120")
    cfg["bsa_attention"]["mode"] = "ternary_alpha_xnor_l1"
    cfg["bsa_attention"]["alpha0"] = 0.02
    cfg["bsa_attention"]["mismatch_penalty"] = 0.5
    cfg["bsa_attention"]["score_scale"] = 2.0
    cfg["bsa_attention"]["center_scores"] = False
    cfg["bsa_attention"]["preserve_mean"] = True
    cfg["bsa_attention"]["consensus_bias"] = 0.02
    cfg["bsa_attention"]["consensus_score_norm"] = "head_dim"
    cfg["note"] = "H18a alpha-XNOR ternary auxiliary gate with exact L1 normalization."
    write_config("h18a_alpha_xnor_l1_guard120.yml", cfg)

    cfg = common(base, "h18b_a2os2a_gate_guard120")
    cfg["bsa_attention"]["mode"] = "a2os2a_gate"
    cfg["bsa_attention"]["score_scale"] = 1.0
    cfg["bsa_attention"]["center_scores"] = False
    cfg["bsa_attention"]["preserve_mean"] = True
    cfg["bsa_attention"]["consensus_bias"] = 1.0e-6
    cfg["bsa_attention"]["consensus_score_norm"] = "head_dim"
    cfg["bsa_attention"]["relu_k_floor"] = 0.0
    cfg["note"] = "H18b A2OS2A-inspired binary-Q nonnegative-K auxiliary gate, carrier-preserving."
    write_config("h18b_a2os2a_gate_guard120.yml", cfg)

    cfg = common(base, "h18c_alpha_xnor_direct_shiftmax_guard120")
    cfg["bsa_attention"]["mode"] = "alpha_xnor_matrix_shiftmax"
    cfg["bsa_attention"]["alpha0"] = 0.02
    cfg["bsa_attention"]["mismatch_penalty"] = 0.25
    cfg["bsa_attention"]["score_scale"] = 1.0
    cfg["bsa_attention"]["center_scores"] = True
    cfg["bsa_attention"]["preserve_mean"] = False
    cfg["bsa_attention"]["value_mode"] = "threshold"
    cfg["bsa_attention"]["consensus_score_norm"] = "head_dim"
    cfg["note"] = "H18c direct alpha-XNOR token-token matrix replacement with Shiftmax."
    write_config("h18c_alpha_xnor_direct_shiftmax_guard120.yml", cfg)

    cfg = common(base, "h18d_alpha_xnor_direct_l1_guard120")
    cfg["bsa_attention"]["mode"] = "alpha_xnor_matrix_l1"
    cfg["bsa_attention"]["alpha0"] = 0.02
    cfg["bsa_attention"]["mismatch_penalty"] = 0.25
    cfg["bsa_attention"]["score_scale"] = 1.0
    cfg["bsa_attention"]["center_scores"] = False
    cfg["bsa_attention"]["preserve_mean"] = True
    cfg["bsa_attention"]["consensus_bias"] = 0.02
    cfg["bsa_attention"]["value_mode"] = "threshold"
    cfg["bsa_attention"]["consensus_score_norm"] = "head_dim"
    cfg["note"] = "H18d direct alpha-XNOR token-token matrix replacement with L1 normalization."
    write_config("h18d_alpha_xnor_direct_l1_guard120.yml", cfg)

    cfg = common(base, "h18e_a2os2a_direct_l1_guard120")
    cfg["bsa_attention"]["mode"] = "a2os2a_direct"
    cfg["bsa_attention"]["score_scale"] = 1.0
    cfg["bsa_attention"]["center_scores"] = False
    cfg["bsa_attention"]["preserve_mean"] = True
    cfg["bsa_attention"]["consensus_bias"] = 1.0e-6
    cfg["bsa_attention"]["value_mode"] = "threshold"
    cfg["bsa_attention"]["consensus_score_norm"] = "head_dim"
    cfg["note"] = "H18e direct A2OS2A-style binary-Q nonnegative-K matrix replacement."
    write_config("h18e_a2os2a_direct_l1_guard120.yml", cfg)

    for alpha0 in (0.0, 0.01, 0.05):
        for penalty in (0.0, 0.25, 0.5):
            stem = f"h18c_sweep_a{str(alpha0).replace('.', 'p')}_b{str(penalty).replace('.', 'p')}_guard120"
            cfg = common(base, stem)
            cfg["bsa_attention"]["mode"] = "alpha_xnor_matrix_shiftmax"
            cfg["bsa_attention"]["alpha0"] = alpha0
            cfg["bsa_attention"]["mismatch_penalty"] = penalty
            cfg["bsa_attention"]["score_scale"] = 1.0
            cfg["bsa_attention"]["center_scores"] = True
            cfg["bsa_attention"]["preserve_mean"] = False
            cfg["bsa_attention"]["value_mode"] = "threshold"
            cfg["bsa_attention"]["consensus_score_norm"] = "head_dim"
            cfg["note"] = f"H18c direct alpha-XNOR sweep alpha0={alpha0}, mismatch_penalty={penalty}."
            write_config(f"{stem}.yml", cfg)

    cfg = common(base, "h13v_target05_lower_lr_guard120")
    cfg["optimizer"]["lr"] = 1.0e-5
    cfg["note"] = "H13n hyperparam repair: lower LR to reduce AAE drift."
    write_config("h13v_target05_lower_lr_guard120.yml", cfg)

    cfg = common(base, "h13w_sparse_feedback_stronger_guard120")
    cfg["atlif_ternary_psn"]["target_rate"] = 0.035
    cfg["atlif_ternary_psn"]["target_rate_eta"] = 0.08
    cfg["atlif_ternary_psn"]["activity_eta"] = 1.0
    cfg["note"] = "H13n hyperparam repair: stronger sparse target feedback."
    write_config("h13w_sparse_feedback_stronger_guard120.yml", cfg)

    cfg = common(base, "h13x_threshold_frozen_guard120")
    cfg["atlif_ternary_psn"]["trainable"] = "atlif_only"
    cfg["atlif_ternary_psn"]["threshold_eta"] = 0.0
    cfg["atlif_ternary_psn"]["target_rate_eta"] = 0.0
    cfg["atlif_ternary_psn"]["activity_eta"] = 0.2
    cfg["note"] = "H13n hyperparam diagnosis: no manual threshold feedback, train ATLIF weights only."
    write_config("h13x_threshold_frozen_guard120.yml", cfg)


if __name__ == "__main__":
    main()
