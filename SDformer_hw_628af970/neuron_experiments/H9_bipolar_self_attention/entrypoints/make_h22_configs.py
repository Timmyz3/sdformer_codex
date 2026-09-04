"""Generate H22 hyperparameter screening configs for H18c-style direct attention.

H18c showed a useful warning: 40-step runs can look collapsed while 120-step
runs recover AEE/AAE. These configs keep the same paper-backed direct
alpha-XNOR + Shiftmax module and sweep the sparse/threshold/attention
hyperparameters independently so failures are not over-attributed to the
replacement itself.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXP_ROOT / "configs"


def load_config(name: str) -> dict[str, Any]:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_config(name: str, config: dict[str, Any]) -> None:
    with (CONFIG_DIR / name).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def common(base: dict[str, Any], experiment: str) -> dict[str, Any]:
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


def set_atlif(cfg: dict[str, Any], **kwargs: Any) -> None:
    cfg.setdefault("atlif_ternary_psn", {}).update(kwargs)


def set_attn(cfg: dict[str, Any], **kwargs: Any) -> None:
    cfg.setdefault("bsa_attention", {}).update(kwargs)


def set_optimizer(cfg: dict[str, Any], **kwargs: Any) -> None:
    cfg.setdefault("optimizer", {}).update(kwargs)


def main() -> None:
    base = load_config("h18c_alpha_xnor_direct_shiftmax_guard120.yml")

    # Sparse target sweep around the H18c point. H18c gives good AEE/AAE at
    # 120 steps but SOPs are high, so these directly test whether stronger
    # target-rate feedback can lower SOPs without the 40-step collapse.
    sparse_grid = [
        ("h22a_h18c_target045_eta04_act07", 0.045, 0.04, 0.7),
        ("h22b_h18c_target040_eta05_act08", 0.040, 0.05, 0.8),
        ("h22c_h18c_target035_eta08_act10", 0.035, 0.08, 1.0),
        ("h22d_h18c_target030_eta10_act12", 0.030, 0.10, 1.2),
    ]
    for name, target_rate, target_eta, activity_eta in sparse_grid:
        cfg = common(base, name)
        set_atlif(
            cfg,
            target_rate=target_rate,
            target_rate_eta=target_eta,
            activity_eta=activity_eta,
        )
        cfg["note"] = (
            "H22 sparse sweep: H18c direct alpha-XNOR + Shiftmax with "
            f"target_rate={target_rate}, target_rate_eta={target_eta}, "
            f"activity_eta={activity_eta}."
        )
        write_config(f"{name}_guard120.yml", cfg)

    # Keep sparse feedback fixed and test whether attention score sharpness is
    # the reason AAE recovers slowly. Lower score_scale is softer; higher is
    # more selective.
    for scale in (0.5, 0.75, 1.5):
        name = f"h22e_h18c_score{str(scale).replace('.', 'p')}"
        cfg = common(base, name)
        set_attn(cfg, score_scale=scale)
        cfg["note"] = f"H22 score-scale sweep: H18c direct Shiftmax score_scale={scale}."
        write_config(f"{name}_guard120.yml", cfg)

    # Alpha-XNOR paper-inspired scoring knobs. alpha0 rewards same-silence;
    # mismatch_penalty controls opposite-polarity suppression.
    alpha_grid = [
        ("h22f_h18c_alpha0_penalty025", 0.0, 0.25),
        ("h22g_h18c_alpha001_penalty05", 0.01, 0.50),
        ("h22h_h18c_alpha005_penalty05", 0.05, 0.50),
    ]
    for name, alpha0, penalty in alpha_grid:
        cfg = common(base, name)
        set_attn(cfg, alpha0=alpha0, mismatch_penalty=penalty)
        cfg["note"] = (
            "H22 alpha-XNOR scoring sweep: "
            f"alpha0={alpha0}, mismatch_penalty={penalty}."
        )
        write_config(f"{name}_guard120.yml", cfg)

    # Direct matrix ablations for compatibility/hardware. sign value is more
    # hardware-friendly but risks losing ATLIF threshold amplitude; active norm
    # makes scores sparse-aware.
    cfg = common(base, "h22i_h18c_active_norm")
    set_attn(cfg, consensus_score_norm="active")
    cfg["note"] = "H22 direct attention with sparse-aware active-count normalization."
    write_config("h22i_h18c_active_norm_guard120.yml", cfg)

    cfg = common(base, "h22j_h18c_sign_value")
    set_attn(cfg, value_mode="sign")
    cfg["note"] = "H22 direct attention with ternary sign values instead of ATLIF threshold amplitude."
    write_config("h22j_h18c_sign_value_guard120.yml", cfg)

    # Optimizer sensitivity: the same module may need a lower LR to keep AAE,
    # or a slightly higher LR to recover from the initial sparse collapse.
    for lr in (1.0e-5, 3.0e-5):
        name = f"h22k_h18c_lr{str(lr).replace('.', 'p').replace('-', 'm')}"
        cfg = common(base, name)
        set_optimizer(cfg, lr=lr)
        cfg["note"] = f"H22 optimizer sweep: H18c direct Shiftmax lr={lr}."
        write_config(f"{name}_guard120.yml", cfg)


if __name__ == "__main__":
    main()
