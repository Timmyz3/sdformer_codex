"""Generate NTS09 hardware-friendly H60 configs with qk threshold freeze.

NTS07b established the best no-carrier / no-Kmag / no-target-rate whole-network
line so far. NTS08 caps the qk threshold; NTS09 keeps the same deploy path but
stops qk threshold updates after a chosen global step so late epochs cannot keep
driving ternary activity downward.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
BASE = GENERATED / "nts07b_hw_h60_ffn_update0_act0_s1224.yml"


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def set_runtime(cfg: dict[str, Any], name: str, note: str) -> None:
    cfg["experiment"] = name
    cfg["note"] = note
    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 1
    loader["batch_size"] = 8
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 1224
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = [0]
    runtime["use_mlflow_model_logging"] = False


def make_config(base: dict[str, Any], spec: dict[str, Any]) -> Path:
    cfg = deepcopy(base)
    set_runtime(cfg, spec["name"], spec["note"])

    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif["threshold_eta"] = float(spec["qk_threshold_eta"])
    atlif["threshold_lr_scale"] = float(spec["qk_threshold_lr_scale"])
    atlif["threshold_freeze_after_step"] = int(spec["freeze_after_step"])
    atlif["max_threshold"] = spec["qk_max_threshold"]
    atlif["target_rate"] = None
    atlif["target_rate_eta"] = 0.0
    atlif["activity_eta"] = 0.0

    for group in atlif.get("target_groups", []):
        group["threshold_eta"] = 0.0
        group["activity_eta"] = 0.0
        group["target_rate"] = None
        group["target_rate_eta"] = 0.0

    attn = cfg.setdefault("bsa_attention", {})
    attn["mode"] = "h60"
    attn["k_magnitude_alpha"] = 0.0
    attn["mismatch_penalty"] = 0.0
    attn["single_active_penalty"] = 0.0
    attn["target_rate"] = None
    attn["bipolar_mu"] = float(spec["bipolar_mu"])
    attn["sc_mu_schedule_enabled"] = True
    attn["sc_mu_start"] = 0.0
    attn["sc_mu_start_step"] = 0
    attn["sc_mu_warmup_steps"] = int(spec["sc_mu_warmup_steps"])

    out = GENERATED / f"{spec['name']}.yml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    base = read_yaml(BASE)
    specs: list[dict[str, Any]] = [
        {
            "name": "nts09a_hw_h60_freeze816_s1224",
            "qk_threshold_eta": 6.5e-4,
            "qk_threshold_lr_scale": 50000.0,
            "freeze_after_step": 816,
            "qk_max_threshold": 2.0,
            "bipolar_mu": 0.05,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-09a: freeze qk threshold updates after 816 steps so late short/full epochs keep NTS07b's mid-training ternary activity.",
        },
        {
            "name": "nts09b_hw_h60_freeze918_s1224",
            "qk_threshold_eta": 6.5e-4,
            "qk_threshold_lr_scale": 50000.0,
            "freeze_after_step": 918,
            "qk_max_threshold": 2.0,
            "bipolar_mu": 0.05,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-09b: freeze qk threshold updates after 75% of short-test steps, preserving more late adaptation than 09a.",
        },
        {
            "name": "nts09c_hw_h60_eta0325_freeze816_s1224",
            "qk_threshold_eta": 3.25e-4,
            "qk_threshold_lr_scale": 50000.0,
            "freeze_after_step": 816,
            "qk_max_threshold": 2.0,
            "bipolar_mu": 0.05,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-09c: combine slower qk threshold drift with late freeze for a more conservative no-carrier line.",
        },
        {
            "name": "nts09d_hw_h60_cap115_freeze816_s1224",
            "qk_threshold_eta": 6.5e-4,
            "qk_threshold_lr_scale": 50000.0,
            "freeze_after_step": 816,
            "qk_max_threshold": 1.15,
            "bipolar_mu": 0.05,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-09d: combine threshold cap 1.15 with late freeze, guarding both saturation and continued upward drift.",
        },
        # Sparse-biased variants: freeze later so epoch0+ can push qk threshold higher
        # before locking, trading a bit of AEE/AAE for lower total_spikes at valid825.
        {
            "name": "nts09e_hw_h60_freeze1224_s1224",
            "qk_threshold_eta": 6.5e-4,
            "qk_threshold_lr_scale": 50000.0,
            "freeze_after_step": 1224,
            "qk_max_threshold": 2.0,
            "bipolar_mu": 0.05,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-09e: freeze at epoch0 end (1224 steps) — mildest late-freeze, vs 09a freeze816.",
        },
        {
            "name": "nts09f_hw_h60_freeze6120_s1224",
            "qk_threshold_eta": 6.5e-4,
            "qk_threshold_lr_scale": 50000.0,
            "freeze_after_step": 6120,
            "qk_max_threshold": 2.0,
            "bipolar_mu": 0.05,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-09f: freeze after 5 epochs (6120 steps) — moderate threshold growth before lock.",
        },
        {
            "name": "nts09g_hw_h60_freeze12240_s1224",
            "qk_threshold_eta": 6.5e-4,
            "qk_threshold_lr_scale": 50000.0,
            "freeze_after_step": 12240,
            "qk_max_threshold": 2.0,
            "bipolar_mu": 0.05,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-09g: freeze after 10 epochs (12240 steps) — stronger sparsity bias.",
        },
        {
            "name": "nts09h_hw_h60_cap115_freeze12240_s1224",
            "qk_threshold_eta": 6.5e-4,
            "qk_threshold_lr_scale": 50000.0,
            "freeze_after_step": 12240,
            "qk_max_threshold": 1.15,
            "bipolar_mu": 0.05,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-09h: cap 1.15 + freeze after 10 epochs — sparse setpoint near NTS07b ep29.",
        },
        {
            "name": "nts09i_hw_h60_eta0013_freeze6120_s1224",
            "qk_threshold_eta": 1.3e-3,
            "qk_threshold_lr_scale": 50000.0,
            "freeze_after_step": 6120,
            "qk_max_threshold": 2.0,
            "bipolar_mu": 0.05,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-09i: 2x threshold_eta + freeze after 5 epochs — faster drift then lock.",
        },
    ]
    for spec in specs:
        print(make_config(base, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
