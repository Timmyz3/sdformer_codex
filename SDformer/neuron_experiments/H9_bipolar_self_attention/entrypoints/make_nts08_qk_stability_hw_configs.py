"""Generate NTS08 hardware-friendly H60 configs with qk threshold stabilization.

NTS07b fixed FFN ATLIF collapse by disabling FFN sparse manual update/loss, but
full30 still pushed qk ternary activity from ~7.3% to ~3.6%. NTS08 keeps the
same deploy path:

    score(Q,K) -> Shiftmax -> K * gate

No carrier, no K_mag, no target-rate controller. Only training-time qk
threshold dynamics and the TX/SC score fusion coefficient are swept.
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
    atlif["max_threshold"] = spec["qk_max_threshold"]
    atlif["target_rate"] = None
    atlif["target_rate_eta"] = 0.0
    atlif["activity_eta"] = 0.0

    # Keep NTS07b's FFN binary ATLIF modules installed, but keep their sparse
    # update/loss disabled. This changes neither old experiments nor deploy ops.
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
            "name": "nts08a_hw_h60_qk_eta0325_s1224",
            "qk_threshold_eta": 3.25e-4,
            "qk_threshold_lr_scale": 50000.0,
            "qk_max_threshold": 2.0,
            "bipolar_mu": 0.05,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-08a: NTS07b deploy path; halve qk threshold_eta to slow late ternary sparsification.",
        },
        {
            "name": "nts08b_hw_h60_qk_scale25k_s1224",
            "qk_threshold_eta": 6.5e-4,
            "qk_threshold_lr_scale": 25000.0,
            "qk_max_threshold": 2.0,
            "bipolar_mu": 0.05,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-08b: NTS07b deploy path; halve qk threshold_lr_scale to reduce manual threshold drift.",
        },
        {
            "name": "nts08c_hw_h60_qk_cap115_s1224",
            "qk_threshold_eta": 6.5e-4,
            "qk_threshold_lr_scale": 50000.0,
            "qk_max_threshold": 1.15,
            "bipolar_mu": 0.05,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-08c: NTS07b deploy path; cap qk threshold at 1.15 to prevent late over-sparsification.",
        },
        {
            "name": "nts08d_hw_h60_qk_eta0325_cap115_s1224",
            "qk_threshold_eta": 3.25e-4,
            "qk_threshold_lr_scale": 50000.0,
            "qk_max_threshold": 1.15,
            "bipolar_mu": 0.05,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-08d: NTS07b deploy path; combine slower qk update with threshold cap.",
        },
        {
            "name": "nts08e_hw_h60_mu0075_qk_eta0325_s1224",
            "qk_threshold_eta": 3.25e-4,
            "qk_threshold_lr_scale": 50000.0,
            "qk_max_threshold": 2.0,
            "bipolar_mu": 0.075,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-08e: NTS07b deploy path; modestly stronger SC residual with stabilized qk threshold.",
        },
        {
            "name": "nts08f_hw_h60_mu003_qk_eta0325_s1224",
            "qk_threshold_eta": 3.25e-4,
            "qk_threshold_lr_scale": 50000.0,
            "qk_max_threshold": 2.0,
            "bipolar_mu": 0.03,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-08f: NTS07b deploy path; weaker SC residual with stabilized qk threshold.",
        },
    ]
    for spec in specs:
        print(make_config(base, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
