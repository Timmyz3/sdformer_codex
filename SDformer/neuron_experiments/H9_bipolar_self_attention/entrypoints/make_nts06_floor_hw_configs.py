"""Generate no-Kmag H60 configs with weaker single-active pressure.

NTS05d showed that single_active_penalty=0.025 still over-sparsifies during
full training. NTS06 keeps the same deploy path and tests only weaker training
regularization.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
BASE = GENERATED / "ntx_h60_full30.yml"


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def make_config(base: dict[str, Any], spec: dict[str, Any]) -> Path:
    cfg = deepcopy(base)
    cfg["experiment"] = spec["name"]
    cfg["note"] = spec["note"]

    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 1
    loader["batch_size"] = 8

    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 360
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = [0]
    runtime["use_mlflow_model_logging"] = False

    attn = cfg.setdefault("bsa_attention", {})
    attn["enabled"] = True
    attn["mode"] = "h60"
    attn["center_scores"] = True
    attn["preserve_mean"] = True
    attn["consensus_score_norm"] = "head_dim"
    attn["value_mode"] = "threshold"
    attn["k_magnitude_alpha"] = 0.0
    attn["mismatch_penalty"] = float(spec["mismatch_penalty"])
    attn["single_active_penalty"] = float(spec["single_active_penalty"])
    attn["single_active_penalty_grad"] = "ste"
    attn["bipolar_mu"] = float(spec["bipolar_mu"])
    attn["alpha0"] = 0.02
    attn["sc_mu_schedule_enabled"] = True
    attn["sc_mu_start"] = 0.0
    attn["sc_mu_start_step"] = 0
    attn["sc_mu_warmup_steps"] = int(spec["sc_mu_warmup_steps"])

    optimizer = cfg.setdefault("optimizer", {})
    optimizer["milestones"] = [20, 25]
    groups = optimizer.setdefault("param_groups", {})
    groups["enabled"] = True
    groups["backbone_lr"] = 1.0e-6
    groups["neuron_lr"] = 3.0e-5
    groups["threshold_lr"] = 5.0e-6
    warmup = optimizer.setdefault("lr_warmup", {})
    warmup["enabled"] = True
    warmup["steps"] = 200
    warmup["start_factor"] = 0.1

    out = GENERATED / f"{spec['name']}.yml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    base = read_yaml(BASE)
    specs: list[dict[str, Any]] = [
        {
            "name": "nts06a_hw_mu005_mis000_sap000_w720_s360",
            "bipolar_mu": 0.05,
            "mismatch_penalty": 0.0,
            "single_active_penalty": 0.0,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-06a: no-Kmag H60, no mismatch, no single-active; weakest full-stability probe.",
        },
        {
            "name": "nts06b_hw_mu005_mis000_sap001_w720_s360",
            "bipolar_mu": 0.05,
            "mismatch_penalty": 0.0,
            "single_active_penalty": 0.01,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-06b: no-Kmag H60, no mismatch, very weak single-active=0.01.",
        },
        {
            "name": "nts06c_hw_mu0075_mis000_sap000_w720_s360",
            "bipolar_mu": 0.075,
            "mismatch_penalty": 0.0,
            "single_active_penalty": 0.0,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-06c: no-Kmag H60, keep mu=0.075 but remove single-active.",
        },
        {
            "name": "nts06d_hw_mu005_mis000_sap000_w1440_s360",
            "bipolar_mu": 0.05,
            "mismatch_penalty": 0.0,
            "single_active_penalty": 0.0,
            "sc_mu_warmup_steps": 1440,
            "note": "NTS-06d: no-Kmag H60, weakest penalty with slower mu warmup.",
        },
    ]
    for spec in specs:
        print(make_config(base, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
