"""Generate weaker no-Kmag hardware-friendly NTS configs.

NTS04 showed that strong mismatch/single-active penalties can over-sparsify
full training even when short valid40 looks acceptable. NTS05 keeps the same
H60 no-carrier/no-Kmag inference path and only weakens training regularizers.
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


def set_short_runtime(cfg: dict[str, Any]) -> None:
    cfg.setdefault("loader", {})["n_epochs"] = 1
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 360
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = [0]
    runtime["use_mlflow_model_logging"] = False


def set_attention(cfg: dict[str, Any], spec: dict[str, Any]) -> None:
    attn = cfg.setdefault("bsa_attention", {})
    attn["enabled"] = True
    attn["mode"] = "h60"
    attn["k_magnitude_alpha"] = 0.0
    attn["bipolar_mu"] = float(spec["bipolar_mu"])
    attn["alpha0"] = 0.02
    attn["mismatch_penalty"] = float(spec["mismatch_penalty"])
    attn["single_active_penalty"] = float(spec["single_active_penalty"])
    attn["single_active_penalty_grad"] = "ste"
    attn["center_scores"] = True
    attn["preserve_mean"] = True
    attn["consensus_score_norm"] = "head_dim"
    attn["value_mode"] = "threshold"
    attn["sc_mu_schedule_enabled"] = True
    attn["sc_mu_start"] = 0.0
    attn["sc_mu_start_step"] = 0
    attn["sc_mu_warmup_steps"] = int(spec["sc_mu_warmup_steps"])


def set_optimizer(cfg: dict[str, Any], spec: dict[str, Any]) -> None:
    optimizer = cfg.setdefault("optimizer", {})
    optimizer["milestones"] = [20, 25]
    groups = optimizer.setdefault("param_groups", {})
    groups["enabled"] = True
    groups["backbone_lr"] = float(spec.get("backbone_lr", 1.0e-6))
    groups["neuron_lr"] = float(spec.get("neuron_lr", 3.0e-5))
    groups["threshold_lr"] = float(spec.get("threshold_lr", 5.0e-6))
    warmup = optimizer.setdefault("lr_warmup", {})
    warmup["enabled"] = True
    warmup["steps"] = 200
    warmup["start_factor"] = 0.1


def make_config(base: dict[str, Any], spec: dict[str, Any]) -> Path:
    cfg = deepcopy(base)
    cfg["experiment"] = spec["name"]
    cfg["note"] = spec["note"]
    cfg.setdefault("loader", {})["batch_size"] = int(spec.get("batch_size", 8))
    set_short_runtime(cfg)
    set_attention(cfg, spec)
    set_optimizer(cfg, spec)
    out = GENERATED / f"{spec['name']}.yml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    base = read_yaml(BASE)
    specs: list[dict[str, Any]] = [
        {
            "name": "nts05a_hw_mu0075_mis005_sap0025_w720_s360",
            "bipolar_mu": 0.075,
            "mismatch_penalty": 0.05,
            "single_active_penalty": 0.025,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-05a: no-Kmag H60, weak mismatch/single-active penalties, mu 0->0.075 over 720 steps.",
        },
        {
            "name": "nts05b_hw_mu005_mis005_sap0025_w720_s360",
            "bipolar_mu": 0.05,
            "mismatch_penalty": 0.05,
            "single_active_penalty": 0.025,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-05b: no-Kmag H60, weaker mu 0->0.05 over 720 steps.",
        },
        {
            "name": "nts05c_hw_mu0075_mis005_sap0025_w1440_s360",
            "bipolar_mu": 0.075,
            "mismatch_penalty": 0.05,
            "single_active_penalty": 0.025,
            "sc_mu_warmup_steps": 1440,
            "note": "NTS-05c: no-Kmag H60, weak penalties with slower 1440-step mu warmup.",
        },
        {
            "name": "nts05d_hw_mu0075_mis000_sap0025_w720_s360",
            "bipolar_mu": 0.075,
            "mismatch_penalty": 0.0,
            "single_active_penalty": 0.025,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-05d: no-Kmag H60, no mismatch penalty, only weak single-active regularizer.",
        },
    ]
    for spec in specs:
        print(make_config(base, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
