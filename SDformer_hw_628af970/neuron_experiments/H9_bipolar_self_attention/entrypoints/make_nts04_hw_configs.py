"""Generate hardware-clean NTS-00 refinement configs.

These configs keep the NTS-00 story intact:
no K_mag, no carrier, TX+SC score-level fusion, one Shiftmax, S2-only blocks.
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
    attn["mismatch_penalty"] = float(spec.get("mismatch_penalty", 0.25))
    attn["single_active_penalty"] = float(spec.get("single_active_penalty", 0.05))
    attn["single_active_penalty_grad"] = "ste"
    attn["center_scores"] = True
    attn["preserve_mean"] = True
    attn["consensus_score_norm"] = "head_dim"
    attn["value_mode"] = "threshold"
    attn["sc_mu_schedule_enabled"] = bool(spec.get("sc_mu_schedule_enabled", False))
    attn["sc_mu_start"] = float(spec.get("sc_mu_start", 0.0))
    attn["sc_mu_start_step"] = int(spec.get("sc_mu_start_step", 0))
    attn["sc_mu_warmup_steps"] = int(spec.get("sc_mu_warmup_steps", 0))


def set_optimizer(cfg: dict[str, Any], spec: dict[str, Any]) -> None:
    optimizer = cfg.setdefault("optimizer", {})
    optimizer["milestones"] = list(spec.get("milestones", [20, 25]))
    groups = optimizer.setdefault("param_groups", {})
    groups["enabled"] = True
    groups["backbone_lr"] = float(spec.get("backbone_lr", 1.0e-6))
    groups["neuron_lr"] = float(spec.get("neuron_lr", 3.0e-5))
    groups["threshold_lr"] = float(spec.get("threshold_lr", 5.0e-6))
    warmup = optimizer.setdefault("lr_warmup", {})
    warmup["enabled"] = True
    warmup["steps"] = int(spec.get("warmup_steps", 200))
    warmup["start_factor"] = float(spec.get("warmup_start", 0.1))


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
            "name": "nts04a_hw_mu0075_s360",
            "bipolar_mu": 0.075,
            "note": "NTS-04a: no-Kmag hardware-clean NTS, fixed mu=0.075.",
        },
        {
            "name": "nts04b_hw_mu010_sap025_s360",
            "bipolar_mu": 0.10,
            "single_active_penalty": 0.025,
            "note": "NTS-04b: no-Kmag fixed mu=0.10, weaker single-active penalty.",
        },
        {
            "name": "nts04c_hw_mu010_mis020_s360",
            "bipolar_mu": 0.10,
            "mismatch_penalty": 0.20,
            "note": "NTS-04c: no-Kmag fixed mu=0.10, weaker opposite-sign penalty.",
        },
        {
            "name": "nts04d_hw_mu0125_s360",
            "bipolar_mu": 0.125,
            "note": "NTS-04d: no-Kmag fixed mu=0.125.",
        },
        {
            "name": "nts04e_hw_sched005_w360_s360",
            "bipolar_mu": 0.05,
            "sc_mu_schedule_enabled": True,
            "sc_mu_start": 0.0,
            "sc_mu_start_step": 0,
            "sc_mu_warmup_steps": 360,
            "note": "NTS-04e: no-Kmag mu schedule 0->0.05 over 360 steps.",
        },
        {
            "name": "nts04f_hw_sched010_w360_s360",
            "bipolar_mu": 0.10,
            "sc_mu_schedule_enabled": True,
            "sc_mu_start": 0.0,
            "sc_mu_start_step": 0,
            "sc_mu_warmup_steps": 360,
            "note": "NTS-04f: no-Kmag mu schedule 0->0.10 over 360 steps.",
        },
        {
            "name": "nts04g_hw_sched010_w720_s360",
            "bipolar_mu": 0.10,
            "sc_mu_schedule_enabled": True,
            "sc_mu_start": 0.0,
            "sc_mu_start_step": 0,
            "sc_mu_warmup_steps": 720,
            "note": "NTS-04g: no-Kmag slow mu schedule 0->0.10 over 720 steps.",
        },
        {
            "name": "nts04h_hw_sched010_slowlr_s360",
            "bipolar_mu": 0.10,
            "sc_mu_schedule_enabled": True,
            "sc_mu_start": 0.0,
            "sc_mu_start_step": 0,
            "sc_mu_warmup_steps": 360,
            "neuron_lr": 2.0e-5,
            "threshold_lr": 3.0e-6,
            "backbone_lr": 5.0e-7,
            "note": "NTS-04h: no-Kmag mu schedule 0->0.10 with slower LR.",
        },
    ]
    for spec in specs:
        print(make_config(base, spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
