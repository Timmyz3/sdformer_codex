"""Generate NTS pure TX+SC fusion configs (k_magnitude_alpha=0) for 360-step short tests."""

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


def set_nts_attention(cfg: dict[str, Any], spec: dict[str, Any]) -> None:
    attn = cfg.setdefault("bsa_attention", {})
    attn["enabled"] = True
    attn["mode"] = "h60"
    attn["k_magnitude_alpha"] = 0.0
    attn["bipolar_mu"] = float(spec["bipolar_mu"])
    attn["alpha0"] = 0.02
    attn["mismatch_penalty"] = 0.25
    attn["single_active_penalty"] = 0.05
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
    groups = cfg.setdefault("optimizer", {}).setdefault("param_groups", {})
    groups["enabled"] = True
    groups["backbone_lr"] = float(spec.get("backbone_lr", 1.0e-6))
    groups["neuron_lr"] = float(spec.get("neuron_lr", 3.0e-5))
    groups["threshold_lr"] = float(spec.get("threshold_lr", 5.0e-6))
    warmup = cfg.setdefault("optimizer", {}).setdefault("lr_warmup", {})
    warmup["enabled"] = True
    warmup["steps"] = int(spec.get("warmup_steps", 200))
    warmup["start_factor"] = float(spec.get("warmup_start", 0.1))


def make_config(base: dict[str, Any], spec: dict[str, Any]) -> Path:
    cfg = deepcopy(base)
    cfg["experiment"] = spec["name"]
    cfg["note"] = spec["note"]
    cfg.setdefault("loader", {})["batch_size"] = int(spec.get("batch_size", 8))
    set_short_runtime(cfg)
    set_nts_attention(cfg, spec)
    set_optimizer(cfg, spec)
    out = GENERATED / f"{spec['name']}.yml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    base = read_yaml(BASE)
    specs: list[dict[str, Any]] = [
        {
            "name": "nts00a_mu005_std_s360",
            "bipolar_mu": 0.05,
            "note": "NTS-00a: pure TX+SC score fusion, mu=0.05, no K_mag, default LR.",
        },
        {
            "name": "nts00b_mu010_std_s360",
            "bipolar_mu": 0.10,
            "note": "NTS-00b: pure TX+SC, mu=0.10 (NTS-01 mu without K_mag), default LR.",
        },
        {
            "name": "nts00c_mu015_std_s360",
            "bipolar_mu": 0.15,
            "note": "NTS-00c: pure TX+SC, mu=0.15, no K_mag, default LR.",
        },
        {
            "name": "nts00d_mu005_fast_s360",
            "bipolar_mu": 0.05,
            "neuron_lr": 5.0e-5,
            "backbone_lr": 2.0e-6,
            "note": "NTS-00d: mu=0.05, faster neuron/backbone LR.",
        },
        {
            "name": "nts00e_mu010_fast_s360",
            "bipolar_mu": 0.10,
            "neuron_lr": 5.0e-5,
            "backbone_lr": 2.0e-6,
            "note": "NTS-00e: mu=0.10, faster LR.",
        },
        {
            "name": "nts00f_mu005_slow_s360",
            "bipolar_mu": 0.05,
            "neuron_lr": 2.0e-5,
            "backbone_lr": 5.0e-7,
            "note": "NTS-00f: mu=0.05, slower LR for stability.",
        },
        {
            "name": "nts00g_mu010_bs6_s360",
            "bipolar_mu": 0.10,
            "batch_size": 6,
            "note": "NTS-00g: mu=0.10, batch_size=6 (match NTS-01 full30).",
        },
        {
            "name": "nts00h_mu005_sched_s360",
            "bipolar_mu": 0.05,
            "sc_mu_schedule_enabled": True,
            "sc_mu_start": 0.0,
            "sc_mu_start_step": 0,
            "sc_mu_warmup_steps": 360,
            "note": "NTS-00h: mu 0→0.05 linear schedule over 360 steps, no K_mag.",
        },
    ]
    paths = [make_config(base, spec) for spec in specs]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())