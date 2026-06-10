"""Generate H62 confidence-calibrated NTS short-test configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
BASE = GENERATED / "nts03_mu005_a003_full30.yml"


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def set_short_runtime(cfg: dict[str, Any]) -> None:
    cfg.setdefault("loader", {})["n_epochs"] = 1
    cfg.setdefault("loader", {})["batch_size"] = 8
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 360
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = [0]
    runtime["use_mlflow_model_logging"] = False


def set_h62_attention(cfg: dict[str, Any], spec: dict[str, Any]) -> None:
    attn = cfg.setdefault("bsa_attention", {})
    attn["enabled"] = True
    attn["mode"] = "h62"
    attn["center_scores"] = True
    attn["preserve_mean"] = True
    attn["alpha0"] = 0.02
    attn["mismatch_penalty"] = 0.25
    attn["single_active_penalty"] = 0.05
    attn["single_active_penalty_grad"] = "ste"
    attn["consensus_score_norm"] = "head_dim"
    attn["value_mode"] = "threshold"
    attn["bipolar_mu"] = float(spec.get("bipolar_mu", 0.05))
    attn["k_magnitude_alpha"] = float(spec.get("k_magnitude_alpha", 0.02))
    attn["directional_residual_gamma"] = float(spec.get("directional_residual_gamma", 0.0))
    attn["confidence_floor"] = float(spec.get("confidence_floor", 0.0))
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
    set_short_runtime(cfg)
    set_h62_attention(cfg, spec)
    set_optimizer(cfg, spec)
    out = GENERATED / f"{spec['name']}.yml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    base = read_yaml(BASE)
    specs: list[dict[str, Any]] = [
        {
            "name": "h62a_conf_sc_mu005_k002_s360",
            "bipolar_mu": 0.05,
            "k_magnitude_alpha": 0.02,
            "directional_residual_gamma": 0.0,
            "note": "H62a: NTS TX + confidence-gated SC residual, mu=0.05, k_mag=0.02.",
        },
        {
            "name": "h62b_conf_sc_mu010_k002_s360",
            "bipolar_mu": 0.10,
            "k_magnitude_alpha": 0.02,
            "directional_residual_gamma": 0.0,
            "note": "H62b: NTS TX + confidence-gated SC residual, mu=0.10, k_mag=0.02.",
        },
        {
            "name": "h62c_conf_sc_dir_g002_k002_s360",
            "bipolar_mu": 0.05,
            "k_magnitude_alpha": 0.02,
            "directional_residual_gamma": 0.02,
            "note": "H62c: confidence-gated SC plus tiny FAPS directional residual gamma=0.02.",
        },
        {
            "name": "h62d_conf_sc_dir_g005_k002_s360",
            "bipolar_mu": 0.05,
            "k_magnitude_alpha": 0.02,
            "directional_residual_gamma": 0.05,
            "note": "H62d: confidence-gated SC plus stronger FAPS directional residual gamma=0.05.",
        },
        {
            "name": "h62e_late_conf_sc_dir_g002_k002_s360",
            "bipolar_mu": 0.05,
            "k_magnitude_alpha": 0.02,
            "directional_residual_gamma": 0.02,
            "sc_mu_schedule_enabled": True,
            "sc_mu_start": 0.0,
            "sc_mu_warmup_steps": 360,
            "note": "H62e: H62c with mu schedule 0->0.05 over 360 steps.",
        },
        {
            "name": "h62f_conf_sc_dir_g002_nokmag_s360",
            "bipolar_mu": 0.05,
            "k_magnitude_alpha": 0.0,
            "directional_residual_gamma": 0.02,
            "note": "H62f: pure event H62, confidence-gated SC+DIR, no K_mag.",
        },
    ]
    for path in [make_config(base, spec) for spec in specs]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
