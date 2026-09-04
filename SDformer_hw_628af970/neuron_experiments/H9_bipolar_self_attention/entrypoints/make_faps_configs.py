"""Generate FAPS short-test configs (360 steps) for rapid_screen."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
BASE = GENERATED / "ntx_h60_v2_mu005_a003_full30.yml"


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


def set_faps_attention(cfg: dict[str, Any], spec: dict[str, Any]) -> None:
    attn = cfg.setdefault("bsa_attention", {})
    mode = str(spec.get("mode", "faps"))
    attn["enabled"] = True
    attn["mode"] = mode
    attn["center_scores"] = True
    attn["preserve_mean"] = True
    attn["alpha0"] = 0.02
    attn["mismatch_penalty"] = 0.25
    attn["single_active_penalty"] = 0.05
    attn["single_active_penalty_grad"] = "ste"
    attn["consensus_score_norm"] = "head_dim"
    attn["value_mode"] = "threshold"
    attn["directional_channels_enabled"] = bool(spec.get("directional_channels_enabled", True))
    attn["directional_merge_mode"] = str(spec.get("directional_merge_mode", "mean"))
    attn["flow_disagreement_gamma"] = float(spec.get("flow_disagreement_gamma", 0.0))
    attn["k_magnitude_alpha"] = float(spec.get("k_magnitude_alpha", 0.0))
    attn["confidence_min_active"] = int(spec.get("confidence_min_active", 0) or 0)
    attn["kmag_quantize_bits"] = int(spec.get("kmag_quantize_bits", 2) or 2)
    if mode == "h60":
        attn["bipolar_mu"] = float(spec.get("bipolar_mu", 0.05))
        attn["k_magnitude_alpha"] = float(spec.get("k_magnitude_alpha", 0.0))


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
    set_faps_attention(cfg, spec)
    set_optimizer(cfg, spec)
    out = GENERATED / f"{spec['name']}.yml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    base = read_yaml(BASE)
    specs: list[dict[str, Any]] = [
        {
            "name": "faps00a_dir_nokmag_s360",
            "note": "FAPS-00a: directional x/y unified dyadic popcount, no K_mag.",
            "k_magnitude_alpha": 0.0,
        },
        {
            "name": "faps00b_dir_kmag032_s360",
            "note": "FAPS-00b: directional + sparse 2-bit K_mag (alpha=1/32, active>=8).",
            "k_magnitude_alpha": 0.03125,
            "confidence_min_active": 8,
        },
        {
            "name": "faps00c_h60_nokmag_s360",
            "note": "FAPS-00c control: h60 TX+SC (mu=0.05), no K_mag, no directional.",
            "mode": "h60",
            "directional_channels_enabled": False,
            "bipolar_mu": 0.05,
            "k_magnitude_alpha": 0.0,
        },
        {
            "name": "faps00a_fast_s360",
            "note": "FAPS-00a fast LR: directional unified dyadic, no K_mag.",
            "k_magnitude_alpha": 0.0,
            "neuron_lr": 5.0e-5,
            "backbone_lr": 2.0e-6,
        },
        {
            "name": "faps00b_fast_s360",
            "note": "FAPS-00b fast LR: directional + sparse K_mag.",
            "k_magnitude_alpha": 0.03125,
            "confidence_min_active": 8,
            "neuron_lr": 5.0e-5,
            "backbone_lr": 2.0e-6,
        },
        {
            "name": "faps00d_gamma025_s360",
            "note": "FAPS-00d: directional + flow disagreement gamma=0.25, no K_mag.",
            "k_magnitude_alpha": 0.0,
            "flow_disagreement_gamma": 0.25,
        },
    ]
    paths = [make_config(base, spec) for spec in specs]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())