"""Generate NTX-04 carrier-preserving TX configs from the NTX-01 line.

NTX-04 starts from the best standard TX line (`stride_h41_tx_s02c_v2`) instead
of the weaker qkselector branch. The candidate family keeps QKFormer's native
signed carrier and treats ternary TX evidence as a polarity-consistency
modulator. This is more paper-friendly than calling it an external gate while
remaining compatible with all existing H9 experiments.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
BASE = GENERATED / "stride_h41_tx_s02c_v2.yml"


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def set_runtime(cfg: dict[str, Any]) -> None:
    loader = cfg.setdefault("loader", {})
    loader["batch_size"] = 8
    loader["n_workers"] = 8
    loader["pin_memory"] = True
    loader["persistent_workers"] = True
    loader["prefetch_factor"] = 4
    loader["non_blocking"] = True
    loader["n_epochs"] = 30
    runtime = cfg.setdefault("runtime", {})
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = [0, 4, 9, 14, 19, 24, 28, 29]
    runtime["use_mlflow_model_logging"] = False
    runtime["snn_backend"] = "cupy"


def set_attention(cfg: dict[str, Any], spec: dict[str, Any]) -> None:
    bsa = cfg.setdefault("bsa_attention", {})
    bsa["enabled"] = True
    bsa["mode"] = spec["mode"]
    bsa["center_scores"] = True
    bsa["preserve_mean"] = True
    bsa["alpha0"] = spec.get("alpha0", 0.02)
    bsa["mismatch_penalty"] = spec.get("mismatch_penalty", 0.25)
    bsa["single_active_penalty"] = spec.get("single_active_penalty", 0.0)
    bsa["single_active_penalty_grad"] = spec.get("single_active_penalty_grad", "ste")
    bsa["single_active_ste_slope"] = spec.get("single_active_ste_slope", 4.0)
    bsa["single_active_ste_margin"] = spec.get("single_active_ste_margin", 0.25)
    bsa["residual_alpha"] = spec.get("residual_alpha", 1.0)
    bsa["score_scale"] = spec.get("score_scale", 1.0)
    bsa["consensus_bias"] = spec.get("consensus_bias", 0.02)
    bsa["consensus_score_norm"] = "head_dim"
    bsa["value_mode"] = "threshold"


def set_lr(cfg: dict[str, Any], spec: dict[str, Any]) -> None:
    opt = cfg.setdefault("optimizer", {})
    groups = opt.setdefault("param_groups", {})
    strategy = spec.get("lr_strategy", "ntx01")
    if strategy == "ntx01":
        opt["lr"] = 2.0e-5
        opt["milestones"] = [22, 27]
        groups["backbone_lr"] = 1.0e-6
        groups["norm_lr"] = 1.0e-6
        groups["neuron_lr"] = 3.0e-5
        groups["threshold_lr"] = 5.0e-6
    elif strategy == "slowbb":
        opt["lr"] = 1.5e-5
        opt["milestones"] = [20, 25]
        groups["backbone_lr"] = 5.0e-7
        groups["norm_lr"] = 5.0e-7
        groups["neuron_lr"] = 2.4e-5
        groups["threshold_lr"] = 4.0e-6
    elif strategy == "warm":
        opt["lr"] = 2.0e-5
        opt["milestones"] = [22, 27]
        groups["backbone_lr"] = 1.0e-6
        groups["norm_lr"] = 1.0e-6
        groups["neuron_lr"] = 3.0e-5
        groups["threshold_lr"] = 5.0e-6
        opt["lr_warmup"] = {"enabled": True, "steps": 300, "start_factor": 0.1}
    else:
        raise ValueError(f"unknown lr_strategy: {strategy}")
    opt["use_amp"] = True
    cfg.setdefault("atlif_ternary_psn", {})["threshold_base_lr"] = groups["threshold_lr"]


def make_config(base: dict[str, Any], spec: dict[str, Any]) -> Path:
    cfg = deepcopy(base)
    cfg["experiment"] = spec["name"]
    cfg["note"] = spec["note"]
    set_runtime(cfg)
    set_attention(cfg, spec)
    set_lr(cfg, spec)
    out = GENERATED / f"{spec['name']}.yml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    base = read_yaml(BASE)
    specs: list[dict[str, Any]] = [
        {
            "name": "ntx04a_cptc_ntx01",
            "mode": "ternary_alpha_xnor_shiftmax",
            "mismatch_penalty": 0.25,
            "single_active_penalty": 0.0,
            "lr_strategy": "ntx01",
            "note": "NTX-04A: exact NTX-01 carrier-preserving ternary consistency attention, standard rerun control.",
        },
        {
            "name": "ntx04b_cptc_single005",
            "mode": "ternary_alpha_xnor_shiftmax",
            "mismatch_penalty": 0.25,
            "single_active_penalty": 0.05,
            "lr_strategy": "ntx01",
            "note": "NTX-04B: NTX-01 plus weak one-sided active/silent conflict penalty.",
        },
        {
            "name": "ntx04c_cptc_m04_single005",
            "mode": "ternary_alpha_xnor_shiftmax",
            "mismatch_penalty": 0.40,
            "single_active_penalty": 0.05,
            "lr_strategy": "ntx01",
            "note": "NTX-04C: NTX-01 with stronger opposite-polarity penalty and weak one-sided penalty.",
        },
        {
            "name": "ntx04d_cptc_res075",
            "mode": "ternary_alpha_xnor_shiftmax_residual",
            "residual_alpha": 0.75,
            "mismatch_penalty": 0.25,
            "single_active_penalty": 0.0,
            "lr_strategy": "ntx01",
            "note": "NTX-04D: residual carrier-preserving TX, alpha=0.75, reduces modulation strength.",
        },
        {
            "name": "ntx04e_cptc_res050",
            "mode": "ternary_alpha_xnor_shiftmax_residual",
            "residual_alpha": 0.50,
            "mismatch_penalty": 0.25,
            "single_active_penalty": 0.0,
            "lr_strategy": "ntx01",
            "note": "NTX-04E: residual carrier-preserving TX, alpha=0.50, precision-first ablation.",
        },
        {
            "name": "ntx04f_cptc_res075_single005",
            "mode": "ternary_alpha_xnor_shiftmax_residual",
            "residual_alpha": 0.75,
            "mismatch_penalty": 0.25,
            "single_active_penalty": 0.05,
            "lr_strategy": "ntx01",
            "note": "NTX-04F: residual carrier-preserving TX with weak active/silent penalty.",
        },
        {
            "name": "ntx04g_cptc_m04_single005_slowbb",
            "mode": "ternary_alpha_xnor_shiftmax",
            "mismatch_penalty": 0.40,
            "single_active_penalty": 0.05,
            "lr_strategy": "slowbb",
            "note": "NTX-04G: stronger consistency penalty with slower backbone LR to protect baseline weights.",
        },
        {
            "name": "ntx04h_cptc_ntx01_warm",
            "mode": "ternary_alpha_xnor_shiftmax",
            "mismatch_penalty": 0.25,
            "single_active_penalty": 0.0,
            "lr_strategy": "warm",
            "note": "NTX-04H: NTX-01 carrier-preserving attention with 300-step LR warmup.",
        },
    ]
    for spec in specs:
        print(make_config(base, spec).relative_to(EXP_ROOT / "configs"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
