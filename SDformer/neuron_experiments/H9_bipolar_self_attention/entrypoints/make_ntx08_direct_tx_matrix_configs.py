"""Generate NTX-08 direct TX matrix attention screening configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
BASE = GENERATED / "stride_h41_tx_s02c_v2.yml"
S1_BLOCKS = ["1:0", "1:1"]
S2_BLOCKS = ["2:0", "2:1", "2:2", "2:3", "2:4", "2:5"]


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def blocks_for(scope: str) -> list[str]:
    if scope == "s2":
        return S2_BLOCKS
    if scope == "s12":
        return S1_BLOCKS + S2_BLOCKS
    if scope == "all":
        return []
    raise ValueError(f"unknown scope: {scope}")


def set_common(cfg: dict[str, Any], spec: dict[str, Any]) -> None:
    cfg["experiment"] = spec["name"]
    cfg.setdefault("test", {})["scale_factor"] = 1

    loader = cfg.setdefault("loader", {})
    loader["batch_size"] = 8
    loader["n_workers"] = 8
    loader["pin_memory"] = True
    loader["persistent_workers"] = True
    loader["prefetch_factor"] = 4
    loader["non_blocking"] = True

    runtime = cfg.setdefault("runtime", {})
    runtime["skip_state_save"] = True
    runtime["use_mlflow_model_logging"] = False
    runtime["snn_backend"] = "cupy"

    attn = cfg.setdefault("bsa_attention", {})
    attn["enabled"] = True
    attn["mode"] = spec["mode"]
    attn["center_scores"] = True
    attn["preserve_mean"] = True
    attn["alpha0"] = 0.02
    attn["mismatch_penalty"] = float(spec["beta"])
    attn["single_active_penalty"] = float(spec["gamma"])
    attn["single_active_penalty_grad"] = "ste"
    attn["single_active_ste_slope"] = 4.0
    attn["single_active_ste_margin"] = 0.25
    attn["score_scale"] = float(spec.get("score_scale", 1.0))
    attn["consensus_bias"] = 0.02
    attn["consensus_score_norm"] = "head_dim"
    attn["matrix_diag_bias"] = float(spec.get("diag_bias", 0.5))
    attn["value_mode"] = str(spec.get("value_mode", "threshold"))
    attn["value_branch"] = str(spec.get("value_branch", "reuse_k"))
    attn["value_init"] = "copy_k"

    target_blocks = blocks_for(str(spec["scope"]))
    if target_blocks:
        attn["target_blocks"] = target_blocks
        attn.pop("stage_selection", None)
    else:
        attn.pop("target_blocks", None)
        attn["stage_selection"] = "all"

    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif["target_rate"] = None
    atlif["target_rate_eta"] = 0.0
    atlif["activity_eta"] = 0.0
    atlif["threshold_base_lr"] = 5.0e-6

    opt = cfg.setdefault("optimizer", {})
    opt["lr"] = float(spec.get("lr", 2.0e-5))
    opt["milestones"] = [22, 27]
    opt["lr_warmup"] = {"enabled": True, "steps": 450, "start_factor": 0.05}
    groups = opt.setdefault("param_groups", {})
    groups["enabled"] = True
    groups["backbone_lr"] = 1.0e-6
    groups["norm_lr"] = 1.0e-6
    groups["neuron_lr"] = 3.0e-5
    groups["threshold_lr"] = 5.0e-6

    cfg["note"] = (
        "NTX-08 direct TX matrix attention: score_ij -> Shiftmax -> matmul(V), no carrier*gate. "
        f"name={spec['name']}; mode={attn['mode']}; scope={spec['scope']}; beta={attn['mismatch_penalty']}; "
        f"gamma={attn['single_active_penalty']}; diag_bias={attn['matrix_diag_bias']}; "
        f"value_mode={attn['value_mode']}; value_branch={attn['value_branch']}."
    )


def main() -> int:
    base = read_yaml(BASE)
    specs: list[dict[str, Any]] = [
        {
            "name": "ntx08a_direct_tx_s2_b025_g005_d05_kv",
            "mode": "ternary_alpha_xnor_ssa_kreuse_shiftmax",
            "scope": "s2",
            "beta": 0.25,
            "gamma": 0.05,
            "diag_bias": 0.5,
        },
        {
            "name": "ntx08b_direct_tx_s2_b035_g005_d05_kv",
            "mode": "ternary_alpha_xnor_ssa_kreuse_shiftmax",
            "scope": "s2",
            "beta": 0.35,
            "gamma": 0.05,
            "diag_bias": 0.5,
        },
        {
            "name": "ntx08c_direct_tx_s2_b035_g008_d10_kv",
            "mode": "ternary_alpha_xnor_ssa_kreuse_shiftmax",
            "scope": "s2",
            "beta": 0.35,
            "gamma": 0.08,
            "diag_bias": 1.0,
        },
        {
            "name": "ntx08d_direct_tx_s12_b025_g005_d05_kv",
            "mode": "ternary_alpha_xnor_ssa_kreuse_shiftmax",
            "scope": "s12",
            "beta": 0.25,
            "gamma": 0.05,
            "diag_bias": 0.5,
        },
        {
            "name": "ntx08e_direct_tx_s2_b025_g005_d05_qkv",
            "mode": "ternary_alpha_xnor_ssa_qkv_shiftmax",
            "scope": "s2",
            "beta": 0.25,
            "gamma": 0.05,
            "diag_bias": 0.5,
            "value_branch": "independent",
        },
    ]

    generated: list[str] = []
    for spec in specs:
        cfg = deepcopy(base)
        set_common(cfg, spec)

        short_cfg = deepcopy(cfg)
        short_cfg["loader"]["n_epochs"] = 1
        short_cfg["runtime"]["max_train_steps"] = 360
        short_cfg["runtime"]["force_save_epochs"] = [0]
        short = GENERATED / f"{spec['name']}_steps360.yml"
        write_yaml(short, short_cfg)
        generated.append(f"generated/{short.name}")

        full_cfg = deepcopy(cfg)
        full_cfg["loader"]["n_epochs"] = 30
        full_cfg["runtime"]["max_train_steps"] = 0
        full_cfg["runtime"]["force_save_epochs"] = list(range(30))
        full = GENERATED / f"{spec['name']}_full30.yml"
        write_yaml(full, full_cfg)
        generated.append(f"generated/{full.name}")

    print("\n".join(generated))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
