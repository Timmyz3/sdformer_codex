"""Generate NTX-06 partial native-attention configs.

NTX-05 showed that replacing every attention block with native QKV attention
hurts AAE badly. NTX-06 keeps the same non-auxiliary-attention story, but limits
native attention to selected stages so the decoder still receives familiar
low-level carriers from the untouched blocks.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
BASE = GENERATED / "stride_h41_tx_s02c_v2.yml"


STAGE_BLOCKS = {
    "s0": ("0:0", "0:1"),
    "s1": ("1:0", "1:1"),
    "s2": ("2:0", "2:1", "2:2", "2:3", "2:4", "2:5"),
    "s3": ("3:0", "3:1"),
}


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def blocks(*stages: str) -> list[str]:
    selected: list[str] = []
    for stage in stages:
        selected.extend(STAGE_BLOCKS[stage])
    return selected


def make_config(base: dict[str, Any], spec: dict[str, Any]) -> Path:
    cfg = deepcopy(base)
    cfg["experiment"] = spec["name"]
    cfg["note"] = spec["note"]
    cfg.setdefault("loader", {})["batch_size"] = 6
    cfg["loader"]["n_epochs"] = 30
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 360
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = [0]
    runtime["use_mlflow_model_logging"] = False

    attn = cfg.setdefault("bsa_attention", {})
    attn.update(
        {
            "enabled": True,
            "stage_selection": "all",
            "target_blocks": spec["target_blocks"],
            "mode": spec["mode"],
            "center_scores": True,
            "preserve_mean": spec.get("preserve_mean", True),
            "alpha0": spec.get("alpha0", 0.02),
            "mismatch_penalty": spec.get("mismatch_penalty", 0.25),
            "single_active_penalty": spec.get("single_active_penalty", 0.0),
            "single_active_penalty_grad": "ste",
            "single_active_ste_slope": 4.0,
            "single_active_ste_margin": 0.25,
            "score_scale": spec.get("score_scale", 1.0),
            "consensus_bias": spec.get("consensus_bias", 0.0),
            "consensus_score_norm": spec.get("consensus_score_norm", "head_dim"),
            "value_mode": spec.get("value_mode", "threshold"),
            "value_branch": "independent",
            "value_init": "copy_k",
            "relu_k_floor": spec.get("relu_k_floor", 0.0),
        }
    )

    out = GENERATED / f"{spec['name']}.yml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    base = read_yaml(BASE)
    specs = [
        {
            "name": "ntx06a_a2os2a_qkv_theta_s2",
            "mode": "a2os2a_qkv_l1",
            "target_blocks": blocks("s2"),
            "note": "NTX-06A: partial native A2OS2A-QKV on stage2 only; no auxiliary NTX01 gate.",
        },
        {
            "name": "ntx06b_a2os2a_qkv_theta_s0s2",
            "mode": "a2os2a_qkv_l1",
            "target_blocks": blocks("s0", "s2"),
            "note": "NTX-06B: partial native A2OS2A-QKV on stage0+stage2; no auxiliary NTX01 gate.",
        },
        {
            "name": "ntx06c_tx_qkv_shiftmax_theta_s2",
            "mode": "ternary_alpha_xnor_ssa_qkv_shiftmax",
            "target_blocks": blocks("s2"),
            "consensus_bias": 0.02,
            "note": "NTX-06C: partial native ternary alpha-XNOR QKV Shiftmax on stage2 only.",
        },
        {
            "name": "ntx06d_strict_bsa_qkv_signv_s2",
            "mode": "strict_bsa_qkv_shiftmax",
            "target_blocks": blocks("s2"),
            "value_mode": "sign",
            "consensus_score_norm": "sqrt_head_dim",
            "consensus_bias": 0.02,
            "note": "NTX-06D: partial strict BSA-QKV sign-V on stage2 only.",
        },
    ]
    for spec in specs:
        print(make_config(base, spec).relative_to(EXP_ROOT / "configs"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
