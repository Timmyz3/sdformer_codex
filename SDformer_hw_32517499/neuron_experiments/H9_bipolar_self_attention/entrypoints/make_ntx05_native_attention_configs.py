"""Generate NTX-05 native-attention configs.

NTX-05 deliberately avoids the NTX-01/H41 auxiliary gate pattern.  It starts
from the current standard TX-v2 fine-tune recipe only to keep the neuron and
optimizer protocol fixed, then replaces attention with native QK/QKV variants.
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


def set_common_runtime(cfg: dict[str, Any]) -> None:
    cfg.setdefault("loader", {})["n_epochs"] = 30
    cfg["loader"]["batch_size"] = 4
    runtime = cfg.setdefault("runtime", {})
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = [0, 4, 9, 14, 19, 24, 28, 29]
    runtime["use_mlflow_model_logging"] = False


def make_config(base: dict[str, Any], spec: dict[str, Any]) -> Path:
    cfg = deepcopy(base)
    cfg["experiment"] = spec["name"]
    cfg["note"] = spec["note"]
    set_common_runtime(cfg)

    attn = cfg.setdefault("bsa_attention", {})
    attn.update(
        {
            "enabled": True,
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
            "consensus_bias": spec.get("consensus_bias", 0.02),
            "consensus_score_norm": spec.get("consensus_score_norm", "head_dim"),
            "value_mode": spec.get("value_mode", "threshold"),
            "value_branch": spec.get("value_branch", "independent"),
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
            "name": "ntx05a_qkv_tx_shiftmax_theta",
            "mode": "ternary_alpha_xnor_ssa_qkv_shiftmax",
            "value_mode": "threshold",
            "note": "NTX-05A: native ternary alpha-XNOR QK matrix + independent threshold V + Shiftmax.",
        },
        {
            "name": "ntx05b_qkv_tx_shiftmax_signv",
            "mode": "ternary_alpha_xnor_ssa_qkv_shiftmax",
            "value_mode": "sign",
            "note": "NTX-05B: native ternary alpha-XNOR QK matrix + independent ternary-sign V + Shiftmax.",
        },
        {
            "name": "ntx05c_strict_bsa_qkv_theta",
            "mode": "strict_bsa_qkv_shiftmax",
            "value_mode": "threshold",
            "consensus_score_norm": "sqrt_head_dim",
            "note": "NTX-05C: strict ternary BSA QK^T + independent threshold V + Shiftmax.",
        },
        {
            "name": "ntx05d_strict_bsa_qkv_signv",
            "mode": "strict_bsa_qkv_shiftmax",
            "value_mode": "sign",
            "consensus_score_norm": "sqrt_head_dim",
            "note": "NTX-05D: strict ternary BSA QK^T + independent sign V + Shiftmax.",
        },
        {
            "name": "ntx05e_a2os2a_qkv_theta",
            "mode": "a2os2a_qkv_l1",
            "value_mode": "threshold",
            "consensus_bias": 0.0,
            "note": "NTX-05E: A2OS2A-style binary-Q nonnegative-K + independent threshold V + L1.",
        },
        {
            "name": "ntx05f_a2os2a_qkv_signv",
            "mode": "a2os2a_qkv_l1",
            "value_mode": "sign",
            "consensus_bias": 0.0,
            "note": "NTX-05F: A2OS2A-style binary-Q nonnegative-K + independent sign V + L1.",
        },
    ]
    for spec in specs:
        print(make_config(base, spec).relative_to(EXP_ROOT / "configs"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
