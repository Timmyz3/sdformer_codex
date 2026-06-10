"""Generate NTX-07 partial token-selector configs.

NTX-05/06 showed that native QKV matrix replacement damages AAE. NTX-07 keeps
the attention linear in tokens and replaces only selected QKFormer token
selectors, avoiding the NTX-01 auxiliary multiply while also avoiding N x N QKV
mixing.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
BASE = GENERATED / "stride_h41_tx_s02c_v2.yml"
S2_BLOCKS = ["2:0", "2:1", "2:2", "2:3", "2:4", "2:5"]


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def make_config(base: dict[str, Any], spec: dict[str, Any]) -> Path:
    cfg = deepcopy(base)
    cfg["experiment"] = spec["name"]
    cfg["note"] = spec["note"]
    cfg.setdefault("loader", {})["batch_size"] = 8
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
            "target_blocks": S2_BLOCKS,
            "mode": spec["mode"],
            "center_scores": True,
            "preserve_mean": True,
            "alpha0": spec.get("alpha0", 0.02),
            "mismatch_penalty": spec.get("mismatch_penalty", 0.25),
            "single_active_penalty": spec.get("single_active_penalty", 0.05),
            "single_active_penalty_grad": "ste",
            "single_active_ste_slope": 4.0,
            "single_active_ste_margin": 0.25,
            "score_scale": 1.0,
            "consensus_bias": 0.02,
            "consensus_score_norm": spec.get("consensus_score_norm", "head_dim"),
            "value_mode": "threshold",
            "bipolar_mu": spec.get("bipolar_mu", 0.0),
            "bipolar_lambda": spec.get("bipolar_lambda", 0.5),
            "bipolar_gate_min": spec.get("bipolar_gate_min", -1.0),
            "bipolar_gate_max": spec.get("bipolar_gate_max", 1.8),
        }
    )

    out = GENERATED / f"{spec['name']}.yml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    base = read_yaml(BASE)
    specs = [
        {
            "name": "ntx07a_h49_qkselector_s2_m025_s005",
            "mode": "ternary_alpha_xnor_qkselector_shiftmax",
            "mismatch_penalty": 0.25,
            "single_active_penalty": 0.05,
            "note": "NTX-07A: partial stage2 H49 TX QK selector replacement; no NTX01 auxiliary multiply.",
        },
        {
            "name": "ntx07b_h49_qkselector_s2_m050_s005",
            "mode": "ternary_alpha_xnor_qkselector_shiftmax",
            "mismatch_penalty": 0.50,
            "single_active_penalty": 0.05,
            "note": "NTX-07B: partial stage2 H49 selector with stronger opposite-polarity penalty.",
        },
        {
            "name": "ntx07c_h54b_three_s2_mu025_l050",
            "mode": "tx_bipolar_qkselector_shiftmax",
            "mismatch_penalty": 0.25,
            "single_active_penalty": 0.05,
            "bipolar_mu": 0.25,
            "bipolar_lambda": 0.50,
            "note": "NTX-07C: partial stage2 three-score TX selector with mild signed correction.",
        },
        {
            "name": "ntx07d_h51_dual_s2_m050_s005",
            "mode": "dual_channel_qkselector_shiftmax",
            "mismatch_penalty": 0.50,
            "single_active_penalty": 0.05,
            "note": "NTX-07D: partial stage2 dual-channel excitation/inhibition selector.",
        },
    ]
    for spec in specs:
        print(make_config(base, spec).relative_to(EXP_ROOT / "configs"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
