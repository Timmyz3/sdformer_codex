"""Generate NTX-03 TX refinement configs from the standard stride NTX-02 base.

NTX-03 keeps the corrected H49/H53 replacement scope and baseline-epoch59
continuation protocol. It only changes the TX selector scoring family and its
penalty calibration, so short-test results are comparable with NTX-02.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
BASE = GENERATED / "stride_h53b_h49_clean_no_stage3_s02_full30.yml"


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def set_runtime(cfg: dict[str, Any]) -> None:
    cfg.setdefault("loader", {})["n_epochs"] = 30
    runtime = cfg.setdefault("runtime", {})
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = [0, 4, 9, 14, 19, 24, 29]
    runtime["use_mlflow_model_logging"] = False


def set_attention(cfg: dict[str, Any], spec: dict[str, Any]) -> None:
    attn = cfg.setdefault("bsa_attention", {})
    attn["enabled"] = True
    attn["mode"] = spec["mode"]
    attn["alpha0"] = spec.get("alpha0", 0.02)
    attn["mismatch_penalty"] = spec.get("mismatch_penalty", 0.25)
    attn["single_active_penalty"] = spec.get("single_active_penalty", 0.0)
    attn["single_active_penalty_grad"] = spec.get("single_active_penalty_grad", "ste")
    attn["single_active_ste_slope"] = spec.get("single_active_ste_slope", 4.0)
    attn["single_active_ste_margin"] = spec.get("single_active_ste_margin", 0.25)
    attn["bipolar_lambda"] = spec.get("bipolar_lambda", 0.0)
    attn["bipolar_mu"] = spec.get("bipolar_mu", 0.0)
    attn["bipolar_gate_min"] = spec.get("bipolar_gate_min", -1.0)
    attn["bipolar_gate_max"] = spec.get("bipolar_gate_max", 1.8)
    attn["center_scores"] = True
    attn["preserve_mean"] = True
    attn["consensus_score_norm"] = "head_dim"
    attn["score_scale"] = 1.0
    attn["value_mode"] = "threshold"


def make_config(base: dict[str, Any], spec: dict[str, Any]) -> Path:
    cfg = deepcopy(base)
    cfg["experiment"] = spec["name"]
    cfg["note"] = spec["note"]
    set_runtime(cfg)
    set_attention(cfg, spec)
    out = GENERATED / f"{spec['name']}.yml"
    write_yaml(out, cfg)
    return out


def main() -> int:
    base = read_yaml(BASE)
    specs: list[dict[str, Any]] = [
        # A: keep H49 token TX selector; recalibrate one-sided vs opposite penalties.
        {
            "name": "ntx03a_tx_m04_s005",
            "mode": "ternary_alpha_xnor_qkselector_shiftmax",
            "mismatch_penalty": 0.40,
            "single_active_penalty": 0.05,
            "note": "NTX-03A: H49 token TX selector, weaker one-sided penalty and stronger opposite-polarity penalty.",
        },
        {
            "name": "ntx03a_tx_m04_s010",
            "mode": "ternary_alpha_xnor_qkselector_shiftmax",
            "mismatch_penalty": 0.40,
            "single_active_penalty": 0.10,
            "note": "NTX-03A: H49 token TX selector, moderate one-sided penalty and stronger opposite-polarity penalty.",
        },
        {
            "name": "ntx03a_tx_m06_s005",
            "mode": "ternary_alpha_xnor_qkselector_shiftmax",
            "mismatch_penalty": 0.60,
            "single_active_penalty": 0.05,
            "note": "NTX-03A: H49 token TX selector, strong opposite-polarity penalty, weak one-sided penalty.",
        },
        {
            "name": "ntx03a_tx_m06_s010",
            "mode": "ternary_alpha_xnor_qkselector_shiftmax",
            "mismatch_penalty": 0.60,
            "single_active_penalty": 0.10,
            "note": "NTX-03A: H49 token TX selector, strong opposite-polarity penalty and moderate one-sided penalty.",
        },
        {
            "name": "ntx03a_tx_m08_s005",
            "mode": "ternary_alpha_xnor_qkselector_shiftmax",
            "mismatch_penalty": 0.80,
            "single_active_penalty": 0.05,
            "note": "NTX-03A: H49 token TX selector, very strong opposite-polarity penalty, weak one-sided penalty.",
        },
        {
            "name": "ntx03a_tx_m06_s005_hardactive",
            "mode": "ternary_alpha_xnor_qkselector_shiftmax",
            "mismatch_penalty": 0.60,
            "single_active_penalty": 0.05,
            "single_active_penalty_grad": "hard",
            "note": "NTX-03A: same as m06/s005 but one-sided activity mask uses hard/no-proxy gradient.",
        },
        # B: split same/opposite TX evidence so the final selector can become signed.
        {
            "name": "ntx03b_two_l025_s005",
            "mode": "bipolar_qkselector_shiftmax",
            "bipolar_lambda": 0.25,
            "single_active_penalty": 0.05,
            "note": "NTX-03B: two-branch same/opposite TX selector, mild negative branch.",
        },
        {
            "name": "ntx03b_two_l050_s005",
            "mode": "bipolar_qkselector_shiftmax",
            "bipolar_lambda": 0.50,
            "single_active_penalty": 0.05,
            "note": "NTX-03B: two-branch same/opposite TX selector, balanced negative branch.",
        },
        {
            "name": "ntx03b_two_l050_s010",
            "mode": "bipolar_qkselector_shiftmax",
            "bipolar_lambda": 0.50,
            "single_active_penalty": 0.10,
            "note": "NTX-03B: two-branch TX selector, balanced negative branch, moderate one-sided evidence.",
        },
        {
            "name": "ntx03b_two_l075_s005",
            "mode": "bipolar_qkselector_shiftmax",
            "bipolar_lambda": 0.75,
            "single_active_penalty": 0.05,
            "note": "NTX-03B: two-branch TX selector, strong negative branch.",
        },
        # C: keep normal TX gate as stable carrier, add signed same/opposite correction.
        {
            "name": "ntx03c_three_mu025_l050_s005",
            "mode": "tx_bipolar_qkselector_shiftmax",
            "bipolar_mu": 0.25,
            "bipolar_lambda": 0.50,
            "single_active_penalty": 0.05,
            "note": "NTX-03C: three-branch TX selector, gentle signed correction.",
        },
        {
            "name": "ntx03c_three_mu050_l050_s005",
            "mode": "tx_bipolar_qkselector_shiftmax",
            "bipolar_mu": 0.50,
            "bipolar_lambda": 0.50,
            "single_active_penalty": 0.05,
            "note": "NTX-03C: three-branch TX selector, balanced signed correction.",
        },
        {
            "name": "ntx03c_three_mu050_l075_s005",
            "mode": "tx_bipolar_qkselector_shiftmax",
            "bipolar_mu": 0.50,
            "bipolar_lambda": 0.75,
            "single_active_penalty": 0.05,
            "note": "NTX-03C: three-branch TX selector, stronger opposite branch.",
        },
        {
            "name": "ntx03c_three_mu050_l050_s010",
            "mode": "tx_bipolar_qkselector_shiftmax",
            "bipolar_mu": 0.50,
            "bipolar_lambda": 0.50,
            "single_active_penalty": 0.10,
            "note": "NTX-03C: three-branch TX selector, balanced correction and moderate one-sided evidence.",
        },
    ]
    for spec in specs:
        print(make_config(base, spec).relative_to(EXP_ROOT / "configs"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
