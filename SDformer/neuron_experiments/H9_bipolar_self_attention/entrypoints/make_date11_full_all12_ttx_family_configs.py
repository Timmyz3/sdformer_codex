"""Generate full-all12 pure-TX BTTX hardware-family screening configs.

Every candidate keeps the same no-carrier H60 selector topology and changes
only the ternary neuron threshold policy. Partial stage/block replacement and
TX/SC score mixing are intentionally unsupported.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
GEN = EXP_ROOT / "configs/generated"
BASE = GEN / "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8.yml"
MANIFEST = GEN / "date11_full_all12_ttx_family_manifest.json"

TTX_RESUME = Path(
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/"
    "checkpoint_epoch2.pth"
)

ALL12_BLOCKS = [
    "0:0", "0:1", "1:0", "1:1", "2:0", "2:1",
    "2:2", "2:3", "2:4", "2:5", "3:0", "3:1",
]


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def set_short_runtime(cfg: dict[str, Any]) -> None:
    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 1
    loader["batch_size"] = 8
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 360
    runtime["skip_state_save"] = False
    runtime["force_save_epochs"] = [0]
    runtime["use_mlflow_model_logging"] = False


def set_event_alphabet(
    cfg: dict[str, Any], output_mode: str, *, negative_threshold_scale: float = 1.0
) -> None:
    neuron = cfg.setdefault("atlif_ternary_psn", {})
    neuron["output_mode"] = output_mode
    if output_mode == "binary":
        threshold_mode = "official_atlif"
        center_mode = "zero"
    elif output_mode == "ternary":
        threshold_mode = (
            "symmetric_bsa_tsn"
            if float(negative_threshold_scale) == 1.0
            else "asymmetric_scale"
        )
        center_mode = "zero"
    else:
        raise ValueError(f"unsupported output mode: {output_mode}")

    neuron["threshold_mode"] = threshold_mode
    neuron["center_mode"] = center_mode
    neuron["negative_threshold_scale"] = float(negative_threshold_scale)
    neuron["target"] = "qk"
    neuron["stage_selection"] = "all"
    neuron["target_rate"] = None
    neuron["target_rate_eta"] = 0.0
    neuron["activity_eta"] = 0.0
    neuron["threshold_eta"] = 0.0

    groups = neuron.get("target_groups", [])
    if len(groups) != 2:
        raise ValueError("TTX base config must contain sn2_q and all_non_qk target groups")
    groups[0]["name"] = f"sn2q_{output_mode}"
    groups[1]["name"] = f"all_non_qk_{output_mode}_atlif"
    for group in groups:
        group["output_mode"] = output_mode
        group["threshold_mode"] = threshold_mode
        group["center_mode"] = center_mode
        group["negative_threshold_scale"] = float(negative_threshold_scale)
        group["target_rate"] = None
        group["target_rate_eta"] = 0.0
        group["activity_eta"] = 0.0
        group["threshold_eta"] = 0.0


def set_score(
    cfg: dict[str, Any], *, alpha0: float, mu: float, hardware_quant: bool = False
) -> None:
    attention = cfg.setdefault("bsa_attention", {})
    attention.update(
        {
            "enabled": True,
            "mode": "h60",
            "target_blocks": list(ALL12_BLOCKS),
            "center_scores": True,
            "preserve_mean": True,
            "alpha0": float(alpha0),
            "mismatch_penalty": 0.0,
            "single_active_penalty": 0.0,
            "bipolar_mu": float(mu),
            "k_magnitude_alpha": 0.0,
            "sc_mu_schedule_enabled": False,
            "sc_mu_start": 0.0,
            "sc_mu_start_step": 0,
            "sc_mu_warmup_steps": 0,
            "target_rate": None,
            "hardware_quant_enabled": bool(hardware_quant),
        }
    )
    if hardware_quant:
        attention.update(
            {
                "hardware_mu_pow2_shift": 4 if float(mu) != 0.0 else 0,
                "hardware_score_step": 1.0 / 128.0,
                "hardware_score_min": -2.0,
                "hardware_score_max": 2.0,
                "hardware_gate_step": 1.0 / 128.0,
                "hardware_gate_min": 0.0,
                "hardware_gate_max": 2.0,
            }
        )


def make_candidate(
    base: dict[str, Any],
    *,
    name: str,
    output_mode: str,
    alpha0: float,
    mu: float,
    score_formula: str,
    hardware_quant: bool = False,
    negative_threshold_scale: float = 1.0,
) -> tuple[Path, dict[str, Any]]:
    cfg = deepcopy(base)
    cfg["experiment"] = name
    set_short_runtime(cfg)
    set_event_alphabet(
        cfg, output_mode, negative_threshold_scale=negative_threshold_scale
    )
    set_score(cfg, alpha0=alpha0, mu=mu, hardware_quant=hardware_quant)
    cfg["note"] = (
        "DATE11 full-network all12 TTX hardware-family screen. "
        f"output={output_mode}; no carrier/no Kmag/no target-rate; {score_formula}. "
        "Warm-start from DSEC TTX epoch2; no partial replacement."
    )
    path = GEN / f"{name}.yml"
    write_yaml(path, cfg)
    return path, {
        "name": name,
        "config": str(path),
        "resume": str(TTX_RESUME),
        "scope": "full-network all12",
        "output_mode": output_mode,
        "attention_mode": "h60",
        "alpha0": alpha0,
        "bipolar_mu": mu,
        "hardware_quant": hardware_quant,
        "negative_threshold_scale": negative_threshold_scale,
        "score_formula": score_formula,
        "atlif_expected": 105,
        "shiftmax_expected": 12,
        "steps": 360,
        "status": "config_generated_not_run",
    }


def main() -> int:
    if not BASE.is_file():
        raise FileNotFoundError(f"missing base config: {BASE}")
    base = read_yaml(BASE)
    specs = [
        {
            "name": "date11full_ttx_dyadic_txonly_all12_deploy_int8",
            "output_mode": "binary",
            "alpha0": 1.0 / 64.0,
            "mu": 0.0,
            "hardware_quant": True,
            "score_formula": "pure integer TX-only score=64*same+1*zero with int8 score/gate",
        },
        {
            "name": "date11full_bttx_txonly_all12_s360",
            "output_mode": "ternary",
            "alpha0": 0.02,
            "mu": 0.0,
            "score_formula": "exact BTTX TX-only score=same+0.02*zero",
        },
        {
            "name": "date11full_bttx_a4_txonly_all12_s360",
            "output_mode": "ternary",
            "alpha0": 0.02,
            "mu": 0.0,
            "negative_threshold_scale": 4.0,
            "score_formula": "pure TX-only score=same+0.02*zero; negative threshold=-4*theta",
        },
    ]

    rows: list[dict[str, Any]] = []
    for spec in specs:
        path, row = make_candidate(base, **spec)
        rows.append(row)
        print(path)
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
