"""Generate DATE-oriented full-replacement NTS11 ablation configs.

This fixes replacement scope to the full encoder attention set (all12) and
generates the mechanism matrix:

  neuron: PSN / all-binary ATLIF / all-ternary ATLIF
  attention: original / TX / SC / NTS(H60)

NB0 already covers PSN + original attention, so this script emits the other
11 full30 configs plus a manifest with expected module counts.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_nts11_two_neuron_only_configs import read_yaml, write_yaml
from make_nts11bd_unified_attn_sweep_configs import ALL12_BLOCKS, NB0, RECIPES, SN2Q_PATHS, apply_recipe


EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
BASE_CONFIG = EXP_ROOT / "configs/generated/nts11lite_u12_qkonly_w720_fastlr_full30.yml"
MANIFEST = GENERATED / "date11_full_factorial_manifest.json"


def _binary_group(name: str, *, paths: list[str] | None = None, path_selection: str = "") -> dict[str, Any]:
    group: dict[str, Any] = {
        "name": name,
        "output_mode": "binary",
        "threshold_mode": "official_atlif",
        "center_mode": "zero",
        "threshold_eta": 0.0,
        "activity_eta": 0.0,
        "target_rate": None,
        "target_rate_eta": 0.0,
    }
    if paths is not None:
        group["paths"] = list(paths)
    if path_selection:
        group["path_selection"] = path_selection
    return group


def _ternary_group(name: str, *, paths: list[str] | None = None, path_selection: str = "") -> dict[str, Any]:
    group: dict[str, Any] = {
        "name": name,
        "output_mode": "ternary",
        "threshold_mode": "symmetric_bsa_tsn",
        "center_mode": "bias",
        "threshold_eta": 6.5e-4,
        "threshold_lr_scale": 50000.0,
        "activity_eta": 0.0,
        "target_rate": None,
        "target_rate_eta": 0.0,
    }
    if paths is not None:
        group["paths"] = list(paths)
    if path_selection:
        group["path_selection"] = path_selection
    return group


def apply_neuron_policy(cfg: dict[str, Any], policy: str) -> dict[str, Any]:
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    if policy == "psn":
        atlif["enabled"] = False
        atlif["target"] = "none"
        atlif["target_groups"] = []
        return {"atlif_expected": 0, "ternary_expected": 0, "binary_expected": 0}

    if policy == "all_binary_atlif":
        atlif.update(
            {
                "enabled": True,
                "target": "qk",
                "stage_selection": "all",
                "output_mode": "binary",
                "threshold_mode": "official_atlif",
                "center_mode": "zero",
                "threshold_eta": 0.0,
                "activity_eta": 0.0,
                "target_rate": None,
                "target_rate_eta": 0.0,
            }
        )
        atlif.pop("target_paths", None)
        atlif["target_groups"] = [
            _binary_group("sn2q_binary", paths=SN2Q_PATHS),
            _binary_group("all_non_qk_binary_atlif", path_selection="all_non_qk"),
        ]
        return {"atlif_expected": 105, "ternary_expected": 0, "binary_expected": 105}

    if policy == "all_ternary_atlif":
        atlif.update(
            {
                "enabled": True,
                "target": "qk",
                "stage_selection": "all",
                "output_mode": "ternary",
                "threshold_mode": "symmetric_bsa_tsn",
                "center_mode": "bias",
                "threshold_eta": 6.5e-4,
                "threshold_lr_scale": 50000.0,
                "activity_eta": 0.0,
                "target_rate": None,
                "target_rate_eta": 0.0,
            }
        )
        atlif.pop("target_paths", None)
        atlif["target_groups"] = [
            _ternary_group("sn2q_ternary", paths=SN2Q_PATHS),
            _ternary_group("all_non_qk_ternary_atlif", path_selection="all_non_qk"),
        ]
        return {"atlif_expected": 105, "ternary_expected": 105, "binary_expected": 0}

    raise ValueError(f"unknown neuron policy: {policy}")


def apply_attention_policy(cfg: dict[str, Any], policy: str) -> dict[str, Any]:
    attn = cfg.setdefault("bsa_attention", {})
    if policy == "original":
        attn["enabled"] = False
        attn["target_blocks"] = []
        return {"shiftmax_expected": 0, "attention_mode": "original"}

    mode = {
        "tx": "ternary_alpha_xnor_shiftmax",
        "sc": "signed_consensus_shiftmax",
        "nts": "h60",
    }[policy]
    attn.update(
        {
            "enabled": True,
            "mode": mode,
            "center_scores": True,
            "preserve_mean": True,
            "alpha0": 0.02,
            "mismatch_penalty": 0.0,
            "single_active_penalty": 0.0,
            "score_scale": 1.0,
            "consensus_bias": 0.02,
            "consensus_score_norm": "head_dim",
            "eps": 1.0e-6,
            "relu_k_floor": 0.0,
            "value_mode": "threshold",
            "target_blocks": list(ALL12_BLOCKS),
            "bipolar_mu": 0.05,
            "bipolar_lambda": 0.5,
            "bipolar_gate_min": -1.0,
            "bipolar_gate_max": 1.8,
            "k_magnitude_alpha": 0.0,
            "sc_mu_schedule_enabled": policy == "nts",
            "sc_mu_start": 0.0,
            "sc_mu_start_step": 0,
            "sc_mu_warmup_steps": 720,
            "target_rate": None,
        }
    )
    attn.pop("stage_selection", None)
    return {"shiftmax_expected": 12, "attention_mode": mode}


def make_config(spec: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    cfg = deepcopy(read_yaml(BASE_CONFIG))
    cfg["experiment"] = spec["name"] + "_full30"
    cfg["note"] = spec["note"]

    neuron_meta = apply_neuron_policy(cfg, str(spec["neuron"]))
    attn_meta = apply_attention_policy(cfg, str(spec["attention"]))
    apply_recipe(cfg, RECIPES["w720_fastlr"])

    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 30
    loader["batch_size"] = 8
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = False
    runtime["force_save_epochs"] = [0, 4, 9, 14, 19, 24, 28, 29]
    runtime["use_mlflow_model_logging"] = False
    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    cfg.setdefault("test", {})["sample"] = 10

    out = GENERATED / f"{spec['name']}_full30.yml"
    write_yaml(out, cfg)
    manifest = {
        "priority": spec["priority"],
        "name": spec["name"],
        "config": str(out),
        "resume": str(NB0),
        "status": "config_generated_not_run",
        "neuron": spec["neuron"],
        "attention": spec["attention"],
        **neuron_meta,
        **attn_meta,
    }
    return out, manifest


def main() -> int:
    if not NB0.is_file():
        raise FileNotFoundError(f"missing NB0 checkpoint: {NB0}")
    if not BASE_CONFIG.is_file():
        raise FileNotFoundError(f"missing base config: {BASE_CONFIG}")

    specs: list[dict[str, Any]] = []
    priorities = {
        ("all_binary_atlif", "original"): "P0",
        ("all_ternary_atlif", "original"): "P0",
        ("all_ternary_atlif", "tx"): "P0",
        ("all_ternary_atlif", "sc"): "P0",
        ("all_ternary_atlif", "nts"): "P0",
        ("all_binary_atlif", "tx"): "P1",
        ("all_binary_atlif", "sc"): "P1",
        ("all_binary_atlif", "nts"): "P1",
        ("psn", "tx"): "P2",
        ("psn", "sc"): "P2",
        ("psn", "nts"): "P2",
    }
    for neuron, attention in priorities:
        specs.append(
            {
                "priority": priorities[(neuron, attention)],
                "neuron": neuron,
                "attention": attention,
                "name": f"date11full_{neuron}_{attention}_w720_fastlr",
                "note": (
                    "DATE full-factorial ablation: full all12 scope; "
                    f"neuron={neuron}; attention={attention}; warm720/fastlr/freeze1224."
                ),
            }
        )

    GENERATED.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "priority": "P0",
            "name": "NB0",
            "status": "done",
            "neuron": "psn",
            "attention": "original",
            "config": "configs/generated/upstream_baseline_stride.yml",
            "result": "AEE=1.4872 AAE=9.9300 total_spikes=44.0488G",
        }
    ]
    for spec in sorted(specs, key=lambda s: (s["priority"], s["name"])):
        out, row = make_config(spec)
        rows.append(row)
        print(out)
    MANIFEST.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
