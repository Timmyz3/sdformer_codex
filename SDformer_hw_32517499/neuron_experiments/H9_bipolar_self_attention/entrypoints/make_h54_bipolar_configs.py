"""Generate H54 bipolar selector sweep configs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXP_ROOT / "configs"
BASE_CONFIG = CONFIG_DIR / "generated/h53b_h49_clean_no_stage3_s02.yml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def apply_lr_strategy(config: dict[str, Any], strategy: str) -> None:
    optimizer = config.setdefault("optimizer", {})
    groups = optimizer.setdefault("param_groups", {})
    if strategy == "base":
        return
    if strategy == "fast":
        optimizer["lr"] = 1.5e-5
        groups["backbone_lr"] = 3.0e-7
        groups["norm_lr"] = 3.0e-7
        groups["neuron_lr"] = 2.0e-5
        groups["threshold_lr"] = 5.0e-6
        return
    if strategy == "warm":
        optimizer["lr"] = 1.2e-5
        groups["backbone_lr"] = 2.0e-7
        groups["norm_lr"] = 2.0e-7
        groups["neuron_lr"] = 1.2e-5
        groups["threshold_lr"] = 3.0e-6
        optimizer["lr_warmup"] = {
            "enabled": True,
            "steps": 100,
            "start_factor": 0.2,
        }
        return
    if strategy == "slowbb_warm":
        optimizer["lr"] = 1.5e-5
        groups["backbone_lr"] = 1.0e-7
        groups["norm_lr"] = 1.0e-7
        groups["neuron_lr"] = 1.8e-5
        groups["threshold_lr"] = 4.0e-6
        optimizer["lr_warmup"] = {
            "enabled": True,
            "steps": 120,
            "start_factor": 0.15,
        }
        return
    raise ValueError(f"Unknown H54 lr strategy: {strategy}")


def make_variant(base: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(base)
    config["experiment"] = spec["name"]
    config["note"] = spec["note"]
    bsa = config.setdefault("bsa_attention", {})
    bsa["mode"] = spec["mode"]
    bsa["bipolar_lambda"] = spec["bipolar_lambda"]
    bsa["bipolar_mu"] = spec.get("bipolar_mu", 0.0)
    bsa["single_active_penalty"] = spec["single_active_penalty"]
    bsa["single_active_penalty_grad"] = "ste"
    bsa["bipolar_gate_min"] = spec.get("bipolar_gate_min", -1.0)
    bsa["bipolar_gate_max"] = spec.get("bipolar_gate_max", 1.8)
    bsa["preserve_mean"] = True
    bsa["center_scores"] = True
    apply_lr_strategy(config, spec["lr_strategy"])
    config.setdefault("runtime", {})["max_train_steps"] = 720
    config.setdefault("runtime", {})["skip_state_save"] = True
    return config


def make_full_variant(short_config: dict[str, Any], name: str, note: str) -> dict[str, Any]:
    config = deepcopy(short_config)
    config["experiment"] = name
    config["note"] = note
    runtime = config.setdefault("runtime", {})
    runtime.pop("max_train_steps", None)
    runtime["force_save_epochs"] = list(range(30))
    runtime["skip_state_save"] = True
    loader = config.setdefault("loader", {})
    loader["n_epochs"] = 30
    return config


def main() -> int:
    base = load_yaml(BASE_CONFIG)
    specs = [
        {
            "name": "h54a_two_l03_g10_base",
            "mode": "bipolar_qkselector_shiftmax",
            "bipolar_lambda": 0.3,
            "single_active_penalty": 0.10,
            "lr_strategy": "base",
            "note": "H54a two-score bipolar selector, mild opposite branch, base LR.",
        },
        {
            "name": "h54a_two_l05_g10_base",
            "mode": "bipolar_qkselector_shiftmax",
            "bipolar_lambda": 0.5,
            "single_active_penalty": 0.10,
            "lr_strategy": "base",
            "note": "H54a two-score bipolar selector, balanced opposite branch, base LR.",
        },
        {
            "name": "h54a_two_l05_g20_base",
            "mode": "bipolar_qkselector_shiftmax",
            "bipolar_lambda": 0.5,
            "single_active_penalty": 0.20,
            "lr_strategy": "base",
            "note": "H54a two-score bipolar selector, stronger one-sided penalty, base LR.",
        },
        {
            "name": "h54a_two_l05_g20_fast",
            "mode": "bipolar_qkselector_shiftmax",
            "bipolar_lambda": 0.5,
            "single_active_penalty": 0.20,
            "lr_strategy": "fast",
            "note": "H54a two-score bipolar selector, stronger penalty, faster neuron/threshold LR.",
        },
        {
            "name": "h54a_two_l08_g10_fast",
            "mode": "bipolar_qkselector_shiftmax",
            "bipolar_lambda": 0.8,
            "single_active_penalty": 0.10,
            "lr_strategy": "fast",
            "note": "H54a two-score bipolar selector, strong opposite branch, faster LR.",
        },
        {
            "name": "h54b_three_mu03_l08_g10_base",
            "mode": "tx_bipolar_qkselector_shiftmax",
            "bipolar_mu": 0.3,
            "bipolar_lambda": 0.8,
            "single_active_penalty": 0.10,
            "lr_strategy": "base",
            "note": "H54b three-score TX plus mild bipolar correction, base LR.",
        },
        {
            "name": "h54b_three_mu05_l08_g10_base",
            "mode": "tx_bipolar_qkselector_shiftmax",
            "bipolar_mu": 0.5,
            "bipolar_lambda": 0.8,
            "single_active_penalty": 0.10,
            "lr_strategy": "base",
            "note": "H54b three-score TX plus balanced bipolar correction, base LR.",
        },
        {
            "name": "h54b_three_mu05_l10_g10_base",
            "mode": "tx_bipolar_qkselector_shiftmax",
            "bipolar_mu": 0.5,
            "bipolar_lambda": 1.0,
            "single_active_penalty": 0.10,
            "lr_strategy": "base",
            "note": "H54b three-score TX plus stronger opposite correction, base LR.",
        },
        {
            "name": "h54b_three_mu05_l08_g20_fast",
            "mode": "tx_bipolar_qkselector_shiftmax",
            "bipolar_mu": 0.5,
            "bipolar_lambda": 0.8,
            "single_active_penalty": 0.20,
            "lr_strategy": "fast",
            "note": "H54b three-score TX plus balanced correction, stronger penalty, faster LR.",
        },
        {
            "name": "h54b_three_mu07_l08_g10_fast",
            "mode": "tx_bipolar_qkselector_shiftmax",
            "bipolar_mu": 0.7,
            "bipolar_lambda": 0.8,
            "single_active_penalty": 0.10,
            "lr_strategy": "fast",
            "note": "H54b three-score TX plus strong correction, faster LR.",
        },
        {
            "name": "h54b_three_mu02_l05_g10_warm",
            "mode": "tx_bipolar_qkselector_shiftmax",
            "bipolar_mu": 0.2,
            "bipolar_lambda": 0.5,
            "single_active_penalty": 0.10,
            "lr_strategy": "warm",
            "note": "H54b three-score TX plus gentle bipolar correction with warmup.",
        },
        {
            "name": "h54b_three_mu03_l05_g15_warm",
            "mode": "tx_bipolar_qkselector_shiftmax",
            "bipolar_mu": 0.3,
            "bipolar_lambda": 0.5,
            "single_active_penalty": 0.15,
            "lr_strategy": "warm",
            "note": "H54b three-score TX plus moderate bipolar correction and warmup.",
        },
        {
            "name": "h54b_three_mu03_l08_g15_slowbb_warm",
            "mode": "tx_bipolar_qkselector_shiftmax",
            "bipolar_mu": 0.3,
            "bipolar_lambda": 0.8,
            "single_active_penalty": 0.15,
            "lr_strategy": "slowbb_warm",
            "note": "H54b three-score with slow backbone warmup to protect baseline continuation.",
        },
        {
            "name": "h54a_two_l03_g15_warm",
            "mode": "bipolar_qkselector_shiftmax",
            "bipolar_lambda": 0.3,
            "single_active_penalty": 0.15,
            "lr_strategy": "warm",
            "note": "H54a two-score signed selector with warmup, testing whether pure signed gate can be stabilized.",
        },
    ]
    for spec in specs:
        path = CONFIG_DIR / "generated" / f"{spec['name']}.yml"
        config = make_variant(base, spec)
        dump_yaml(path, config)
        print(path.relative_to(CONFIG_DIR))
        if spec["name"] == "h54b_three_mu05_l08_g20_fast":
            full_path = CONFIG_DIR / "generated" / "h54b_three_mu05_l08_g20_fast_full30.yml"
            dump_yaml(
                full_path,
                make_full_variant(
                    config,
                    "h54b_three_mu05_l08_g20_fast_full30",
                    "H54b full30 main candidate: TX + bipolar three-score selector, strong one-sided penalty, fast differential LR.",
                ),
            )
            print(full_path.relative_to(CONFIG_DIR))
        if spec["name"] == "h54a_two_l03_g15_warm":
            full_path = CONFIG_DIR / "generated" / "h54a_two_l03_g15_warm_full30.yml"
            dump_yaml(
                full_path,
                make_full_variant(
                    config,
                    "h54a_two_l03_g15_warm_full30",
                    "H54a full30 backup: pure two-score signed selector with warmup.",
                ),
            )
            print(full_path.relative_to(CONFIG_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
