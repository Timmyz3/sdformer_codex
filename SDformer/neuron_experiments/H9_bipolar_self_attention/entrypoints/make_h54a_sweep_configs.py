"""Generate H54a bipolar-lambda sweep + LR strategy short-test configs.

H54a = bipolar_qkselector_shiftmax (two-score signed gate).
The opposite-penalty lever is bipolar_lambda (lambda) in:
    gate = Shiftmax(same) - lambda * Shiftmax(opp)

This script sweeps lambda x LR strategy for 360-step short tests.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

EXP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXP_ROOT / "configs"
BASE_CONFIG = CONFIG_DIR / "generated/h54a_two_l03_g15_warm.yml"


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
    groups["enabled"] = True
    # remove warmup from previous strategy
    optimizer.pop("lr_warmup", None)

    if strategy == "fast_warm":
        optimizer["lr"] = 1.2e-5
        groups["backbone_lr"] = 2.0e-7
        groups["norm_lr"] = 2.0e-7
        groups["neuron_lr"] = 1.2e-5
        groups["threshold_lr"] = 3.0e-6
        optimizer["lr_warmup"] = {"enabled": True, "steps": 100, "start_factor": 0.2}
    elif strategy == "slowbb":
        optimizer["lr"] = 1.0e-5
        groups["backbone_lr"] = 2.0e-7
        groups["norm_lr"] = 2.0e-7
        groups["neuron_lr"] = 1.2e-5
        groups["threshold_lr"] = 3.0e-6
    elif strategy == "midbb":
        optimizer["lr"] = 1.5e-5
        groups["backbone_lr"] = 3.0e-7
        groups["norm_lr"] = 3.0e-7
        groups["neuron_lr"] = 2.0e-5
        groups["threshold_lr"] = 5.0e-6
    else:
        raise ValueError(f"unknown LR strategy: {strategy}")


def make_short_config(
    base: dict[str, Any],
    lam: float,
    lr_strategy: str,
    steps: int = 360,
) -> dict[str, Any]:
    cfg = deepcopy(base)
    cfg["bsa_attention"]["bipolar_lambda"] = float(lam)
    cfg["experiment"] = f"h54a_swp_lam{str(lam).replace('.','p')}_{lr_strategy}_s{steps}"
    cfg.setdefault("note", "")
    cfg["note"] = (
        f"H54a short: bipolar_lambda={lam}, LR={lr_strategy}, steps={steps}. "
        + cfg.get("note", "")
    )
    cfg["runtime"]["max_train_steps"] = steps
    cfg["loader"]["n_epochs"] = 1
    # Only save epoch 0 for short tests
    cfg["runtime"]["force_save_epochs"] = [0]
    apply_lr_strategy(cfg, lr_strategy)
    # Short test: use local816 split
    cfg.setdefault("data", {})["sequence_list_overrides"] = {
        "train": "/root/private_data/SothisAI/dataset/Console/DSEC/main/DSEC/saved_flow_data/sequence_lists/train_split_seq.csv.local816_backup_20260526_083510",
        "valid": "/root/private_data/SothisAI/dataset/Console/DSEC/main/DSEC/saved_flow_data/sequence_lists/valid_split_seq.csv.local816_backup_20260526_083510",
    }
    return cfg


def main() -> None:
    base = load_yaml(BASE_CONFIG)
    # Fix the base: set it as short test template
    base["runtime"]["max_train_steps"] = 360
    base["loader"]["n_epochs"] = 1
    base["runtime"]["force_save_epochs"] = [0]

    lambdas = [0.3, 0.5, 1.0, 2.0]  # 0.3 = baseline
    lr_strategies = ["fast_warm", "slowbb"]

    configs: list[dict[str, Any]] = []
    for lam in lambdas:
        for lr_strategy in lr_strategies:
            cfg = make_short_config(base, lam, lr_strategy, steps=360)
            fname = f"h54a_swp_lam{str(lam).replace('.','p')}_{lr_strategy}_s360.yml"
            path = CONFIG_DIR / "generated" / fname
            dump_yaml(path, cfg)
            configs.append(cfg)
            print(f"wrote {path}")

    print(f"\nGenerated {len(configs)} short-test configs.")
    print("Lambda values:", lambdas)
    print("LR strategies:", lr_strategies)


if __name__ == "__main__":
    main()
