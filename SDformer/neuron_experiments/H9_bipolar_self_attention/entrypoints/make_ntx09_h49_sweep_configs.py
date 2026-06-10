"""Generate NTX-09 H49 qkselector sweep configs.

H49 formula: attn = K * Shiftmax(TX(Q_i, K_i))  — no carrier, no external gate.
Sweep: beta x LR strategy x warmup x single_active
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED = EXP_ROOT / "configs" / "generated"
BASE = GENERATED / "ntx07a_h49_qkselector_s2_m025_s005.yml"


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def apply_lr(config: dict[str, Any], strategy: str) -> None:
    opt = config.setdefault("optimizer", {})
    groups = opt.setdefault("param_groups", {})
    groups["enabled"] = True
    opt.pop("lr_warmup", None)

    if strategy == "slowbb":  # NTX-01 style
        opt["lr"] = 1.0e-5
        groups["backbone_lr"] = 2.0e-7
        groups["norm_lr"] = 2.0e-7
        groups["neuron_lr"] = 1.2e-5
        groups["threshold_lr"] = 3.0e-6
    elif strategy == "fastbb":  # NTX-04H style
        opt["lr"] = 2.0e-5
        groups["backbone_lr"] = 1.0e-6
        groups["norm_lr"] = 1.0e-6
        groups["neuron_lr"] = 3.0e-5
        groups["threshold_lr"] = 5.0e-6
    elif strategy == "midbb":  # moderate
        opt["lr"] = 1.5e-5
        groups["backbone_lr"] = 5.0e-7
        groups["norm_lr"] = 5.0e-7
        groups["neuron_lr"] = 2.0e-5
        groups["threshold_lr"] = 4.0e-6


def add_warmup(config: dict[str, Any], steps: int = 300) -> None:
    config.setdefault("optimizer", {})["lr_warmup"] = {
        "enabled": True,
        "steps": steps,
        "start_factor": 0.1,
    }


def make_short_config(
    base: dict[str, Any],
    beta: float,
    lr: str,
    warmup: bool,
    single_active: float,
    steps: int = 360,
) -> dict[str, Any]:
    cfg = deepcopy(base)
    name = f"ntx09_h49_b{str(beta).replace('.','p')}_{lr}"
    if warmup:
        name += "_warm"
    name += f"_s{str(single_active).replace('.','p')}_s{steps}"
    cfg["experiment"] = name
    cfg["note"] = f"NTX-09 H49 sweep: beta={beta}, LR={lr}, warmup={warmup}, single={single_active}"

    cfg["bsa_attention"]["mismatch_penalty"] = float(beta)
    cfg["bsa_attention"]["single_active_penalty"] = float(single_active)
    cfg["bsa_attention"]["single_active_penalty_grad"] = "ste"

    cfg["loader"]["n_epochs"] = 1
    cfg.setdefault("runtime", {})["max_train_steps"] = steps
    cfg["runtime"]["force_save_epochs"] = [0]
    cfg["runtime"]["skip_state_save"] = True

    apply_lr(cfg, lr)
    if warmup:
        add_warmup(cfg, steps=200)

    return cfg


def main() -> int:
    base = read_yaml(BASE)
    betas = [0.25, 0.5, 0.75, 1.0]
    lr_strategies = ["slowbb", "midbb", "fastbb"]
    warmup_opts = [False, True]
    single_actives = [0.0, 0.05]

    configs = []
    for beta in betas:
        for lr in lr_strategies:
            for warmup in warmup_opts:
                for sa in single_actives:
                    # Skip some combos to keep sweep manageable: only test warmup×sa combos on best beta
                    if beta > 0.5 and lr != "midbb":
                        continue  # focus mid LR for higher betas
                    cfg = make_short_config(base, beta, lr, warmup, sa, steps=360)
                    path = GENERATED / f"{cfg['experiment']}.yml"
                    write_yaml(path, cfg)
                    configs.append(cfg["experiment"])

    # Also add full30 configs for a few promising combos
    full_combos = [
        (0.25, "midbb", True, 0.05, 30),
        (0.5, "midbb", True, 0.05, 30),
        (0.25, "fastbb", True, 0.00, 30),
        (0.75, "midbb", True, 0.05, 30),
    ]
    for beta, lr, warmup, sa, epochs in full_combos:
        cfg = make_short_config(base, beta, lr, warmup, sa, steps=0)
        cfg["loader"]["n_epochs"] = epochs
        cfg["runtime"]["max_train_steps"] = 0
        cfg["runtime"]["force_save_epochs"] = list(range(epochs))
        cfg["runtime"]["skip_state_save"] = False
        name = f"ntx09_h49_b{str(beta).replace('.','p')}_{lr}"
        if warmup:
            name += "_warm"
        name += f"_s{str(sa).replace('.','p')}_full{epochs}"
        cfg["experiment"] = name
        path = GENERATED / f"{name}.yml"
        write_yaml(path, cfg)

    print(f"Generated {len(configs)} short-test + {len(full_combos)} full30 configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
