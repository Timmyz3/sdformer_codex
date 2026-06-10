"""Generate H56a SC agree/disagree hyperparameter + LR sweep configs.

Based on H41 SC S012C slowbb. Sweeps across:
  - λ (bipolar_lambda): 0.3, 0.5, 0.8, 1.0
  - LR strategies: slowbb, fast, warm
  - target_rate: 0.05, 0.07

All variants use sc_agree_disagree_shiftmax mode. Short-test candidates
run 360 steps (from baseline checkpoint). After sweep, the best valid40
config is promoted to full30.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

EXP_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = EXP_ROOT / "configs" / "generated"
BASE = GENERATED_DIR / "h41_scs012c_slowbb_full30_20260523_133312.yml"


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def apply_lr_strategy(cfg: dict[str, Any], strategy: str) -> None:
    opt = cfg.setdefault("optimizer", {})
    groups = opt.setdefault("param_groups", {})
    groups["enabled"] = True

    strategies: dict[str, dict[str, Any]] = {
        "slowbb": {
            "backbone_lr": 2.0e-7,
            "norm_lr": 2.0e-7,
            "neuron_lr": 1.2e-5,
            "threshold_lr": 3.0e-6,
            "threshold_base_lr": 3.0e-6,
            "warmup": None,
        },
        "fast": {
            "backbone_lr": 3.0e-7,
            "norm_lr": 3.0e-7,
            "neuron_lr": 2.0e-5,
            "threshold_lr": 5.0e-6,
            "threshold_base_lr": 5.0e-6,
            "warmup": None,
        },
        "warm": {
            "backbone_lr": 2.0e-7,
            "norm_lr": 2.0e-7,
            "neuron_lr": 1.2e-5,
            "threshold_lr": 3.0e-6,
            "threshold_base_lr": 3.0e-6,
            "warmup": {"enabled": True, "steps": 100, "start_factor": 0.2},
        },
        "fast_warm": {
            "backbone_lr": 3.0e-7,
            "norm_lr": 3.0e-7,
            "neuron_lr": 2.0e-5,
            "threshold_lr": 4.0e-6,
            "threshold_base_lr": 4.0e-6,
            "warmup": {"enabled": True, "steps": 120, "start_factor": 0.15},
        },
    }

    s = strategies[strategy]
    groups.update(
        {k: v for k, v in s.items() if k not in {"threshold_base_lr", "warmup"}}
    )
    cfg.setdefault("atlif_ternary_psn", {})["threshold_base_lr"] = float(
        s["threshold_base_lr"]
    )
    if s["warmup"] is not None:
        opt["lr_warmup"] = deepcopy(s["warmup"])


def _desc(cfg: dict[str, Any], lam: float, lr: str, tr: float) -> str:
    parts = [
        "H56a SC agree/disagree sweep",
        f"λ={lam}",
        f"LR={lr}",
        f"target_rate={tr}",
    ]
    return "; ".join(parts) + ". Based on H41 SC S012C."


def main() -> int:
    base = read_yaml(BASE)

    variants: list[dict[str, Any]] = []

    # ── Core sweep: λ × LR × target_rate ──
    for lam in [0.3, 0.5, 0.8, 1.0]:
        for lr_strat in ["slowbb", "fast", "warm", "fast_warm"]:
            for tr in [0.05, 0.07]:
                variants.append(
                    {
                        "name": f"h56a_swp_l{int(lam*10):02d}_{lr_strat}_tr{int(tr*100):02d}",
                        "bipolar_lambda": lam,
                        "lr": lr_strat,
                        "target_rate": tr,
                    }
                )

    generated: list[dict[str, str]] = []
    for variant in variants:
        cfg = deepcopy(base)
        name = str(variant["name"])
        lam = float(variant["bipolar_lambda"])
        lr_strat = str(variant["lr"])
        tr = float(variant["target_rate"])

        cfg["experiment"] = name
        cfg.pop("note", None)

        attn = cfg.setdefault("bsa_attention", {})
        attn["mode"] = "sc_agree_disagree_shiftmax"
        attn["bipolar_lambda"] = lam
        attn["deadzone_epsilon"] = 0.0
        attn["confidence_enabled"] = False
        attn["k_consistency_mod"] = False
        attn["consensus_score_norm"] = "head_dim"

        atlif = cfg.setdefault("atlif_ternary_psn", {})
        atlif["target_rate"] = tr

        apply_lr_strategy(cfg, lr_strat)

        cfg["note"] = _desc(cfg, lam, lr_strat, tr)

        # short-test: 360 steps, force_save epoch 0 only
        short_path = GENERATED_DIR / f"{name}_steps360.yml"
        short_cfg = deepcopy(cfg)
        short_cfg["loader"]["n_epochs"] = 1
        short_cfg["runtime"]["max_train_steps"] = 360
        short_cfg["runtime"]["force_save_epochs"] = [0]
        write_yaml(short_path, short_cfg)

        # full30: 30 epochs, milestones [20, 25], save all epochs
        full_path = GENERATED_DIR / f"{name}_full30.yml"
        full_cfg = deepcopy(cfg)
        full_cfg["loader"]["n_epochs"] = 30
        full_cfg["optimizer"]["milestones"] = [20, 25]
        full_cfg["runtime"]["force_save_epochs"] = list(range(30))
        write_yaml(full_path, full_cfg)

        generated.append(
            {"short": f"generated/{short_path.name}", "full": f"generated/{full_path.name}"}
        )

    short_list = "\n".join(
        f"  {item['short']}" for item in generated
    )
    print(f"Generated {len(generated)} config pairs (short + full):")
    print(short_list)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
