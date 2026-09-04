"""Generate H23 combo configs after the H13v low-LR signal.

H13v indicates that lowering LR can recover early AEE/AAE, while SOPs remain
too high. H22 tests one hyperparameter axis at a time; H23 tests small combined
settings that pair lower LR with stronger sparse pressure.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXP_ROOT / "configs"


def load_config(name: str) -> dict[str, Any]:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_config(name: str, config: dict[str, Any]) -> None:
    with (CONFIG_DIR / name).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def common(base: dict[str, Any], experiment: str) -> dict[str, Any]:
    cfg = deepcopy(base)
    cfg["experiment"] = experiment
    cfg.setdefault("runtime", {})["max_train_steps"] = 120
    cfg["runtime"]["skip_state_save"] = True
    cfg.setdefault("loader", {})["n_epochs"] = 1
    cfg["loader"]["batch_size"] = 8
    cfg["loader"]["n_workers"] = 8
    cfg["loader"]["pin_memory"] = False
    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    cfg.setdefault("test", {})["sample"] = 10
    return cfg


def update_sparse(cfg: dict[str, Any], target_rate: float, target_eta: float, activity_eta: float) -> None:
    cfg.setdefault("atlif_ternary_psn", {}).update(
        {
            "target_rate": target_rate,
            "target_rate_eta": target_eta,
            "activity_eta": activity_eta,
        }
    )


def main() -> None:
    h18c = load_config("h18c_alpha_xnor_direct_shiftmax_guard120.yml")
    h13v = load_config("h13v_target05_lower_lr_guard120.yml")

    combos = [
        (
            h18c,
            "h23a_h18c_lr1e5_target040",
            1.0e-5,
            0.040,
            0.05,
            0.8,
            {},
        ),
        (
            h18c,
            "h23b_h18c_lr1e5_target035",
            1.0e-5,
            0.035,
            0.08,
            1.0,
            {},
        ),
        (
            h18c,
            "h23c_h18c_lr1e5_target040_score075",
            1.0e-5,
            0.040,
            0.05,
            0.8,
            {"score_scale": 0.75},
        ),
        (
            h13v,
            "h23d_h13v_lr1e5_target040",
            1.0e-5,
            0.040,
            0.05,
            0.8,
            {},
        ),
        (
            h13v,
            "h23e_h13v_lr1e5_target035",
            1.0e-5,
            0.035,
            0.08,
            1.0,
            {},
        ),
    ]

    for base, name, lr, target_rate, target_eta, activity_eta, attn_updates in combos:
        cfg = common(base, name)
        cfg.setdefault("optimizer", {})["lr"] = lr
        update_sparse(cfg, target_rate, target_eta, activity_eta)
        cfg.setdefault("bsa_attention", {}).update(attn_updates)
        cfg["note"] = (
            "H23 combo sweep: low LR plus sparse target feedback. "
            f"lr={lr}, target_rate={target_rate}, target_rate_eta={target_eta}, "
            f"activity_eta={activity_eta}, attention_updates={attn_updates}."
        )
        write_config(f"{name}_guard120.yml", cfg)


if __name__ == "__main__":
    main()
