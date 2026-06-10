"""Generate follow-up ATLIF budget configs after the first J58-J60 screen."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXP_ROOT / "configs"
BASE_CONFIG = CONFIG_DIR / "generated/h54b_three_mu05_l08_g20_fast_full30.yml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def is_qk_group(group: dict[str, Any]) -> bool:
    return str(group.get("name", "")).startswith("qk_")


def set_runtime(config: dict[str, Any], steps: int | None) -> None:
    runtime = config.setdefault("runtime", {})
    loader = config.setdefault("loader", {})
    if steps is None:
        runtime.pop("max_train_steps", None)
        runtime["force_save_epochs"] = list(range(30))
        loader["n_epochs"] = 30
    else:
        runtime["max_train_steps"] = steps
        runtime["force_save_epochs"] = [0]
        loader["n_epochs"] = 1
    runtime["skip_state_save"] = True


def apply_quantile_budget(
    config: dict[str, Any],
    *,
    qk_q: float,
    ffn_q: float,
    qk_margin: float,
    ffn_margin: float,
    qk_min_guard: float,
    ffn_min_guard: float,
) -> None:
    for group in config.setdefault("atlif_ternary_psn", {}).setdefault("target_groups", []):
        qk = is_qk_group(group)
        group["quantile_q"] = qk_q if qk else ffn_q
        group["quantile_momentum"] = 0.95
        group["quantile_guard_margin"] = qk_margin if qk else ffn_margin
        group["quantile_min_guard"] = qk_min_guard if qk else ffn_min_guard
        group["quantile_sample_size"] = 4096


def apply_importance(config: dict[str, Any], *, qk_scale: float, ffn_scale: float, min_guard: float) -> None:
    for group in config.setdefault("atlif_ternary_psn", {}).setdefault("target_groups", []):
        group["importance_enabled"] = True
        group["importance_momentum"] = 0.9
        group["importance_min_guard"] = min_guard
        group["importance_scale"] = qk_scale if is_qk_group(group) else ffn_scale


def make_variant(base: dict[str, Any], name: str, note: str, steps: int | None) -> dict[str, Any]:
    config = deepcopy(base)
    config["experiment"] = name
    config["note"] = note
    set_runtime(config, steps)
    return config


def main() -> int:
    base = load_yaml(BASE_CONFIG)
    variants: list[tuple[str, str, dict[str, Any]]] = [
        (
            "j61a_quantile_budget_q98_fullguard",
            "J61a: stronger quantile budget than J59. Lower membrane percentile plus wider margin makes the ATLIF positive-threshold growth slow down earlier.",
            {
                "quantile": {
                    "qk_q": 0.98,
                    "ffn_q": 0.995,
                    "qk_margin": 2.0,
                    "ffn_margin": 2.0,
                    "qk_min_guard": 0.10,
                    "ffn_min_guard": 0.05,
                }
            },
        ),
        (
            "j62a_quantile_budget_weak_importance",
            "J62a: J61a budget with weak importance guard. Tests whether the J60 stability benefit can be kept without over-suppressing useful threshold growth.",
            {
                "quantile": {
                    "qk_q": 0.98,
                    "ffn_q": 0.995,
                    "qk_margin": 2.0,
                    "ffn_margin": 2.0,
                    "qk_min_guard": 0.10,
                    "ffn_min_guard": 0.05,
                },
                "importance": {"qk_scale": 8.0, "ffn_scale": 16.0, "min_guard": 0.30},
            },
        ),
    ]
    for name, note, controls in variants:
        for suffix, steps in (("steps360", 360), ("steps720", 720), ("full30", None)):
            config = make_variant(base, f"{name}_{suffix}", f"{note} {suffix}.", steps)
            apply_quantile_budget(config, **controls["quantile"])
            if "importance" in controls:
                apply_importance(config, **controls["importance"])
            path = CONFIG_DIR / "generated" / f"{name}_{suffix}.yml"
            dump_yaml(path, config)
            print(path.relative_to(CONFIG_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
