"""Generate J58-J60 ATLIF control configs from the H54b baseline candidate."""

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


def _is_qk_group(group: dict[str, Any]) -> bool:
    return str(group.get("name", "")).startswith("qk_")


def _set_short_runtime(config: dict[str, Any], steps: int = 360) -> None:
    runtime = config.setdefault("runtime", {})
    runtime["max_train_steps"] = steps
    runtime["force_save_epochs"] = [0]
    runtime["skip_state_save"] = True
    loader = config.setdefault("loader", {})
    loader["n_epochs"] = 1


def _set_full_runtime(config: dict[str, Any]) -> None:
    runtime = config.setdefault("runtime", {})
    runtime.pop("max_train_steps", None)
    runtime["force_save_epochs"] = list(range(30))
    runtime["skip_state_save"] = True
    loader = config.setdefault("loader", {})
    loader["n_epochs"] = 30


def _apply_importance(config: dict[str, Any], qk_scale: float, ffn_scale: float) -> None:
    for group in config.setdefault("atlif_ternary_psn", {}).setdefault("target_groups", []):
        group["importance_enabled"] = True
        group["importance_momentum"] = 0.9
        group["importance_min_guard"] = 0.15
        group["importance_scale"] = qk_scale if _is_qk_group(group) else ffn_scale


def _apply_quantile(config: dict[str, Any], qk_q: float, ffn_q: float) -> None:
    for group in config.setdefault("atlif_ternary_psn", {}).setdefault("target_groups", []):
        group["quantile_q"] = qk_q if _is_qk_group(group) else ffn_q
        group["quantile_momentum"] = 0.95
        group["quantile_guard_margin"] = 0.20 if _is_qk_group(group) else 0.15
        group["quantile_min_guard"] = 0.05 if _is_qk_group(group) else 0.02
        group["quantile_sample_size"] = 4096


def make_variant(base: dict[str, Any], name: str, note: str, controls: str, full: bool = False) -> dict[str, Any]:
    config = deepcopy(base)
    config["experiment"] = name
    config["note"] = note
    if controls == "importance":
        _apply_importance(config, qk_scale=25.0, ffn_scale=50.0)
    elif controls == "quantile":
        _apply_quantile(config, qk_q=0.995, ffn_q=0.9995)
    elif controls == "quantile_importance":
        _apply_quantile(config, qk_q=0.995, ffn_q=0.9995)
        _apply_importance(config, qk_scale=20.0, ffn_scale=40.0)
    else:
        raise ValueError(f"unknown controls: {controls}")
    if full:
        _set_full_runtime(config)
    else:
        _set_short_runtime(config, steps=360)
    return config


def main() -> int:
    base = load_yaml(BASE_CONFIG)
    variants = [
        (
            "j58a_importance_h54b_steps360",
            "J58a: H54b structure with importance-aware ATLIF update guard. Protects high-saliency spikes by reducing positive threshold growth after backward.",
            "importance",
        ),
        (
            "j59a_quantile_h54b_steps360",
            "J59a: H54b structure with quantile ATLIF guard. Threshold growth slows when threshold catches the layer membrane distribution budget.",
            "quantile",
        ),
        (
            "j60a_quantile_importance_h54b_steps360",
            "J60a: H54b structure with quantile plus importance ATLIF guards. Quantile controls sparsity budget; saliency protects task-critical spikes.",
            "quantile_importance",
        ),
    ]
    for name, note, controls in variants:
        short_path = CONFIG_DIR / "generated" / f"{name}.yml"
        dump_yaml(short_path, make_variant(base, name, note, controls, full=False))
        print(short_path.relative_to(CONFIG_DIR))
        full_name = name.replace("_steps360", "_full30")
        full_path = CONFIG_DIR / "generated" / f"{full_name}.yml"
        dump_yaml(full_path, make_variant(base, full_name, note.replace("J", "J") + " Full30 candidate.", controls, full=True))
        print(full_path.relative_to(CONFIG_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
