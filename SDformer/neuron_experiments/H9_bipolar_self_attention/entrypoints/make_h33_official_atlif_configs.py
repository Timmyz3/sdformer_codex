"""生成 H33 官方 ATLIF 范式复核配置。

H33 目的：
1. 把 ATLIF 部分恢复为 Activity-Pruning-SNN 官方 binary ATLIF 更新闭环；
2. 区分“纯官方 ATLIF-PSN”和“H9 三值注意力 + 官方 ATLIF 高 SOP 层”；
3. 先生成 rapid 配置，避免直接全量浪费训练时间。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXP_ROOT / "configs"
BASE_H28B_FULL = CONFIG_DIR / "h28b_diff_lr_newfast_steps360_auto_full_20260520_201852.yml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def official_group(group: dict[str, Any], *, activity_eta: float, threshold_lr_scale: float) -> dict[str, Any]:
    converted = deepcopy(group)
    converted.update(
        {
            "output_mode": "binary",
            "center_mode": "zero",
            "threshold_mode": "official_atlif",
            "threshold_init": 0.1,
            "threshold_eta": 0.001,
            "threshold_lr_scale": threshold_lr_scale,
            "min_threshold": None,
            "max_threshold": None,
            "target_rate": None,
            "target_rate_eta": 0.0,
            "activity_eta": activity_eta,
        }
    )
    converted["name"] = f"{group.get('name', 'group')}_official_atlif"
    return converted


def set_runtime(cfg: dict[str, Any], *, name: str, rapid: bool, note: str) -> None:
    cfg["experiment"] = name
    cfg["note"] = note
    runtime = cfg.setdefault("runtime", {})
    runtime["skip_state_save"] = True
    runtime["use_mlflow_model_logging"] = False
    if rapid:
        runtime["max_train_steps"] = 360
        runtime["force_save_epochs"] = []
        cfg.setdefault("loader", {})["n_epochs"] = 1
        cfg.setdefault("test", {})["sample"] = 10
    else:
        runtime["max_train_steps"] = 0
        runtime["force_save_epochs"] = [9, 19, 29]
        cfg.setdefault("loader", {})["n_epochs"] = 30
    loader = cfg.setdefault("loader", {})
    loader["batch_size"] = 8
    loader["n_workers"] = 8
    loader["persistent_workers"] = True
    loader["prefetch_factor"] = 4
    loader["pin_memory"] = True


def set_diff_lr(cfg: dict[str, Any]) -> None:
    opt = cfg.setdefault("optimizer", {})
    opt["use_amp"] = True
    opt["lr"] = 1.0e-6
    opt["param_groups"] = {
        "enabled": True,
        "backbone_lr": 1.0e-6,
        "neuron_lr": 3.0e-5,
        "threshold_lr": 1.0e-5,
        "norm_lr": 1.0e-6,
        "norm_wd": 0.0,
        "threshold_wd": 0.0,
    }


def set_top_level_official_binary(
    cfg: dict[str, Any],
    *,
    activity_eta: float,
    threshold_lr_scale: float,
    target_groups: bool,
) -> None:
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif.update(
        {
            "enabled": True,
            "target": "qk",
            "stage_selection": "all",
            "output_mode": "binary",
            "threshold_mode": "official_atlif",
            "center_mode": "zero",
            "threshold_init": 0.1,
            "threshold_eta": 0.001,
            "threshold_lr_scale": threshold_lr_scale,
            "threshold_base_lr": 1.0e-5,
            "min_threshold": None,
            "max_threshold": None,
            "target_rate": None,
            "target_rate_eta": 0.0,
            "activity_eta": activity_eta,
            "trainable": "all",
            "log_interval_steps": 20,
        }
    )
    groups = atlif.get("target_groups", []) or []
    atlif["target_groups"] = [
        official_group(group, activity_eta=activity_eta, threshold_lr_scale=threshold_lr_scale)
        for group in groups
    ] if target_groups else []
    cfg.setdefault("bsa_attention", {})["enabled"] = False


def set_h9_qk_ternary_with_official_groups(
    cfg: dict[str, Any],
    *,
    activity_eta: float,
    threshold_lr_scale: float,
) -> None:
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    groups = atlif.get("target_groups", []) or []
    atlif["target_groups"] = [
        official_group(group, activity_eta=activity_eta, threshold_lr_scale=threshold_lr_scale)
        for group in groups
    ]
    atlif["threshold_base_lr"] = 1.0e-5
    atlif["log_interval_steps"] = 20


def build_specs() -> list[tuple[str, str, Any]]:
    return [
        (
            "h33a_official_qk_binary_scale150k_act2",
            "H33a：纯官方范式。只把所有 attention Q/K 替换成 binary official ATLIF-PSN，关闭 Shiftmax/BSA。",
            lambda cfg: set_top_level_official_binary(
                cfg, activity_eta=2.0, threshold_lr_scale=150000.0, target_groups=False
            ),
        ),
        (
            "h33b_official_qk_highsop_binary_scale150k_act2",
            "H33b：纯官方范式。Q/K + H28b 已选 FFN/downsample 全部使用 binary official ATLIF-PSN，关闭 Shiftmax/BSA。",
            lambda cfg: set_top_level_official_binary(
                cfg, activity_eta=2.0, threshold_lr_scale=150000.0, target_groups=True
            ),
        ),
        (
            "h33c_h9_qkternary_highsop_official_scale150k_act2",
            "H33c：保留 H9/H28b 的 Q/K 三值 + alpha-XNOR Shiftmax；但 FFN/downsample 改为官方 binary ATLIF 范式。",
            lambda cfg: set_h9_qk_ternary_with_official_groups(
                cfg, activity_eta=2.0, threshold_lr_scale=150000.0
            ),
        ),
        (
            "h33d_h9_qkternary_highsop_official_scale300k_act4",
            "H33d：H33c 的更强稀疏版本，提高 official ATLIF 阈值更新强度和 activity penalty。",
            lambda cfg: set_h9_qk_ternary_with_official_groups(
                cfg, activity_eta=4.0, threshold_lr_scale=300000.0
            ),
        ),
    ]


def main() -> int:
    base = load_yaml(BASE_H28B_FULL)
    for base_name, note, mutate in build_specs():
        for mode, rapid in (("rapid", True), ("full", False)):
            cfg = deepcopy(base)
            set_diff_lr(cfg)
            mutate(cfg)
            set_runtime(
                cfg,
                name=f"{base_name}_{mode}",
                rapid=rapid,
                note=f"{note}\n\n官方参照：Activity-Pruning-SNN models/submodules/layers.py 的 ATLIF/Surrogate/threshold_update。",
            )
            path = CONFIG_DIR / f"{base_name}_{mode}.yml"
            dump_yaml(path, cfg)
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
