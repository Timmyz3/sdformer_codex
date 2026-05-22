"""生成 H31 更稀疏的分组 LR / binary target-rate / BSA sweep 配置。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXP_ROOT / "configs"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def prepare(cfg: dict[str, Any], name: str, note: str) -> dict[str, Any]:
    cfg = deepcopy(cfg)
    cfg["experiment"] = name
    cfg["note"] = note
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = True
    runtime["use_mlflow_model_logging"] = False
    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 1
    loader["batch_size"] = 8
    loader["n_workers"] = 8
    loader["persistent_workers"] = True
    loader["prefetch_factor"] = 4
    loader["pin_memory"] = False
    cfg.setdefault("optimizer", {})["use_amp"] = True
    cfg.setdefault("test", {})["sample"] = 10
    return cfg


def set_diff_lr(cfg: dict[str, Any], backbone: float, neuron: float, threshold: float, norm: float) -> None:
    opt = cfg.setdefault("optimizer", {})
    opt["lr"] = backbone
    opt["param_groups"] = {
        "enabled": True,
        "backbone_lr": backbone,
        "neuron_lr": neuron,
        "threshold_lr": threshold,
        "norm_lr": norm,
        "norm_wd": 0.0,
        "threshold_wd": 0.0,
    }
    cfg.setdefault("atlif_ternary_psn", {})["threshold_base_lr"] = threshold


def set_qk_sparse(cfg: dict[str, Any], target_rate: float, eta: float, max_threshold: float) -> None:
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif["target_rate"] = target_rate
    atlif["target_rate_eta"] = eta
    atlif["max_threshold"] = max_threshold


def set_binary_sparse(cfg: dict[str, Any], default_rate: float, eta: float, max_threshold: float) -> None:
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    for group in atlif.get("target_groups", []) or []:
        if str(group.get("output_mode", "")) != "binary":
            continue
        name = str(group.get("name", ""))
        rate = default_rate
        if "stage0" in name or "downsample" in name:
            rate = default_rate + 0.01
        if "stage2" in name or "stage3" in name:
            rate = max(0.035, default_rate - 0.005)
        group["target_rate"] = float(rate)
        group["target_rate_eta"] = float(eta)
        group["max_threshold"] = float(max_threshold)


def set_strict_bsa(cfg: dict[str, Any], value_mode: str) -> None:
    cfg.setdefault("bsa_attention", {}).update(
        {
            "enabled": True,
            "stage_selection": "all",
            "mode": "strict_bsa_shiftmax",
            "score_scale": 1.0,
            "center_scores": True,
            "preserve_mean": False,
            "consensus_score_norm": "sqrt_head_dim",
            "value_mode": value_mode,
        }
    )


def main() -> int:
    h28b = load_yaml(CONFIG_DIR / "h28b_diff_lr_newfast.yml")
    h29b = load_yaml(CONFIG_DIR / "h29b_diff_lr_binary_target_strong.yml")
    h30b = load_yaml(CONFIG_DIR / "h30b_strict_bsa_thresholdv_diff_lr.yml")

    specs: list[tuple[str, dict[str, Any]]] = []

    cfg = prepare(h28b, "h31a_newfast_sparse030_bin055", "H31a：H28b 分组 LR，QK target=0.030，binary target≈0.055。")
    set_diff_lr(cfg, 1.0e-6, 3.0e-5, 1.0e-5, 1.0e-6)
    set_qk_sparse(cfg, 0.030, 0.10, 1.9)
    set_binary_sparse(cfg, 0.055, 0.04, 0.20)
    specs.append((cfg["experiment"], cfg))

    cfg = prepare(h28b, "h31b_newfast_sparse028_bin045", "H31b：H28b 更强稀疏，QK target=0.028，binary target≈0.045。")
    set_diff_lr(cfg, 1.0e-6, 3.0e-5, 1.0e-5, 1.0e-6)
    set_qk_sparse(cfg, 0.028, 0.12, 2.0)
    set_binary_sparse(cfg, 0.045, 0.05, 0.24)
    specs.append((cfg["experiment"], cfg))

    cfg = prepare(h29b, "h31c_h29b_lower_threshold_lr", "H31c：H29b 强 binary 稀疏，但阈值更新 LR 降低以保护 AAE。")
    set_diff_lr(cfg, 1.0e-6, 3.0e-5, 7.5e-6, 1.0e-6)
    set_qk_sparse(cfg, 0.035, 0.08, 1.8)
    set_binary_sparse(cfg, 0.055, 0.035, 0.18)
    specs.append((cfg["experiment"], cfg))

    cfg = prepare(h29b, "h31d_h29b_high_binary_eta", "H31d：H29b 强化 binary target-rate 反馈，测试 SOPs 下限。")
    set_diff_lr(cfg, 1.0e-6, 3.0e-5, 1.0e-5, 1.0e-6)
    set_qk_sparse(cfg, 0.033, 0.10, 1.9)
    set_binary_sparse(cfg, 0.050, 0.06, 0.25)
    specs.append((cfg["experiment"], cfg))

    cfg = prepare(h30b, "h31e_strict_bsa_sparse030_bin055", "H31e：H30b strict BSA threshold-V + 更稀疏 QK/binary。")
    set_diff_lr(cfg, 1.0e-6, 3.0e-5, 1.0e-5, 1.0e-6)
    set_strict_bsa(cfg, "threshold")
    set_qk_sparse(cfg, 0.030, 0.10, 1.9)
    set_binary_sparse(cfg, 0.055, 0.04, 0.20)
    specs.append((cfg["experiment"], cfg))

    cfg = prepare(h30b, "h31f_strict_bsa_sparse028_bin045", "H31f：H30b strict BSA threshold-V + 最强稀疏短测。")
    set_diff_lr(cfg, 1.0e-6, 3.0e-5, 1.0e-5, 1.0e-6)
    set_strict_bsa(cfg, "threshold")
    set_qk_sparse(cfg, 0.028, 0.12, 2.0)
    set_binary_sparse(cfg, 0.045, 0.05, 0.24)
    specs.append((cfg["experiment"], cfg))

    for name, cfg in specs:
        path = CONFIG_DIR / f"{name}.yml"
        dump_yaml(path, cfg)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
