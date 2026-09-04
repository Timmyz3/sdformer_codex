"""生成 H28/H29/H30 学习率、binary 发放和 BSA 对照配置。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXP_ROOT / "configs"
BASE_H23B = CONFIG_DIR / "h23b_h18c_lr1e5_target035_auto_full_20260520_125502.yml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def set_diff_lr(
    cfg: dict[str, Any],
    *,
    backbone_lr: float,
    neuron_lr: float,
    threshold_lr: float,
    norm_lr: float,
) -> None:
    opt = cfg.setdefault("optimizer", {})
    opt["lr"] = backbone_lr
    opt["use_amp"] = True
    opt["param_groups"] = {
        "enabled": True,
        "backbone_lr": backbone_lr,
        "neuron_lr": neuron_lr,
        "threshold_lr": threshold_lr,
        "norm_lr": norm_lr,
        "norm_wd": 0.0,
        "threshold_wd": 0.0,
    }
    cfg.setdefault("atlif_ternary_psn", {})["threshold_base_lr"] = threshold_lr


def set_binary_targets(
    cfg: dict[str, Any],
    *,
    rates: dict[str, float],
    eta: float,
    max_threshold: float,
) -> None:
    groups = cfg.setdefault("atlif_ternary_psn", {}).setdefault("target_groups", [])
    for group in groups:
        if str(group.get("output_mode", "")) != "binary":
            continue
        name = str(group.get("name", ""))
        rate = rates.get(name)
        if rate is None:
            rate = rates.get("default")
        if rate is None:
            continue
        group["target_rate"] = float(rate)
        group["target_rate_eta"] = float(eta)
        group["max_threshold"] = float(max_threshold)
        group["threshold_lr_scale"] = float(group.get("threshold_lr_scale", 6000.0))


def set_strict_bsa(cfg: dict[str, Any], *, value_mode: str) -> None:
    bsa = cfg.setdefault("bsa_attention", {})
    bsa.update(
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


def base_config(name: str, note: str) -> dict[str, Any]:
    cfg = load_yaml(BASE_H23B)
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
    return cfg


def main() -> int:
    specs: list[tuple[str, dict[str, Any]]] = []

    lr_specs = [
        ("h28a_diff_lr_safe", 3.0e-6, 2.0e-5, 1.0e-5, 1.0e-6),
        ("h28b_diff_lr_newfast", 1.0e-6, 3.0e-5, 1.0e-5, 1.0e-6),
        ("h28c_diff_lr_balanced", 5.0e-6, 2.0e-5, 5.0e-6, 2.0e-6),
    ]
    for name, backbone_lr, neuron_lr, threshold_lr, norm_lr in lr_specs:
        cfg = base_config(
            name,
            "H28：续训分组学习率。旧 backbone 小 LR，新 ATLIF/PSN 模块较大 LR，阈值更新单独 LR。",
        )
        set_diff_lr(cfg, backbone_lr=backbone_lr, neuron_lr=neuron_lr, threshold_lr=threshold_lr, norm_lr=norm_lr)
        specs.append((name, cfg))

    h29a = base_config(
        "h29a_diff_lr_binary_target_mild",
        "H29a：H28a 分组 LR + binary FFN/downsample target-rate，温和压低 binary firing。",
    )
    set_diff_lr(h29a, backbone_lr=3.0e-6, neuron_lr=2.0e-5, threshold_lr=1.0e-5, norm_lr=1.0e-6)
    set_binary_targets(
        h29a,
        rates={
            "stage0_all_ffn_binary": 0.08,
            "stage1_half_even_ffn_binary": 0.075,
            "stage2_half_even_ffn_binary": 0.065,
            "stage3_block0_ffn_binary": 0.065,
            "downsample_stage0_stage2_binary": 0.08,
            "default": 0.075,
        },
        eta=0.02,
        max_threshold=0.13,
    )
    specs.append(("h29a_diff_lr_binary_target_mild", h29a))

    h29b = base_config(
        "h29b_diff_lr_binary_target_strong",
        "H29b：H28a 分组 LR + 更强 binary target-rate，验证 SOPs 是否能明显下降。",
    )
    set_diff_lr(h29b, backbone_lr=3.0e-6, neuron_lr=2.0e-5, threshold_lr=1.0e-5, norm_lr=1.0e-6)
    set_binary_targets(
        h29b,
        rates={
            "stage0_all_ffn_binary": 0.065,
            "stage1_half_even_ffn_binary": 0.06,
            "stage2_half_even_ffn_binary": 0.055,
            "stage3_block0_ffn_binary": 0.055,
            "downsample_stage0_stage2_binary": 0.065,
            "default": 0.06,
        },
        eta=0.03,
        max_threshold=0.16,
    )
    specs.append(("h29b_diff_lr_binary_target_strong", h29b))

    for name, value_mode in (
        ("h30a_strict_bsa_signv_diff_lr", "sign"),
        ("h30b_strict_bsa_thresholdv_diff_lr", "threshold"),
    ):
        cfg = base_config(
            name,
            f"H30：在 H23b 神经元范围上测试 strict BSA 矩阵注意力，value_mode={value_mode}，并使用 H28a 分组 LR。",
        )
        set_diff_lr(cfg, backbone_lr=3.0e-6, neuron_lr=2.0e-5, threshold_lr=1.0e-5, norm_lr=1.0e-6)
        set_strict_bsa(cfg, value_mode=value_mode)
        specs.append((name, cfg))

    for name, cfg in specs:
        path = CONFIG_DIR / f"{name}.yml"
        dump_yaml(path, cfg)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
