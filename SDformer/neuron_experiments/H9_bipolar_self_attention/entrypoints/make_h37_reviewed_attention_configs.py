"""生成 H37：按外部 review 修正后的注意力短测配置。

H37 专门处理三个命名/范式问题：
- `strict_bsa_shiftmax` 过去是 QKFormer no-V 适配版；H37 增加独立 V 分支的
  `strict_bsa_qkv_shiftmax`，并强制使用 `sqrt_head_dim`。
- alpha-XNOR 原论文是二元相似性；H37 增加只看正脉冲/静默的
  `binary_alpha_xnor_matrix_*`，不再混入三值负极性冲突项。
- A2OS2A 原范式有 Q/K/V 三路；H37 增加 `a2os2a_qkv_l1`，让 V 走独立
  PSN+ATLIF 分支。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = EXP_ROOT / "configs"
BASE_CONFIG = CONFIG_DIR / "h34_hybrid_h9_stage02_highsop_s150k_act2p0.yml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def set_lr_strategy(cfg: dict[str, Any], strategy: dict[str, Any]) -> None:
    opt = cfg.setdefault("optimizer", {})
    opt["use_amp"] = True
    opt["lr"] = float(strategy["backbone_lr"])
    opt["param_groups"] = {
        "enabled": True,
        "backbone_lr": float(strategy["backbone_lr"]),
        "neuron_lr": float(strategy["neuron_lr"]),
        "threshold_lr": float(strategy["threshold_lr"]),
        "norm_lr": float(strategy["norm_lr"]),
        "norm_wd": 0.0,
        "threshold_wd": 0.0,
    }
    cfg.setdefault("atlif_ternary_psn", {})["threshold_base_lr"] = float(strategy["threshold_base_lr"])


def set_runtime(cfg: dict[str, Any], name: str, note: str) -> None:
    cfg["experiment"] = name
    cfg["note"] = note
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = [0]
    runtime["use_mlflow_model_logging"] = False
    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 1
    loader["batch_size"] = 8
    loader["n_workers"] = 8
    loader["persistent_workers"] = True
    loader["prefetch_factor"] = 4
    loader["pin_memory"] = True
    cfg.setdefault("test", {})["sample"] = 10
    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]


def set_attention(cfg: dict[str, Any], **kwargs: Any) -> None:
    base = {
        "enabled": True,
        "stage_selection": "all",
        "score_scale": 1.0,
        "center_scores": True,
        "preserve_mean": False,
        "eps": 1.0e-6,
        "consensus_score_norm": "head_dim",
        "consensus_bias": 0.02,
        "value_mode": "threshold",
        "value_branch": "reuse_k",
        "value_init": "copy_k",
        "alpha0": 0.02,
        "mismatch_penalty": 0.0,
        "relu_k_floor": 0.0,
    }
    base.update(kwargs)
    cfg["bsa_attention"] = base


def main() -> int:
    base = load_yaml(BASE_CONFIG)
    strategies = [
        {
            "name": "conservative",
            "backbone_lr": 5.0e-7,
            "norm_lr": 5.0e-7,
            "neuron_lr": 1.5e-5,
            "threshold_lr": 5.0e-6,
            "threshold_base_lr": 5.0e-6,
            "note": "H36 当前 valid40 最稳的保守续训学习率。",
        },
        {
            "name": "neuronfast",
            "backbone_lr": 5.0e-7,
            "norm_lr": 5.0e-7,
            "neuron_lr": 5.0e-5,
            "threshold_lr": 1.0e-5,
            "threshold_base_lr": 1.0e-5,
            "note": "新 PSN+ATLIF 参数更快适配，backbone 慢速续训。",
        },
    ]
    variants: list[tuple[str, dict[str, Any], str]] = [
        (
            "strict_bsa_qkv_sqrt_signv",
            {
                "mode": "strict_bsa_qkv_shiftmax",
                "consensus_score_norm": "sqrt_head_dim",
                "center_scores": True,
                "value_mode": "sign",
                "value_branch": "independent_v",
            },
            "严格 BSA-QKV：sign(Q) @ sign(K)^T / sqrt(d) -> Shiftmax -> sign(V)，V 为独立分支。",
        ),
        (
            "strict_bsa_qkv_sqrt_thetav",
            {
                "mode": "strict_bsa_qkv_shiftmax",
                "consensus_score_norm": "sqrt_head_dim",
                "center_scores": True,
                "value_mode": "threshold",
                "value_branch": "independent_v",
            },
            "严格 BSA-QKV：同上，但 V 保留 ATLIF 阈值幅度，测试精度红利。",
        ),
        (
            "binary_axnor_shiftmax",
            {
                "mode": "binary_alpha_xnor_matrix_shiftmax",
                "consensus_score_norm": "head_dim",
                "center_scores": True,
                "alpha0": 0.02,
                "mismatch_penalty": 0.0,
                "value_mode": "threshold",
            },
            "二元 alpha-XNOR 矩阵：正脉冲/静默匹配，不使用三值负极性冲突项。",
        ),
        (
            "binary_axnor_l1",
            {
                "mode": "binary_alpha_xnor_matrix_l1",
                "consensus_score_norm": "head_dim",
                "center_scores": False,
                "consensus_bias": 0.02,
                "alpha0": 0.02,
                "mismatch_penalty": 0.0,
                "value_mode": "threshold",
            },
            "二元 alpha-XNOR + L1 归一，去掉 Shiftmax 指数项的硬件友好对照。",
        ),
        (
            "a2os2a_qkv_signv",
            {
                "mode": "a2os2a_qkv_l1",
                "consensus_score_norm": "head_dim",
                "center_scores": False,
                "consensus_bias": 1.0e-6,
                "preserve_mean": True,
                "value_mode": "sign",
                "value_branch": "independent_v",
            },
            "A2OS2A-QKV：binary Q、非负 K、独立 sign(V)，L1 归一。",
        ),
        (
            "a2os2a_qkv_thetav",
            {
                "mode": "a2os2a_qkv_l1",
                "consensus_score_norm": "head_dim",
                "center_scores": False,
                "consensus_bias": 1.0e-6,
                "preserve_mean": True,
                "value_mode": "threshold",
                "value_branch": "independent_v",
            },
            "A2OS2A-QKV：binary Q、非负 K、独立 threshold(V)，测试精度红利。",
        ),
    ]

    written: list[Path] = []
    for variant_name, attention, variant_note in variants:
        for strategy in strategies:
            cfg = deepcopy(base)
            name = f"h37_{variant_name}_{strategy['name']}"
            set_attention(cfg, **attention)
            set_lr_strategy(cfg, strategy)
            set_runtime(
                cfg,
                name,
                f"H37 review 修正版注意力短测。{variant_note} 学习率：{strategy['note']} "
                "神经元范围沿用 H34 stage02_highsop：Q/K 三值 PSN+ATLIF，高 SOP 层二值 official ATLIF。",
            )
            out = CONFIG_DIR / f"{name}.yml"
            dump_yaml(out, cfg)
            written.append(out)
            print(out)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
