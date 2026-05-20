"""生成 H24/H25 中文实验配置。

H24: 固定 alpha-XNOR + Shiftmax 注意力，回到 H9a 的低 SOPs 替换范围，
     重点扫学习率、角度 loss、ATLIF 稀疏强度。
H25: Q/K 硬性保持三值 ATLIF，细分 FFN 升维/降维、二值/三值、downsample
     的排列组合。
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
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


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
    cfg.setdefault("optimizer", {})["use_amp"] = True
    return cfg


def set_alpha_xnor_shiftmax(cfg: dict[str, Any]) -> None:
    cfg["bsa_attention"] = {
        "enabled": True,
        "stage_selection": "all",
        "mode": "alpha_xnor_matrix_shiftmax",
        "score_scale": 1.0,
        "center_scores": True,
        "preserve_mean": False,
        "eps": 1.0e-6,
        "consensus_score_norm": "head_dim",
        "alpha0": 0.02,
        "mismatch_penalty": 0.25,
        "value_mode": "threshold",
    }


def set_qk_atlif(
    cfg: dict[str, Any],
    *,
    target_rate: float | None = None,
    target_rate_eta: float | None = None,
    activity_eta: float | None = None,
    threshold_init: float = 0.1,
    max_threshold: float = 0.13,
    negative_threshold_scale: float = 30.0,
) -> None:
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif.update(
        {
            "enabled": True,
            "target": "qk",
            "stage_selection": "all",
            "output_mode": "ternary",
            "threshold_init": threshold_init,
            "threshold_eta": 0.001,
            "threshold_lr_scale": 50000.0,
            "min_threshold": 0.001,
            "max_threshold": max_threshold,
            "negative_threshold_scale": negative_threshold_scale,
            "activity_eta": 2.0 if activity_eta is None else activity_eta,
            "trainable": "all",
            "log_interval_steps": 20,
        }
    )
    if target_rate is None:
        atlif.pop("target_rate", None)
        atlif.pop("target_rate_eta", None)
    else:
        atlif["target_rate"] = target_rate
        atlif["target_rate_eta"] = 0.05 if target_rate_eta is None else target_rate_eta


def group(name: str, paths: list[str], output_mode: str = "binary", *, activity_eta: float = 0.02) -> dict[str, Any]:
    return {
        "name": name,
        "output_mode": output_mode,
        "threshold_eta": 8.0e-5,
        "threshold_lr_scale": 8000.0,
        "max_threshold": 0.105,
        "activity_eta": activity_eta,
        "paths": paths,
    }


STAGE0_SN1 = [
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn1",
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.mlp.sn1",
]
STAGE0_SN2 = [
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn2",
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.mlp.sn2",
]
STAGE3B0_SN1 = ["sttmultires_unet.encoders.swin3d.layers.3.swin_blocks.0.mlp.sn1"]
STAGE3B0_SN2 = ["sttmultires_unet.encoders.swin3d.layers.3.swin_blocks.0.mlp.sn2"]
DOWN02 = [
    "sttmultires_unet.encoders.swin3d.layers.0.downsample.sn",
    "sttmultires_unet.encoders.swin3d.layers.2.downsample.sn",
]


def h9a_scope_groups(ffn_mode: str = "binary", downsample: bool = True) -> list[dict[str, Any]]:
    groups = [
        group("stage0_ffn", STAGE0_SN1 + STAGE0_SN2, output_mode=ffn_mode, activity_eta=0.02),
        group("stage3_block0_ffn", STAGE3B0_SN1 + STAGE3B0_SN2, output_mode=ffn_mode, activity_eta=0.006),
    ]
    if downsample:
        groups.append(group("downsample_stage0_stage2", DOWN02, output_mode="binary", activity_eta=0.02))
    return groups


def main() -> None:
    base = load_config("h9a_shiftmax_compat_h8m_speed_bs14.yml")

    h24_specs = [
        ("h24a_h9ascope_axnor_base", {}, {}, "H9a 替换范围 + alpha-XNOR Shiftmax 基线"),
        ("h24b_h9ascope_axnor_lr1e5", {"lr": 1.0e-5}, {}, "低学习率，测试是否保住 H13v 的精度红利"),
        (
            "h24c_h9ascope_axnor_sparse040",
            {"lr": 1.0e-5},
            {"target_rate": 0.040, "target_rate_eta": 0.05, "activity_eta": 2.5},
            "低学习率 + 中等稀疏反馈",
        ),
        (
            "h24d_h9ascope_axnor_sparse035",
            {"lr": 1.0e-5},
            {"target_rate": 0.035, "target_rate_eta": 0.08, "activity_eta": 3.0},
            "低学习率 + 更强稀疏反馈",
        ),
        (
            "h24e_h9ascope_axnor_ang002",
            {"lr": 1.0e-5},
            {"target_rate": 0.040, "target_rate_eta": 0.05, "activity_eta": 2.5},
            "低学习率 + 稀疏反馈 + 小角度 loss",
        ),
        (
            "h24f_h9ascope_axnor_ang005",
            {"lr": 1.0e-5},
            {"target_rate": 0.040, "target_rate_eta": 0.05, "activity_eta": 2.5},
            "低学习率 + 稀疏反馈 + 稍大角度 loss",
        ),
        (
            "h24g_h9ascope_axnor_flowreg0003",
            {"lr": 1.0e-5},
            {"target_rate": 0.040, "target_rate_eta": 0.05, "activity_eta": 2.5},
            "降低 flow_regul_weight，测试正则项对 AAE/稀疏的影响",
        ),
    ]
    for name, optim_updates, atlif_updates, note in h24_specs:
        cfg = common(base, name)
        set_alpha_xnor_shiftmax(cfg)
        set_qk_atlif(cfg, **atlif_updates)
        cfg["atlif_ternary_psn"]["target_groups"] = h9a_scope_groups("binary", downsample=True)
        cfg.setdefault("optimizer", {}).update(optim_updates)
        if name.endswith("ang002"):
            cfg["loss"]["use_angular_loss"] = True
            cfg["loss"]["lambda_ang"] = 0.02
        if name.endswith("ang005"):
            cfg["loss"]["use_angular_loss"] = True
            cfg["loss"]["lambda_ang"] = 0.05
        if name.endswith("flowreg0003"):
            cfg["loss"]["flow_regul_weight"] = 0.0003
        cfg["note"] = note
        write_config(f"{name}_guard120.yml", cfg)

    h25_specs = [
        ("h25a_ffn_sn1_only_binary", [group("ffn_sn1_only_binary", STAGE0_SN1 + STAGE3B0_SN1, "binary", activity_eta=0.02), group("down02_binary", DOWN02, "binary", activity_eta=0.02)]),
        ("h25b_ffn_sn2_only_binary", [group("ffn_sn2_only_binary", STAGE0_SN2 + STAGE3B0_SN2, "binary", activity_eta=0.02), group("down02_binary", DOWN02, "binary", activity_eta=0.02)]),
        ("h25c_ffn_sn1_ternary_sn2_binary", [group("ffn_sn1_ternary", STAGE0_SN1 + STAGE3B0_SN1, "ternary", activity_eta=0.02), group("ffn_sn2_binary", STAGE0_SN2 + STAGE3B0_SN2, "binary", activity_eta=0.02), group("down02_binary", DOWN02, "binary", activity_eta=0.02)]),
        ("h25d_ffn_sn1_binary_sn2_ternary", [group("ffn_sn1_binary", STAGE0_SN1 + STAGE3B0_SN1, "binary", activity_eta=0.02), group("ffn_sn2_ternary", STAGE0_SN2 + STAGE3B0_SN2, "ternary", activity_eta=0.02), group("down02_binary", DOWN02, "binary", activity_eta=0.02)]),
        ("h25e_no_ffn_downsample_only", [group("down02_binary", DOWN02, "binary", activity_eta=0.02)]),
        ("h25f_ffn_all_ternary", h9a_scope_groups("ternary", downsample=True)),
        ("h25g_ffn_all_binary_no_downsample", h9a_scope_groups("binary", downsample=False)),
    ]
    for name, groups in h25_specs:
        cfg = common(base, name)
        set_alpha_xnor_shiftmax(cfg)
        set_qk_atlif(cfg, target_rate=0.040, target_rate_eta=0.05, activity_eta=2.5)
        cfg["optimizer"]["lr"] = 1.0e-5
        cfg["atlif_ternary_psn"]["target_groups"] = groups
        cfg["note"] = (
            "H25 模块排列组合：Q/K 固定三值 ATLIF + alpha-XNOR Shiftmax；"
            "只改变 FFN 升维/降维、二值/三值、downsample 是否替换。"
        )
        write_config(f"{name}_guard120.yml", cfg)


if __name__ == "__main__":
    main()
