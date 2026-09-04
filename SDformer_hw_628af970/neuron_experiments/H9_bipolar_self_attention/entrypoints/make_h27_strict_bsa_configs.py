"""生成 H27 标准 BSA 范式复测配置。

H14 已经实现过 strict BSA，但当时使用 H13n 的替换范围。H27 把标准
BSA 矩阵注意力放回当前主线的 H9a 低 SOPs 替换范围，并扫 value/norm/稀疏
强度，避免因为旧替换范围或超参把 BSA 范式误判为不可用。
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


def group(name: str, paths: list[str], *, activity_eta: float) -> dict[str, Any]:
    return {
        "name": name,
        "output_mode": "binary",
        "threshold_eta": 8.0e-5,
        "threshold_lr_scale": 8000.0,
        "max_threshold": 0.105,
        "activity_eta": activity_eta,
        "paths": paths,
    }


STAGE0_FFN = [
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn1",
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn2",
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.mlp.sn1",
    "sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.mlp.sn2",
]
STAGE3B0_FFN = [
    "sttmultires_unet.encoders.swin3d.layers.3.swin_blocks.0.mlp.sn1",
    "sttmultires_unet.encoders.swin3d.layers.3.swin_blocks.0.mlp.sn2",
]
DOWN02 = [
    "sttmultires_unet.encoders.swin3d.layers.0.downsample.sn",
    "sttmultires_unet.encoders.swin3d.layers.2.downsample.sn",
]


def make_config(
    base: dict[str, Any],
    name: str,
    *,
    value_mode: str,
    norm: str,
    score_scale: float,
    target_rate: float,
    target_eta: float,
    activity_eta: float,
) -> dict[str, Any]:
    cfg = deepcopy(base)
    cfg["experiment"] = name
    cfg.setdefault("runtime", {})["max_train_steps"] = 120
    cfg["runtime"]["skip_state_save"] = True
    cfg.setdefault("loader", {})["n_epochs"] = 1
    cfg["loader"]["batch_size"] = 8
    cfg["loader"]["n_workers"] = 8
    cfg["loader"]["pin_memory"] = False
    cfg.setdefault("optimizer", {})["lr"] = 1.0e-5
    cfg["optimizer"]["use_amp"] = True
    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    cfg.setdefault("test", {})["sample"] = 10

    cfg["bsa_attention"] = {
        "enabled": True,
        "stage_selection": "all",
        "mode": "strict_bsa_shiftmax",
        "score_scale": score_scale,
        "center_scores": True,
        "preserve_mean": False,
        "eps": 1.0e-6,
        "consensus_score_norm": norm,
        "value_mode": value_mode,
    }
    cfg["atlif_ternary_psn"] = {
        "enabled": True,
        "target": "qk",
        "stage_selection": "all",
        "output_mode": "ternary",
        "threshold_init": 0.1,
        "threshold_eta": 0.001,
        "threshold_lr_scale": 50000.0,
        "min_threshold": 0.001,
        "max_threshold": 0.13,
        "negative_threshold_scale": 30.0,
        "activity_eta": activity_eta,
        "target_rate": target_rate,
        "target_rate_eta": target_eta,
        "trainable": "all",
        "log_interval_steps": 20,
        "target_groups": [
            group("stage0_ffn_binary", STAGE0_FFN, activity_eta=0.02),
            group("stage3_block0_ffn_binary", STAGE3B0_FFN, activity_eta=0.006),
            group("downsample_stage0_stage2_binary", DOWN02, activity_eta=0.02),
        ],
    }
    cfg["note"] = (
        "H27 标准 BSA：sign(Q) @ sign(K)^T -> Shiftmax -> @ V；"
        "由于 baseline 是 QKFormer no-V block，V 由 K 复用。"
        f"value_mode={value_mode}, norm={norm}, score_scale={score_scale}, "
        f"target_rate={target_rate}。"
    )
    return cfg


def main() -> None:
    base = load_config("h9a_shiftmax_compat_h8m_speed_bs14.yml")
    specs = [
        ("h27a_strict_bsa_signv_sqrt_sparse040", "sign", "sqrt_head_dim", 1.0, 0.040, 0.05, 2.5),
        ("h27b_strict_bsa_thetav_sqrt_sparse040", "threshold", "sqrt_head_dim", 1.0, 0.040, 0.05, 2.5),
        ("h27c_strict_bsa_signv_head_sparse040", "sign", "head_dim", 2.0, 0.040, 0.05, 2.5),
        ("h27d_strict_bsa_thetav_head_sparse040", "threshold", "head_dim", 2.0, 0.040, 0.05, 2.5),
        ("h27e_strict_bsa_signv_active_sparse040", "sign", "active", 1.0, 0.040, 0.05, 2.5),
        ("h27f_strict_bsa_signv_sqrt_sparse035", "sign", "sqrt_head_dim", 1.0, 0.035, 0.08, 3.0),
    ]
    for name, value_mode, norm, score_scale, target_rate, target_eta, activity_eta in specs:
        write_config(
            f"{name}_guard120.yml",
            make_config(
                base,
                name,
                value_mode=value_mode,
                norm=norm,
                score_scale=score_scale,
                target_rate=target_rate,
                target_eta=target_eta,
                activity_eta=activity_eta,
            ),
        )


if __name__ == "__main__":
    main()
