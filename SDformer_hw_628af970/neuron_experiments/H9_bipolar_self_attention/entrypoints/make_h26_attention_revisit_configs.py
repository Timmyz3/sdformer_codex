"""生成 H26 注意力回收短测配置。

H26 的目的不是再随机开坑，而是把前面被降级的注意力重新放回
H9a 低 SOPs 替换范围里，配合低学习率、ATLIF 稀疏反馈和不同三值方案
重新短测。这样可以区分“注意力机制本身不行”和“当时超参/三值组合不合适”。
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
    cfg.setdefault("optimizer", {})["lr"] = 1.0e-5
    cfg["optimizer"]["use_amp"] = True
    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    cfg.setdefault("test", {})["sample"] = 10
    return cfg


def group(
    name: str,
    paths: list[str],
    output_mode: str = "binary",
    *,
    activity_eta: float = 0.02,
) -> dict[str, Any]:
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


def h9a_groups(ffn_mode: str = "binary", *, downsample: bool = True) -> list[dict[str, Any]]:
    groups = [
        group("stage0_ffn", STAGE0_SN1 + STAGE0_SN2, output_mode=ffn_mode, activity_eta=0.02),
        group("stage3_block0_ffn", STAGE3B0_SN1 + STAGE3B0_SN2, output_mode=ffn_mode, activity_eta=0.006),
    ]
    if downsample:
        groups.append(group("downsample_stage0_stage2", DOWN02, output_mode="binary", activity_eta=0.02))
    return groups


def set_sparse_qk(
    cfg: dict[str, Any],
    *,
    target_rate: float = 0.040,
    target_rate_eta: float = 0.05,
    activity_eta: float = 2.5,
    ffn_mode: str = "binary",
    downsample: bool = True,
) -> None:
    atlif = cfg.setdefault("atlif_ternary_psn", {})
    atlif.update(
        {
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
            "target_rate_eta": target_rate_eta,
            "trainable": "all",
            "log_interval_steps": 20,
            "target_groups": h9a_groups(ffn_mode, downsample=downsample),
        }
    )


def set_attention(
    cfg: dict[str, Any],
    mode: str,
    *,
    value_mode: str = "threshold",
    score_scale: float = 1.0,
    alpha0: float | None = None,
    mismatch_penalty: float | None = None,
    preserve_mean: bool = True,
) -> None:
    attn = {
        "enabled": True,
        "stage_selection": "all",
        "mode": mode,
        "score_scale": score_scale,
        "center_scores": True,
        "preserve_mean": preserve_mean,
        "eps": 1.0e-6,
        "consensus_score_norm": "head_dim",
        "value_mode": value_mode,
    }
    if alpha0 is not None:
        attn["alpha0"] = alpha0
    if mismatch_penalty is not None:
        attn["mismatch_penalty"] = mismatch_penalty
    cfg["bsa_attention"] = attn


def main() -> None:
    base = load_config("h9a_shiftmax_compat_h8m_speed_bs14.yml")

    specs = [
        {
            "name": "h26a_axnor_l1_sparse040",
            "mode": "alpha_xnor_matrix_l1",
            "value_mode": "threshold",
            "score_scale": 1.0,
            "target_rate": 0.040,
            "activity_eta": 2.5,
            "note": "回收 H18d：alpha-XNOR + L1 归一，H9a 替换范围，低学习率和中等稀疏反馈。",
        },
        {
            "name": "h26b_a2os2a_sparse040",
            "mode": "a2os2a_direct",
            "value_mode": "threshold",
            "score_scale": 1.0,
            "target_rate": 0.040,
            "activity_eta": 2.5,
            "note": "回收 H18e：A2OS2A 风格直接注意力，测试论文范式注意力是否可替代 QK carrier。",
        },
        {
            "name": "h26c_hamming_ternary_sparse040",
            "mode": "hamming_ternary_active_direct",
            "value_mode": "threshold",
            "score_scale": 2.0,
            "target_rate": 0.040,
            "activity_eta": 2.5,
            "note": "回收 H21b：三值 active Hamming 注意力，配合更稳低 LR/稀疏反馈。",
        },
        {
            "name": "h26d_hamming_binary_sparse040",
            "mode": "hamming_binary_direct",
            "value_mode": "threshold",
            "score_scale": 2.0,
            "target_rate": 0.040,
            "activity_eta": 2.5,
            "note": "回收 H21a：二值 Hamming 注意力，作为更硬件友好的降级注意力候选。",
        },
        {
            "name": "h26e_axnor_shiftmax_signv_sparse040",
            "mode": "alpha_xnor_matrix_shiftmax",
            "value_mode": "sign",
            "score_scale": 1.0,
            "alpha0": 0.02,
            "mismatch_penalty": 0.25,
            "target_rate": 0.040,
            "activity_eta": 2.5,
            "note": "H18c/H22j 分支：Shiftmax gate 保留，但 value 改成 sign，降低阈值实数乘法影响。",
        },
        {
            "name": "h26f_axnor_l1_ffn_ternary",
            "mode": "alpha_xnor_matrix_l1",
            "value_mode": "threshold",
            "score_scale": 1.0,
            "target_rate": 0.040,
            "activity_eta": 2.5,
            "ffn_mode": "ternary",
            "note": "H18d + FFN 三值替换，测试三值方案换到高 SOPs FFN 后是否额外降 SOPs。",
        },
        {
            "name": "h26g_a2os2a_ffn_sn1_ternary",
            "mode": "a2os2a_direct",
            "value_mode": "threshold",
            "score_scale": 1.0,
            "target_rate": 0.040,
            "activity_eta": 2.5,
            "custom_groups": [
                group("ffn_sn1_ternary", STAGE0_SN1 + STAGE3B0_SN1, "ternary", activity_eta=0.02),
                group("ffn_sn2_binary", STAGE0_SN2 + STAGE3B0_SN2, "binary", activity_eta=0.02),
                group("downsample_stage0_stage2", DOWN02, "binary", activity_eta=0.02),
            ],
            "note": "A2OS2A + FFN 升维三值/降维二值，检查表达和稀疏的折中。",
        },
        {
            "name": "h26h_hamming_ternary_sparse035",
            "mode": "hamming_ternary_active_direct",
            "value_mode": "threshold",
            "score_scale": 2.0,
            "target_rate": 0.035,
            "target_rate_eta": 0.08,
            "activity_eta": 3.0,
            "note": "H21b 加强稀疏版本，验证 Hamming 注意力是否靠更强 ATLIF 反馈降到 H9a SOPs 附近。",
        },
        {
            "name": "h26i_axnor_l1_flowreg0003",
            "mode": "alpha_xnor_matrix_l1",
            "value_mode": "threshold",
            "score_scale": 1.0,
            "target_rate": 0.040,
            "activity_eta": 2.5,
            "flow_regul_weight": 0.0003,
            "note": "H18d + 降低 flow regularization，检查 AAE 是否受正则项牵制。",
        },
    ]

    for spec in specs:
        cfg = common(base, spec["name"])
        set_attention(
            cfg,
            spec["mode"],
            value_mode=spec.get("value_mode", "threshold"),
            score_scale=spec.get("score_scale", 1.0),
            alpha0=spec.get("alpha0"),
            mismatch_penalty=spec.get("mismatch_penalty"),
            preserve_mean=spec.get("preserve_mean", True),
        )
        set_sparse_qk(
            cfg,
            target_rate=spec.get("target_rate", 0.040),
            target_rate_eta=spec.get("target_rate_eta", 0.05),
            activity_eta=spec.get("activity_eta", 2.5),
            ffn_mode=spec.get("ffn_mode", "binary"),
            downsample=spec.get("downsample", True),
        )
        if "custom_groups" in spec:
            cfg["atlif_ternary_psn"]["target_groups"] = spec["custom_groups"]
        if "flow_regul_weight" in spec:
            cfg.setdefault("loss", {})["flow_regul_weight"] = spec["flow_regul_weight"]
        cfg["note"] = spec["note"]
        write_config(f"{spec['name']}_guard120.yml", cfg)


if __name__ == "__main__":
    main()
